import unittest
from unittest.mock import Mock

from predictors import MarkovPrefetcher, NextPagePrefetcher, PerPCStridePrefetcher
from simulator import Access, SimulatorConfig, simulate


class PredictorTests(unittest.TestCase):
    def test_next_page_predicts_vpn_plus_one(self):
        predictor = NextPagePrefetcher()
        self.assertEqual(predictor.observe(Access(0)), 1)
        self.assertEqual(predictor.observe(Access(42, pc=7)), 43)
        self.assertEqual(predictor.metadata_entries, 0)

    def test_every_policy_can_abstain(self):
        cases = (
            NextPagePrefetcher(max_target_vpn=10), PerPCStridePrefetcher(), MarkovPrefetcher(),
        )
        for predictor in cases:
            with self.subTest(predictor=type(predictor).__name__):
                self.assertIsNone(predictor.observe(Access(10)))
                stats = simulate([Access(10)], predictor=predictor)
                self.assertEqual(stats.prefetch_requests_attempted, 0)
        self.assertEqual(cases[0].observe(Access(9)), 10)

    def test_stride_needs_two_matching_strides_per_pc(self):
        predictor = PerPCStridePrefetcher()
        trace = [
            Access(100, 1), Access(40, 2), Access(107, 1), Access(36, 2),
            Access(114, 1), Access(32, 2), Access(121, 1),
        ]
        self.assertEqual(
            [predictor.observe(access) for access in trace],
            [None, None, None, None, 121, 28, 128],
        )

    def test_stride_change_requires_confirmation(self):
        predictor = PerPCStridePrefetcher()
        pages = [100, 107, 114, 117, 120, 120, 120, 119, 118]
        self.assertEqual(
            [predictor.observe(Access(vpn)) for vpn in pages],
            [None, None, 121, None, 123, None, None, None, 117],
        )

    def test_stride_drops_negative_target(self):
        predictor = PerPCStridePrefetcher()
        self.assertEqual(
            [predictor.observe(Access(vpn)) for vpn in (13, 8, 3)],
            [None, None, None],
        )

    def test_stride_evicts_least_recently_seen_pc(self):
        predictor = PerPCStridePrefetcher(max_pcs=2)
        trace = [
            Access(0, 1), Access(20, 2), Access(1, 1), Access(22, 2),
            Access(2, 1), Access(40, 3), Access(24, 2),
        ]
        self.assertEqual(
            [predictor.observe(access) for access in trace],
            [None, None, None, None, 3, None, None],
        )
        self.assertEqual(predictor.metadata_entries, 2)

    def test_markov_learns_observed_successors_and_resolves_ties_by_vpn(self):
        predictor = MarkovPrefetcher()
        pages = [10, 42, 10, 7, 10, 42, 10]
        self.assertEqual(
            [predictor.observe(Access(vpn)) for vpn in pages],
            [None, None, 42, None, 7, 10, 42],
        )
        self.assertEqual(predictor.metadata_entries, 8)

    def test_markov_records_incoming_self_transition_before_predicting(self):
        predictor = MarkovPrefetcher()
        self.assertIsNone(predictor.observe(Access(0)))
        self.assertEqual(predictor.observe(Access(0)), 0)

    def test_markov_evicts_oldest_successor_at_capacity(self):
        predictor = MarkovPrefetcher(max_sources=4, max_successors=2)
        pages = [1, 2, 1, 2, 1, 8, 1, 7, 1]
        self.assertEqual(
            [predictor.observe(Access(vpn)) for vpn in pages],
            [None, None, 2, 1, 2, None, 2, None, 7],
        )
        self.assertEqual(predictor.metadata_entries, 10)

    def test_markov_evicts_oldest_source_at_capacity(self):
        predictor = MarkovPrefetcher(max_sources=2, max_successors=2)
        pages = [1, 2, 1, 2, 3, 1]
        self.assertEqual(
            [predictor.observe(Access(vpn)) for vpn in pages],
            [None, None, 2, 1, None, None],
        )
        self.assertEqual(predictor.metadata_entries, 6)

    def test_markov_counters_saturate(self):
        predictor = MarkovPrefetcher()
        predictor.observe(Access(1))
        for _ in range(260):
            predictor.observe(Access(9))
            predictor.observe(Access(1))
        for _ in range(254):
            predictor.observe(Access(2))
            prediction = predictor.observe(Access(1))
        self.assertEqual(prediction, 9)
        predictor.observe(Access(2))
        self.assertEqual(predictor.observe(Access(1)), 2)

    def test_invalid_predictor_limits_fail_clearly(self):
        with self.assertRaisesRegex(ValueError, "max_target_vpn"):
            NextPagePrefetcher(max_target_vpn=-1)
        with self.assertRaisesRegex(ValueError, "max_pcs"):
            PerPCStridePrefetcher(max_pcs=0)
        with self.assertRaisesRegex(ValueError, "max_sources"):
            MarkovPrefetcher(max_sources=True)
        with self.assertRaisesRegex(ValueError, "max_successors"):
            MarkovPrefetcher(max_successors=-1)


class PredictorIntegrationTests(unittest.TestCase):
    def test_observes_each_demand_before_trace_can_yield_the_next_one(self):
        accesses = (Access(5), Access(8, 1), Access(8, 2))
        predictor = Mock()
        predictor.observe.side_effect = lambda access: access.vpn

        def trace():
            predictor.reset.assert_called_once_with()
            for index, access in enumerate(accesses, start=1):
                yield access
                self.assertEqual(predictor.observe.call_count, index)
                predictor.observe.assert_called_with(access)

        stats = simulate(trace(), predictor=predictor)
        self.assertEqual(stats.demand_tlb_hits, 1)
        self.assertEqual(stats.prefetch_duplicate_rejections, 3)
        self.assertEqual(stats.prefetch_walks_issued, 0)

    def test_reusing_predictors_resets_history_per_run(self):
        cases = (
            (PerPCStridePrefetcher(), [1, 2, 3], [4]),
            (MarkovPrefetcher(), [10, 42, 10], [10]),
        )
        for predictor, warmup, probe in cases:
            with self.subTest(predictor=type(predictor).__name__):
                warmup_stats = simulate(map(Access, warmup), predictor=predictor)
                self.assertGreater(warmup_stats.prefetch_requests_attempted, 0)
                self.assertGreater(predictor.metadata_entries, 0)
                stats = simulate(map(Access, probe), predictor=predictor)
                self.assertEqual(stats, simulate(map(Access, probe)))
                simulate([], predictor=predictor)
                self.assertEqual(predictor.metadata_entries, 0)

    def test_simulator_rejects_predictor_targets_when_pending_is_full(self):
        trace = [Access(10), Access(1), Access(10), Access(2, compute_cycles=20)]
        config = SimulatorConfig(max_pending_prefetches=1)
        predictor = Mock()
        predictor.observe.side_effect = [None, 2, 11, 11]
        stats = simulate(trace, config, predictor=predictor)
        self.assertEqual(stats.prefetch_requests_attempted, 3)
        self.assertEqual(stats.prefetch_walks_issued, 2)
        self.assertEqual(stats.prefetch_duplicate_rejections, 0)
        self.assertEqual(stats.prefetch_capacity_rejections, 1)
        self.assertEqual(stats.demand_wait_cycles, 160)
        self.assertEqual(stats.late_prefetches_consumed, 1)
        self.assertEqual(stats.pending_prefetches, 1)

    def test_correct_late_prediction_waits_without_changing_tlb_misses(self):
        trace = [Access(100), Access(101, compute_cycles=20)]
        baseline = simulate(trace)
        stats = simulate(trace, predictor=NextPagePrefetcher())
        self.assertEqual(baseline.demand_tlb_misses, 2)
        self.assertEqual(stats.demand_tlb_misses, baseline.demand_tlb_misses)
        self.assertEqual(baseline.demand_wait_cycles, 120)
        self.assertEqual(stats.demand_wait_cycles, 100)
        self.assertEqual(stats.demand_walks_issued, 1)
        self.assertEqual(stats.late_prefetches_consumed, 1)
        self.assertEqual(stats.timely_prefetches_consumed, 0)

    def test_script_behaves_like_equivalent_predictor(self):
        trace = [Access(1), Access(2, compute_cycles=60), Access(3, compute_cycles=20)]
        scripted = simulate(trace, prefetch_plan={0: 2, 1: 3, 2: 4})
        self.assertEqual(scripted, simulate(trace, predictor=NextPagePrefetcher()))
        self.assertEqual(scripted.timely_prefetches_consumed, 1)
        self.assertEqual(scripted.late_prefetches_consumed, 1)
        self.assertEqual(scripted.pending_prefetches, 1)

    def test_plan_and_predictor_are_mutually_exclusive_even_with_an_empty_plan(self):
        predictor = Mock()
        with self.assertRaisesRegex(ValueError, "either a predictor or a scripted"):
            simulate([], prefetch_plan={}, predictor=predictor)
        predictor.reset.assert_not_called()

    def test_invalid_predictor_target_fails_clearly(self):
        predictor = Mock()
        predictor.observe.return_value = -1
        with self.assertRaisesRegex(ValueError, "predictor target vpn"):
            simulate([Access(1)], predictor=predictor)


if __name__ == "__main__":
    unittest.main()
