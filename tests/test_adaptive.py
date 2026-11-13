import unittest
from unittest.mock import patch

from compare import LONG_SEEDS, compare, make_long_workloads
from predictors import AdaptivePrefetcher
from simulator import Access, PrefetchFeedback, SimulatorConfig, simulate


class AdaptiveTests(unittest.TestCase):
    def test_every_candidate_observes_each_completed_access_and_reset(self):
        with (
            patch("predictors.NextPagePrefetcher") as next_page,
            patch("predictors.PerPCStridePrefetcher") as stride,
            patch("predictors.MarkovPrefetcher") as markov,
        ):
            candidates = (next_page.return_value, stride.return_value, markov.return_value)
            for candidate in candidates:
                candidate.observe.return_value = None
            predictor = AdaptivePrefetcher()

        trace = (Access(8, 1), Access(5, 2), Access(8, 3))
        before = [candidate.reset.call_count for candidate in candidates]
        stats = simulate(trace, predictor=predictor, on_feedback=predictor.on_feedback)
        for index, candidate in enumerate(candidates):
            self.assertEqual(candidate.reset.call_count, before[index] + 1)
            self.assertEqual(
                [call.args for call in candidate.observe.call_args_list],
                [(access,) for access in trace],
            )
        self.assertEqual(stats.prefetch_requests_attempted, 0)

    def test_probe_order_and_delayed_outcome_attribution(self):
        predictor = AdaptivePrefetcher(probe_interval=2)
        pages = (1, 2, 3, 4, 5, 6, 7)
        actual = []
        for index, vpn in enumerate(pages):
            if index == 2:
                predictor.on_feedback(PrefetchFeedback("accepted", 1, 3, 0))
            if index == 4:
                predictor.on_feedback(PrefetchFeedback("accepted", 3, 5, 0))
                predictor.on_feedback(PrefetchFeedback("timely", 1, 3, 60, saved_cycles=60))
            if index == 6:
                predictor.on_feedback(PrefetchFeedback("evicted", 3, 5, 120))
            actual.append(predictor.observe(Access(vpn)))
        self.assertEqual(actual, [None, 3, None, 5, 6, None, 8])

    def test_probe_schedule_shifts_after_each_full_round(self):
        predictor = AdaptivePrefetcher(probe_interval=3)
        targets = [predictor.observe(Access(vpn)) for vpn in range(1, 14)]
        self.assertEqual(targets, [
            None, None, 4, None, None, 7, None, None, None, None, None, None, 14,
        ])

    def test_eviction_and_recent_window_can_restore_abstention(self):
        for outcome, expected in (("evicted", None), ("late", 5)):
            with self.subTest(outcome=outcome):
                predictor = AdaptivePrefetcher(history_window=2, probe_interval=3)
                for vpn in (1, 2):
                    self.assertIsNone(predictor.observe(Access(vpn)))
                self.assertEqual(predictor.observe(Access(3)), 4)
                predictor.on_feedback(PrefetchFeedback("accepted", 2, 4, 60))
                predictor.on_feedback(PrefetchFeedback(
                    outcome, 2, 4, 120, saved_cycles=20 if outcome == "late" else 0,
                ))
                self.assertEqual(predictor.observe(Access(4)), expected)
                self.assertEqual(predictor.observe(Access(5)), 6 if expected else None)
                self.assertEqual(predictor.observe(Access(6)), 7)
                self.assertIsNone(predictor.observe(Access(7)))

    def test_repeated_runs_reset_and_bound_metadata(self):
        trace = make_long_workloads(11)["Loops"]
        predictor = AdaptivePrefetcher(history_window=3)
        config = SimulatorConfig(
            tlb_capacity=16, prefetch_buffer_capacity=2, max_pending_prefetches=2,
        )

        def run():
            events = []

            def receive(event):
                predictor.on_feedback(event)
                events.append(event)
                self.assertLessEqual(predictor.metadata_entries, 16 + 161 + 9 + 4)

            return simulate(trace, config, predictor=predictor, on_feedback=receive), events

        first = run()
        self.assertGreater(first[0].prefetch_walks_issued, 0)
        self.assertEqual(first, run())
        simulate([], predictor=predictor, on_feedback=predictor.on_feedback)
        self.assertEqual(predictor.metadata_entries, 0)

    def test_seeded_results_and_walk_conservation(self):
        self.assertNotEqual(make_long_workloads(11), make_long_workloads(29))
        for seed in LONG_SEEDS:
            for name, trace in make_long_workloads(seed).items():
                with self.subTest(seed=seed, workload=name):
                    results = compare(trace, include_adaptive=True)
                    self.assertEqual(
                        results, compare(make_long_workloads(seed)[name], include_adaptive=True),
                    )
                    self.assertEqual(len(results), 5)
                    baseline = results[0].stats
                    for result in results:
                        stats = result.stats
                        self.assertEqual(stats.demand_tlb_misses, baseline.demand_tlb_misses)
                        self.assertEqual(
                            stats.demand_tlb_misses,
                            stats.demand_walks_issued + stats.timely_prefetches_consumed
                            + stats.late_prefetches_consumed,
                        )
                        self.assertEqual(
                            stats.prefetch_requests_attempted,
                            stats.prefetch_walks_issued + stats.prefetch_requests_rejected,
                        )
                        self.assertEqual(result.additional_walks, result.wasted_or_unused)
                        self.assertEqual(result.adjusted_cost(0), stats.demand_wait_cycles)
                        self.assertEqual(
                            result.adjusted_cost(20) - result.adjusted_cost(5),
                            15 * result.additional_walks,
                        )


if __name__ == "__main__":
    unittest.main()
