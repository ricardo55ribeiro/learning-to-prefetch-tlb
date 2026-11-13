import unittest
from unittest.mock import patch

from compare import COMPARISON_CONFIG, compare, make_workloads
from simulator import Access, SimulatorConfig, simulate


class ComparisonTests(unittest.TestCase):
    def test_all_four_policies_use_the_same_trace_and_config(self):
        trace = (Access(1), Access(2, compute_cycles=1), Access(3))
        config = SimulatorConfig(tlb_capacity=2, page_walk_latency=7, max_pending_prefetches=0)
        with patch("compare.simulate", wraps=simulate) as run:
            results = compare(trace, config)

        self.assertEqual(len(results), 4)
        self.assertEqual(run.call_count, 4)
        for call, result in zip(run.call_args_list, results):
            self.assertIs(call.args[0], trace)
            self.assertIs(call.args[1], config)
            self.assertEqual(result.stats.demand_wait_cycles, 21)
            self.assertEqual(result.stats.elapsed_cycles, 22)
            self.assertEqual(result.total_walks, 3)
        self.assertEqual(
            [result.stats.prefetch_capacity_rejections for result in results],
            [0, 3, 1, 0],
        )

    def test_fixed_seed_reproduces_traces_and_results(self):
        workloads = make_workloads(42)
        repeated = make_workloads(42)
        different = make_workloads(43)
        self.assertEqual(workloads, repeated)
        self.assertNotEqual(workloads["Random"], different["Random"])
        for name, trace in workloads.items():
            with self.subTest(workload=name):
                self.assertEqual(compare(trace), compare(repeated[name]))
                if name != "Random":
                    self.assertEqual(trace, different[name])

    def test_all_workload_counters_reconcile(self):
        for workload, trace in make_workloads().items():
            results = compare(trace)
            baseline = results[0].stats
            for result in results:
                with self.subTest(workload=workload, policy=result.name):
                    stats = result.stats
                    self.assertEqual(stats.demand_accesses, len(trace))
                    self.assertEqual(stats.demand_tlb_hits, baseline.demand_tlb_hits)
                    self.assertEqual(stats.demand_tlb_misses, baseline.demand_tlb_misses)
                    timely = stats.timely_prefetches_consumed
                    late = stats.late_prefetches_consumed
                    self.assertEqual(
                        stats.demand_tlb_misses, stats.demand_walks_issued + timely + late,
                    )
                    self.assertEqual(
                        stats.prefetch_walks_issued, timely + late + result.wasted_or_unused,
                    )
                    self.assertEqual(
                        stats.prefetch_requests_attempted,
                        stats.prefetch_walks_issued + stats.prefetch_requests_rejected,
                    )
                    self.assertEqual(
                        result.total_walks - baseline.demand_walks_issued, result.wasted_or_unused,
                    )
                    self.assertLessEqual(
                        stats.unused_prefetches, COMPARISON_CONFIG.prefetch_buffer_capacity,
                    )
                    self.assertLessEqual(
                        stats.pending_prefetches, COMPARISON_CONFIG.max_pending_prefetches,
                    )
                    self.assertEqual(
                        stats.elapsed_cycles,
                        stats.demand_wait_cycles + sum(access.compute_cycles for access in trace),
                    )


if __name__ == "__main__":
    unittest.main()
