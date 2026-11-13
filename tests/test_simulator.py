import unittest

from simulator import Access, SimulationStats, SimulatorConfig, simulate


class SimulatorTests(unittest.TestCase):
    def test_empty_trace(self):
        self.assertEqual(simulate([], prefetch_plan={0: 99}), SimulationStats())

    def test_first_access_misses_and_waits_for_a_walk(self):
        stats = simulate([Access(0, pc=25, compute_cycles=7)])
        self.assertEqual(
            stats,
            SimulationStats(
                demand_accesses=1,
                demand_tlb_misses=1,
                demand_walks_issued=1,
                demand_wait_cycles=60,
                elapsed_cycles=67,
            ),
        )

    def test_repeated_page_hits_across_instruction_ids(self):
        stats = simulate([Access(7, pc=1), Access(7, pc=2, compute_cycles=3)])
        self.assertEqual(stats.demand_tlb_hits, 1)
        self.assertEqual(stats.demand_tlb_misses, 1)
        self.assertEqual(stats.demand_walks_issued, 1)
        self.assertEqual(stats.demand_wait_cycles, 60)
        self.assertEqual(stats.elapsed_cycles, 63)

    def test_two_cold_pages_wait_120_cycles(self):
        stats = simulate([Access(1), Access(2)])
        self.assertEqual(stats.demand_tlb_misses, 2)
        self.assertEqual(stats.demand_walks_issued, 2)
        self.assertEqual(stats.demand_wait_cycles, 120)
        self.assertEqual(stats.elapsed_cycles, 120)

    def test_tlb_hit_updates_lru_order(self):
        config = SimulatorConfig(tlb_capacity=2)
        stats = simulate(map(Access, [1, 2, 1, 3, 1, 2]), config)
        self.assertEqual(stats.demand_tlb_hits, 2)
        self.assertEqual(stats.demand_tlb_misses, 4)
        self.assertEqual(stats.demand_walks_issued, 4)
        self.assertEqual(stats.demand_wait_cycles, 240)

    def test_zero_latency_completes_prefetches_on_issue(self):
        config = SimulatorConfig(page_walk_latency=0, max_pending_prefetches=1)
        stats = simulate(map(Access, [1, 2]), config, {0: 2, 1: 3})
        self.assertEqual(stats.demand_walks_issued, 1)
        self.assertEqual(stats.demand_wait_cycles, 0)
        self.assertEqual(stats.elapsed_cycles, 0)
        self.assertEqual(stats.timely_prefetches_consumed, 1)
        self.assertEqual(stats.late_prefetches_consumed, 0)
        self.assertEqual(stats.prefetch_walks_issued, 2)
        self.assertEqual(stats.unused_prefetches, 1)
        self.assertEqual(stats.pending_prefetches, 0)

    def test_timely_prefetch_is_a_tlb_miss_then_installs_as_most_recent(self):
        trace = [Access(1), Access(2), Access(3, compute_cycles=61), Access(4), Access(3)]
        config = SimulatorConfig(tlb_capacity=2)
        stats = simulate(trace, config, {1: 3, 2: 3})
        self.assertEqual(stats.demand_tlb_misses, 4)
        self.assertEqual(stats.demand_tlb_hits, 1)
        self.assertEqual(stats.prefetch_buffer_hits, 1)
        self.assertEqual(stats.timely_prefetches_consumed, 1)
        self.assertEqual(stats.demand_walks_issued, 3)
        self.assertEqual(stats.demand_wait_cycles, 180)
        self.assertEqual(stats.elapsed_cycles, 241)
        self.assertEqual(stats.prefetch_duplicate_rejections, 1)
        self.assertEqual(stats.prefetch_walks_issued, 1)
        self.assertEqual(stats.unused_prefetches, 0)
        self.assertEqual(stats.pending_prefetches, 0)

    def test_late_prefetch_waits_40_cycles_without_a_duplicate_walk(self):
        initial_access = Access(100, compute_cycles=40)
        plan = {0: 200}
        first = simulate([initial_access], prefetch_plan=plan)
        stats = simulate([initial_access, Access(200, compute_cycles=20)], prefetch_plan=plan)
        self.assertEqual(first.elapsed_cycles, 100)
        self.assertEqual(first.pending_prefetches, 1)
        self.assertEqual(stats.elapsed_cycles, 160)
        self.assertEqual(stats.demand_wait_cycles - first.demand_wait_cycles, 40)
        self.assertEqual(
            stats,
            SimulationStats(
                demand_accesses=2,
                demand_tlb_misses=2,
                demands_joined_prefetches=1,
                demand_walks_issued=1,
                demand_wait_cycles=100,
                elapsed_cycles=160,
                prefetch_requests_attempted=1,
                prefetch_walks_issued=1,
            ),
        )

    def test_completion_at_demand_issue_is_timely(self):
        stats = simulate([Access(1), Access(2, compute_cycles=60)], prefetch_plan={0: 2})
        self.assertEqual(stats.demand_wait_cycles, 60)
        self.assertEqual(stats.elapsed_cycles, 120)
        self.assertEqual(stats.prefetch_buffer_hits, 1)
        self.assertEqual(stats.demands_joined_prefetches, 0)
        self.assertEqual(stats.unused_prefetches, 0)
        self.assertEqual(stats.pending_prefetches, 0)

    def test_joined_prefetch_goes_directly_to_tlb_without_evicting_buffer_entry(self):
        trace = [Access(1), Access(1), Access(20, compute_cycles=20), Access(10), Access(20)]
        config = SimulatorConfig(prefetch_buffer_capacity=1)
        stats = simulate(trace, config, {0: 10, 1: 20})
        self.assertEqual(stats.late_prefetches_consumed, 1)
        self.assertEqual(stats.timely_prefetches_consumed, 1)
        self.assertEqual(stats.demand_tlb_hits, 2)
        self.assertEqual(stats.demand_walks_issued, 1)
        self.assertEqual(stats.demand_wait_cycles, 100)
        self.assertEqual(stats.prefetch_evictions, 0)
        self.assertEqual(stats.unused_prefetches, 0)
        self.assertEqual(stats.pending_prefetches, 0)

    def test_trace_end_reports_unused_and_pending_work_without_draining(self):
        stats = simulate([Access(1), Access(1, compute_cycles=60)], prefetch_plan={0: 2, 1: 3})
        self.assertEqual(stats.elapsed_cycles, 120)
        self.assertEqual(stats.prefetch_walks_issued, 2)
        self.assertEqual(stats.timely_prefetches_consumed, 0)
        self.assertEqual(stats.late_prefetches_consumed, 0)
        self.assertEqual(stats.prefetch_evictions, 0)
        self.assertEqual(stats.unused_prefetches, 1)
        self.assertEqual(stats.pending_prefetches, 1)

    def test_prefetch_buffer_evicts_oldest_on_completion(self):
        trace = [
            Access(1), Access(1, compute_cycles=1), Access(1, compute_cycles=1),
            Access(30, compute_cycles=60), Access(20), Access(40),
        ]
        config = SimulatorConfig(prefetch_buffer_capacity=2)
        plan = {0: 40, 1: 30, 2: 20}
        self.assertEqual(simulate(trace[:-1], config, plan).timely_prefetches_consumed, 2)
        stats = simulate(trace, config, plan)
        self.assertEqual(stats.prefetch_evictions, 1)
        self.assertEqual(stats.timely_prefetches_consumed, 2)
        self.assertEqual(stats.demand_walks_issued, 2)
        self.assertEqual(stats.demand_wait_cycles, 120)
        self.assertEqual(stats.unused_prefetches, 0)

    def test_duplicate_requests_are_rejected_in_all_three_locations(self):
        trace = [Access(1), Access(1), Access(1, compute_cycles=60), Access(1)]
        config = SimulatorConfig(max_pending_prefetches=1)
        stats = simulate(trace, config, {0: 2, 1: 2, 2: 2, 3: 1})
        self.assertEqual(stats.prefetch_requests_attempted, 4)
        self.assertEqual(stats.prefetch_walks_issued, 1)
        self.assertEqual(stats.prefetch_duplicate_rejections, 3)
        self.assertEqual(stats.prefetch_capacity_rejections, 0)
        self.assertEqual(stats.unused_prefetches, 1)

    def test_pending_limit_rejects_new_walk_and_completion_frees_slot(self):
        trace = [Access(1), Access(1), Access(1, compute_cycles=60), Access(3)]
        config = SimulatorConfig(max_pending_prefetches=1)
        stats = simulate(trace, config, {0: 2, 1: 3, 2: 4})
        self.assertEqual(stats.prefetch_requests_attempted, 3)
        self.assertEqual(stats.prefetch_walks_issued, 2)
        self.assertEqual(stats.prefetch_capacity_rejections, 1)
        self.assertEqual(stats.prefetch_duplicate_rejections, 0)
        self.assertEqual(stats.demand_walks_issued, 2)
        self.assertEqual(stats.demand_wait_cycles, 120)
        self.assertEqual(stats.unused_prefetches, 2)
        self.assertEqual(stats.pending_prefetches, 0)

    def test_distinct_completions_during_demand_walk_are_processed_in_time_order(self):
        trace = [Access(1), Access(1, compute_cycles=10), Access(2, compute_cycles=10), Access(20)]
        config = SimulatorConfig(prefetch_buffer_capacity=1)
        stats = simulate(trace, config, {0: 90, 1: 20})
        self.assertEqual(stats.prefetch_evictions, 1)
        self.assertEqual(stats.timely_prefetches_consumed, 1)
        self.assertEqual(stats.demand_walks_issued, 2)
        self.assertEqual(stats.demand_wait_cycles, 120)
        self.assertEqual(stats.elapsed_cycles, 140)

    def test_simultaneous_completions_use_issue_order(self):
        trace = [Access(1), Access(1), Access(20, compute_cycles=60)]
        config = SimulatorConfig(prefetch_buffer_capacity=1)
        stats = simulate(trace, config, {1: 20, 0: 90})
        self.assertEqual(stats.prefetch_evictions, 1)
        self.assertEqual(stats.timely_prefetches_consumed, 1)
        self.assertEqual(stats.demand_walks_issued, 1)
        self.assertEqual(stats.demand_wait_cycles, 60)
        self.assertEqual(stats.elapsed_cycles, 120)

    def test_zero_buffer_capacity_still_allows_a_late_join(self):
        trace = [Access(1), Access(2, compute_cycles=20), Access(3, compute_cycles=60)]
        config = SimulatorConfig(prefetch_buffer_capacity=0)
        stats = simulate(trace, config, {0: 2, 1: 3})
        self.assertEqual(stats.late_prefetches_consumed, 1)
        self.assertEqual(stats.timely_prefetches_consumed, 0)
        self.assertEqual(stats.prefetch_evictions, 1)
        self.assertEqual(stats.unused_prefetches, 0)
        self.assertEqual(stats.demand_walks_issued, 2)
        self.assertEqual(stats.demand_wait_cycles, 160)

    def test_zero_pending_capacity_disables_prefetch_walks(self):
        config = SimulatorConfig(max_pending_prefetches=0)
        stats = simulate([Access(1), Access(2)], config, {0: 2, 1: None})
        self.assertEqual(stats.prefetch_requests_attempted, 1)
        self.assertEqual(stats.prefetch_capacity_rejections, 1)
        self.assertEqual(stats.prefetch_walks_issued, 0)
        self.assertEqual(stats.demand_walks_issued, 2)
        self.assertEqual(stats.demand_wait_cycles, 120)

    def test_accounting_and_repeatability(self):
        pages = [1, 1, 1, 1, 1, 3, 4, 3, 3, 3, 3]
        gaps = [0, 0, 0, 60, 0, 0, 10, 0, 0, 0, 0]
        trace = [Access(vpn, compute_cycles=gap) for vpn, gap in zip(pages, gaps)]
        config = SimulatorConfig(
            tlb_capacity=2, prefetch_buffer_capacity=1, max_pending_prefetches=2,
        )
        plan = {0: 2, 1: 3, 2: 9, 3: 3, 5: 4, 6: 5, 7: 6, 8: None}

        stats = simulate(trace, config, plan)
        self.assertEqual(stats, simulate(iter(trace), config, dict(reversed(list(plan.items())))))
        self.assertEqual(
            stats,
            SimulationStats(
                demand_accesses=11,
                demand_tlb_hits=8,
                demand_tlb_misses=3,
                prefetch_buffer_hits=1,
                demands_joined_prefetches=1,
                demand_walks_issued=1,
                demand_wait_cycles=110,
                elapsed_cycles=180,
                prefetch_requests_attempted=7,
                prefetch_walks_issued=5,
                prefetch_duplicate_rejections=1,
                prefetch_capacity_rejections=1,
                prefetch_evictions=1,
                pending_prefetches=2,
            ),
        )
        self.assertEqual(stats.demand_accesses, stats.demand_tlb_hits + stats.demand_tlb_misses)
        self.assertEqual(
            stats.demand_tlb_misses,
            stats.prefetch_buffer_hits + stats.demands_joined_prefetches + stats.demand_walks_issued,
        )
        self.assertEqual(
            stats.prefetch_requests_attempted,
            stats.prefetch_walks_issued + stats.prefetch_requests_rejected,
        )
        self.assertEqual(
            stats.extra_prefetch_walks,
            stats.timely_prefetches_consumed + stats.late_prefetches_consumed
            + stats.prefetch_evictions + stats.unused_prefetches + stats.pending_prefetches,
        )
        self.assertEqual(stats.elapsed_cycles, sum(gaps) + stats.demand_wait_cycles)

    def test_invalid_configuration_is_rejected(self):
        invalid_settings = {
            "page_size": 0,
            "tlb_capacity": 0,
            "prefetch_buffer_capacity": -1,
            "page_walk_latency": 0.5,
            "max_pending_prefetches": True,
        }
        for field, value in invalid_settings.items():
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, field):
                SimulatorConfig(**{field: value})

    def test_invalid_access_and_prefetch_plan_are_rejected(self):
        for field in ("vpn", "pc", "compute_cycles"):
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, field):
                Access(**({"vpn": 0} | {field: -1}))
        with self.assertRaisesRegex(ValueError, "plan index"):
            simulate([Access(1)], prefetch_plan={-1: 2})
        with self.assertRaisesRegex(ValueError, "target vpn"):
            simulate([Access(1)], prefetch_plan={0: -2})


if __name__ == "__main__":
    unittest.main()
