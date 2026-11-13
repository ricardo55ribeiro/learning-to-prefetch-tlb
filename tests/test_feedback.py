import unittest

from simulator import Access, PrefetchFeedback, SimulatorConfig, simulate


class FeedbackTests(unittest.TestCase):
    def test_stall_events_precede_prediction_and_retain_simulated_cycles(self):
        log = []
        targets = (10, 20, 30, 40, 41, 50)

        class Predictor:
            def reset(self):
                self.targets = iter(targets)

            def observe(self, access):
                log.append(access)
                return next(self.targets)

        accesses = (
            Access(1), Access(1, compute_cycles=10), Access(1, compute_cycles=10),
            Access(99, compute_cycles=10), Access(30), Access(40, compute_cycles=20),
        )

        def trace():
            for index, access in enumerate(accesses):
                yield access
                self.assertEqual(sum(isinstance(item, Access) for item in log), index + 1)
                self.assertFalse(
                    any(isinstance(item, PrefetchFeedback) and item.kind in {"pending", "unused"}
                        for item in log)
                )

        config = SimulatorConfig(prefetch_buffer_capacity=1)
        stats = simulate(trace(), config, predictor=Predictor(), on_feedback=log.append)
        self.assertEqual(log, [
            accesses[0], PrefetchFeedback("accepted", 0, 10, 60),
            accesses[1], PrefetchFeedback("accepted", 1, 20, 70),
            accesses[2], PrefetchFeedback("accepted", 2, 30, 80),
            PrefetchFeedback("evicted", 0, 10, 130),
            PrefetchFeedback("evicted", 1, 20, 140),
            accesses[3], PrefetchFeedback("accepted", 3, 40, 150),
            PrefetchFeedback("timely", 2, 30, 150, saved_cycles=60),
            accesses[4], PrefetchFeedback("accepted", 4, 41, 150),
            PrefetchFeedback("late", 3, 40, 210, saved_cycles=20),
            accesses[5], PrefetchFeedback("accepted", 5, 50, 210),
            PrefetchFeedback("unused", 4, 41, 210),
            PrefetchFeedback("pending", 5, 50, 210),
        ])
        self.assertEqual(stats.demand_wait_cycles, 160)
        self.assertEqual(stats.prefetch_walks_issued, 6)
        self.assertEqual(stats.prefetch_evictions, 2)
        self.assertEqual(stats.unused_prefetches, 1)
        self.assertEqual(stats.pending_prefetches, 1)

    def test_ties_and_zero_latency_use_causal_order(self):
        events = []
        config = SimulatorConfig(page_walk_latency=0, prefetch_buffer_capacity=0)
        simulate([Access(1)], config, {0: 2}, on_feedback=events.append)
        self.assertEqual(events, [
            PrefetchFeedback("accepted", 0, 2, 0),
            PrefetchFeedback("evicted", 0, 2, 0),
        ])

        events.clear()
        config = SimulatorConfig(prefetch_buffer_capacity=0)
        simulate(
            [Access(1), Access(1), Access(3, compute_cycles=20)],
            config, {0: 2, 1: 3}, on_feedback=events.append,
        )
        self.assertEqual(events, [
            PrefetchFeedback("accepted", 0, 2, 60),
            PrefetchFeedback("accepted", 1, 3, 60),
            PrefetchFeedback("evicted", 0, 2, 120),
            PrefetchFeedback("late", 1, 3, 120, saved_cycles=20),
        ])

    def test_rejections_are_exclusive_and_repeat_vpn_has_distinct_ids(self):
        trace = (Access(1), Access(1), Access(1), Access(1, compute_cycles=60),
                 Access(1, compute_cycles=60))
        config = SimulatorConfig(prefetch_buffer_capacity=0, max_pending_prefetches=1)
        plan = {0: 2, 1: 2, 2: 3, 3: 2}
        events = []
        stats = simulate(trace, config, plan, on_feedback=events.append)
        self.assertEqual(stats, simulate(trace, config, plan))
        self.assertEqual(stats.prefetch_requests_attempted, 4)
        self.assertEqual(stats.prefetch_walks_issued, 2)
        self.assertEqual(stats.prefetch_evictions, 2)
        self.assertEqual(events, [
            PrefetchFeedback("accepted", 0, 2, 60),
            PrefetchFeedback("rejected", 1, 2, 60, rejection_reason="duplicate"),
            PrefetchFeedback("rejected", 2, 3, 60, rejection_reason="capacity"),
            PrefetchFeedback("evicted", 0, 2, 120),
            PrefetchFeedback("accepted", 3, 2, 120),
            PrefetchFeedback("evicted", 3, 2, 180),
        ])


if __name__ == "__main__":
    unittest.main()
