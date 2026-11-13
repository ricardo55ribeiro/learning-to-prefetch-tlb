import unittest
from unittest.mock import patch

from bandit import RewardSelector
from compare import HELD_OUT_SEEDS, LONG_SEEDS, compare, make_long_workloads
from simulator import Access, PrefetchFeedback, SimulatorConfig, simulate


class RewardSelectorTests(unittest.TestCase):
    def test_context_bits_come_from_current_and_prior_observations(self):
        selector = RewardSelector(contextual=True, epsilon=0)
        accesses = (Access(10), Access(11), Access(12), Access(10, pc=1))
        for context, access in zip((0, 4, 6, 1), accesses):
            selector._values[context][0] = 1
            self.assertEqual(selector.observe(access), access.vpn + 1)
            self.assertEqual(selector._choice, (context, 0))
            selector._values[context][0] = 0

    def test_greedy_selection_abstains_or_ignores_unavailable_candidates(self):
        selector = RewardSelector(epsilon=0)
        self.assertIsNone(selector.observe(Access(10)))
        self.assertEqual(selector._values, [[0.0] * 4])
        selector._values[0][:3] = [1, 9, 8]
        self.assertEqual(selector.observe(Access(11)), 12)
        self.assertEqual(selector.observe(Access(12)), 13)
        self.assertEqual(selector._choice, (0, 1))
        selector._values[0][:3] = [-1, -1, -1]
        self.assertIsNone(selector.observe(Access(13)))
        self.assertEqual(selector._values[0][3], 0)

    def test_seeded_exploration_repeats_and_only_uses_available_actions(self):
        selector = RewardSelector(epsilon=1, seed=7)
        accesses = [Access(vpn, pc=vpn) for vpn in range(20)]
        first = [selector.observe(access) for access in accesses]
        selector.reset()
        self.assertEqual(first, [selector.observe(access) for access in accesses])
        different = RewardSelector(epsilon=1, seed=8)
        self.assertNotEqual(first, [different.observe(access) for access in accesses])
        self.assertTrue(all(
            target in (None, access.vpn + 1) for access, target in zip(accesses, first)
        ))
        self.assertIn(None, first)
        self.assertTrue(any(target is not None for target in first))

    def test_all_candidates_observe_even_when_selector_abstains(self):
        selector = RewardSelector(epsilon=0)
        accesses = (Access(1), Access(2), Access(3))
        traces = [[] for _ in selector._candidates]
        for candidate, observed in zip(selector._candidates, traces):
            candidate.observe = observed.append
        self.assertEqual([selector.observe(access) for access in accesses], [None] * 3)
        self.assertEqual(traces, [list(accesses)] * 3)

    def test_delayed_reward_is_single_update_to_issuing_context_and_action(self):
        selector = RewardSelector(contextual=True, epsilon=0)
        selector._values[0][0] = 10
        self.assertEqual(selector.observe(Access(10)), 11)
        selector.on_feedback(PrefetchFeedback("accepted", 0, 11, 60))
        self.assertEqual(selector._values[0][0], 10)

        selector.observe(Access(12))
        selector._values[6][1] = 10
        self.assertIsNone(selector.observe(Access(13)))
        selector.observe(Access(14))
        self.assertEqual(selector.observe(Access(15)), 16)
        self.assertEqual(selector._choice, (6, 1))
        selector.on_feedback(PrefetchFeedback("accepted", 4, 16, 90))

        outcome = PrefetchFeedback("late", 0, 11, 120, saved_cycles=20)
        selector.on_feedback(outcome)
        selector.on_feedback(outcome)
        self.assertEqual(selector._values[0][0], 12)
        self.assertEqual(selector._values[6][1], 10)
        selector.on_feedback(PrefetchFeedback("timely", 4, 16, 160, saved_cycles=60))
        self.assertEqual(selector._values[6][1], 20)
        self.assertEqual(selector._pending, {})

    def test_evicted_and_terminal_outcomes_charge_exactly_one_penalty(self):
        for kind in ("evicted", "unused", "pending"):
            with self.subTest(kind=kind):
                selector = RewardSelector(epsilon=0, learning_rate=1)
                selector._values[0][0] = 1
                self.assertEqual(selector.observe(Access(100)), 101)
                selector.on_feedback(PrefetchFeedback("accepted", 0, 101, 60))
                selector.on_feedback(PrefetchFeedback(kind, 0, 101, 120))
                self.assertEqual(selector._values[0], [-5, 0, 0, 0])
                self.assertIsNone(selector.observe(Access(400)))

    def test_rejection_resolves_to_zero_and_has_no_delayed_credit(self):
        for reason in ("capacity", "duplicate"):
            with self.subTest(reason=reason):
                selector = RewardSelector(epsilon=0)
                selector._values[0][0] = 10
                selector.observe(Access(1))
                event = PrefetchFeedback("rejected", 0, 2, 60, rejection_reason=reason)
                selector.on_feedback(event)
                selector.on_feedback(event)
                self.assertEqual(selector._values[0][0], 8)
                self.assertEqual(selector._pending, {})

    def test_simulation_reward_conserves_cycles_and_is_causal(self):
        selector = RewardSelector(contextual=True, epsilon=1)
        config = SimulatorConfig(
            tlb_capacity=3, prefetch_buffer_capacity=2, max_pending_prefetches=1,
        )
        accesses = make_long_workloads(11)["Streams"]
        choices = []
        feedback = []
        rewards = []
        original_observe = selector.observe
        original_update = selector._update

        def observe(access):
            choices.append(original_observe(access))
            return choices[-1]

        def trace():
            for index, access in enumerate(accesses, start=1):
                yield access
                self.assertEqual(len(choices), index)
                self.assertFalse(any(event.kind in ("pending", "unused") for event in feedback))

        def receive(event):
            feedback.append(event)
            selector.on_feedback(event)
            self.assertLessEqual(len(selector._pending), 3)
            self.assertLessEqual(selector.metadata_entries, 177 + 32 + 1 + 3)

        def update(choice, reward):
            rewards.append(reward)
            original_update(choice, reward)

        with (
            patch.object(selector, "observe", side_effect=observe),
            patch.object(selector, "_update", side_effect=update),
        ):
            stats = simulate(trace(), config, predictor=selector, on_feedback=receive)
        baseline = simulate(accesses, config)
        total_walks = stats.demand_walks_issued + stats.prefetch_walks_issued
        extra = total_walks - baseline.demand_walks_issued
        self.assertEqual(sum(rewards), baseline.demand_wait_cycles - stats.demand_wait_cycles - 5 * extra)
        self.assertEqual(len(rewards), stats.prefetch_requests_attempted)
        self.assertGreater(stats.prefetch_walks_issued, 0)
        self.assertGreater(extra, 0)
        self.assertEqual(selector._pending, {})
        self.assertEqual(stats.demand_tlb_misses, baseline.demand_tlb_misses)

    def test_learning_prefers_next_page_then_stride_on_controlled_phases(self):
        sequence = [Access(100 + index, pc=index, compute_cycles=60) for index in range(64)]
        stride = [Access(1000 + 7 * index, compute_cycles=60) for index in range(64)]
        selector = RewardSelector(contextual=True)
        choices = []
        original_observe = selector.observe

        def observe(access):
            target = original_observe(access)
            choices.append(selector._choice)
            return target

        with patch.object(selector, "observe", side_effect=observe):
            simulate(sequence + stride, predictor=selector, on_feedback=selector.on_feedback)
        self.assertGreater(choices[40:63].count((4, 0)), 15)
        self.assertGreater(choices[-24:-1].count((2, 1)), 15)
        self.assertGreater(selector._values[4][0], 0)
        self.assertGreater(selector._values[2][1], max(0, selector._values[2][0]))

    def test_independent_runs_reset_models_and_rng(self):
        trace = make_long_workloads(11)["Loops"]
        for contextual in (False, True):
            with self.subTest(contextual=contextual):
                selector = RewardSelector(contextual=contextual, seed=4)
                first = simulate(trace, predictor=selector, on_feedback=selector.on_feedback)
                values = [row.copy() for row in selector._values]
                self.assertTrue(any(any(row) for row in values))
                second = simulate(trace, predictor=selector, on_feedback=selector.on_feedback)
                self.assertEqual(first, second)
                self.assertEqual(values, selector._values)
                simulate([], predictor=selector, on_feedback=selector.on_feedback)
                self.assertEqual(selector.metadata_entries, 32 if contextual else 4)
                self.assertFalse(any(any(row) for row in selector._values))

    def test_evaluation_has_disjoint_seeds_and_fair_configurations(self):
        self.assertEqual(len(set(HELD_OUT_SEEDS)), 10)
        self.assertFalse(set(HELD_OUT_SEEDS) & set(LONG_SEEDS))
        trace = tuple(Access(vpn, compute_cycles=5) for vpn in range(8))
        config = SimulatorConfig(tlb_capacity=2, page_walk_latency=15, max_pending_prefetches=0)
        with patch("compare.simulate", wraps=simulate) as run:
            policies = compare(trace, config, include_adaptive=True, include_reward=True)
        self.assertEqual(len(policies), 7)
        self.assertEqual(run.call_count, 7)
        for call, policy in zip(run.call_args_list, policies):
            self.assertIs(call.args[0], trace)
            self.assertIs(call.args[1], config)
            self.assertEqual(policy.stats.demand_wait_cycles, 120)
            self.assertEqual(policy.total_walks, 8)
            self.assertEqual(policy.additional_walks, 0)
            self.assertLessEqual(policy.metadata_entries, policy.metadata_bound)


if __name__ == "__main__":
    unittest.main()
