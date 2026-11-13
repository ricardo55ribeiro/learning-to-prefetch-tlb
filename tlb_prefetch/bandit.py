"""Small reward selectors that differ only in their use of context."""

from random import Random

from tlb_prefetch.predictors import MarkovPrefetcher, NextPagePrefetcher, PerPCStridePrefetcher
from tlb_prefetch.simulator import Access, PrefetchFeedback, _validate_integer


WASTED_WALK_PENALTY = 5
POLICY_SEED = 0
ABSTAIN = 3


class RewardSelector:
    """Use one value row for all accesses or one row per three-bit context."""

    def __init__(
        self,
        *,
        contextual: bool = False,
        learning_rate: float = 0.2,
        epsilon: float = 0.1,
        seed: int = POLICY_SEED,
    ) -> None:
        if not 0 < learning_rate <= 1:
            raise ValueError("learning_rate must be in (0, 1]")
        if not 0 <= epsilon <= 1:
            raise ValueError("epsilon must be in [0, 1]")
        _validate_integer("seed", seed)
        self._contextual = contextual
        self._learning_rate = learning_rate
        self._epsilon = epsilon
        self._seed = seed
        self._candidates = (NextPagePrefetcher(), PerPCStridePrefetcher(), MarkovPrefetcher())
        self.reset()

    @property
    def value_entries(self) -> int:
        return (8 if self._contextual else 1) * 4

    @property
    def metadata_entries(self) -> int:
        return (
            sum(candidate.metadata_entries for candidate in self._candidates)
            + self.value_entries + len(self._pending) + int(self._previous_vpn is not None)
        )

    def reset(self) -> None:
        for candidate in self._candidates:
            candidate.reset()
        self._rng = Random(self._seed)
        self._values = [[0.0] * 4 for _ in range(self.value_entries // 4)]
        self._previous_vpn: int | None = None
        self._choice: tuple[int, int] | None = None
        self._pending: dict[int, tuple[int, int]] = {}

    def observe(self, access: Access) -> int | None:
        targets = tuple(candidate.observe(access) for candidate in self._candidates)
        context = 0
        if self._contextual:
            sequential = self._previous_vpn is not None and access.vpn == self._previous_vpn + 1
            context = 4 * sequential + 2 * (targets[1] is not None) + (targets[2] is not None)
            self._previous_vpn = access.vpn

        # Put abstention first so it wins ties at zero
        available = [ABSTAIN]
        available.extend(action for action, target in enumerate(targets) if target is not None)
        if self._rng.random() < self._epsilon:
            action = self._rng.choice(available)
        else:
            action = max(available, key=self._values[context].__getitem__)

        self._choice = None
        if action == ABSTAIN:
            # Its initial value and every reward are zero
            return None
        self._choice = context, action
        return targets[action]

    def on_feedback(self, event: PrefetchFeedback) -> None:
        if event.kind in ("accepted", "rejected"):
            choice = self._choice
            self._choice = None
            if choice is None:
                return
            if event.kind == "accepted":
                self._pending[event.issued_access_index] = choice
            else:
                self._update(choice, 0)
            return

        choice = self._pending.pop(event.issued_access_index, None)
        if choice is not None:
            reward = (
                event.saved_cycles if event.kind in ("timely", "late") else -WASTED_WALK_PENALTY
            )
            self._update(choice, reward)

    def _update(self, choice: tuple[int, int], reward: int) -> None:
        context, action = choice
        values = self._values[context]
        values[action] += self._learning_rate * (reward - values[action])
