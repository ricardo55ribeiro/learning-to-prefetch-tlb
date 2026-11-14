"""Learn a direct page offset from recent per-PC differences."""

from collections import OrderedDict
from random import Random

from tlb_prefetch.simulator import Access, PrefetchFeedback, _validate_integer


OFFSET_LIMIT = 8
OFFSETS = (*range(-OFFSET_LIMIT, 0), *range(1, OFFSET_LIMIT + 1))
ABSTAIN = 0
ACTIONS = (ABSTAIN, *OFFSETS)
MAX_PCS = 16
CONTEXT_COUNT = 40
LEARNING_RATE = 0.2
EPSILON = 0.1
WASTED_WALK_PENALTY = 5

_VALID_ACTIONS = tuple(
    tuple(index for index, offset in enumerate(ACTIONS) if vpn + offset >= 0)
    for vpn in range(OFFSET_LIMIT + 1)
)


class _PageHistory:
    def __init__(self) -> None:
        self.entries: OrderedDict[int, tuple[int, int | None]] = OrderedDict()

    def observe(self, access: Access) -> tuple[int | None, bool]:
        previous = self.entries.pop(access.pc, None)
        delta = None
        repeated = False
        if previous is not None:
            last_vpn, last_delta = previous
            delta = access.vpn - last_vpn
            repeated = delta == last_delta
        elif len(self.entries) == MAX_PCS:
            self.entries.popitem(last=False)
        self.entries[access.pc] = access.vpn, delta
        return delta, repeated


class DirectOffsetBandit:
    value_entries = CONTEXT_COUNT * len(ACTIONS)

    def __init__(self, *, seed: int = 0) -> None:
        _validate_integer("seed", seed)
        self._seed = seed
        self.reset()

    def reset(self) -> None:
        self._history = _PageHistory()
        self._values = [[0.0] * len(ACTIONS) for _ in range(CONTEXT_COUNT)]
        self._rng = Random(self._seed)
        self._choice: tuple[int, int] | None = None
        self._pending: dict[int, tuple[int, int]] = {}

    @property
    def metadata_entries(self) -> int:
        return self.value_entries + len(self._history.entries) + len(self._pending)

    def observe(self, access: Access) -> int | None:
        delta, repeated = self._history.observe(access)
        if delta is None:
            category = 19
        elif delta < -OFFSET_LIMIT:
            category = 17
        elif delta > OFFSET_LIMIT:
            category = 18
        else:
            category = delta + OFFSET_LIMIT
        context = 2 * category + int(repeated)

        available = _VALID_ACTIONS[min(access.vpn, OFFSET_LIMIT)]
        if self._rng.random() < EPSILON:
            action = self._rng.choice(available)
        else:
            action = max(available, key=self._values[context].__getitem__)

        if action == ABSTAIN:
            self._choice = None
            return None
        self._choice = context, action
        return access.vpn + ACTIONS[action]

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
        values[action] += LEARNING_RATE * (reward - values[action])


class LastDelta:
    """Copy the latest per-PC difference when it is an available offset."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._history = _PageHistory()

    def observe(self, access: Access) -> int | None:
        delta, _ = self._history.observe(access)
        if delta is not None and delta in OFFSETS and access.vpn + delta >= 0:
            return access.vpn + delta
        return None

    @property
    def metadata_entries(self) -> int:
        return len(self._history.entries)
