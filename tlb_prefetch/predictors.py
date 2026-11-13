"""Fixed predictors and an adaptive selector for virtual page numbers."""

from collections import OrderedDict, deque

from tlb_prefetch.simulator import Access, PrefetchFeedback, _validate_integer


class NextPagePrefetcher:
    def __init__(self, max_target_vpn: int | None = None) -> None:
        if max_target_vpn is not None:
            _validate_integer("max_target_vpn", max_target_vpn)
        self._max_target_vpn = max_target_vpn

    def reset(self) -> None:
        """Start a run with no observation history."""

    def observe(self, access: Access) -> int | None:
        target = access.vpn + 1
        if self._max_target_vpn is not None and target > self._max_target_vpn:
            return None
        return target

    @property
    def metadata_entries(self) -> int:
        return 0


class PerPCStridePrefetcher:
    def __init__(self, max_pcs: int = 16) -> None:
        _validate_integer("max_pcs", max_pcs, minimum=1)
        self._max_pcs = max_pcs
        self._history: OrderedDict[int, tuple[int, int | None]] = OrderedDict()

    def reset(self) -> None:
        self._history.clear()

    def observe(self, access: Access) -> int | None:
        previous = self._history.pop(access.pc, None)
        if previous is None:
            if len(self._history) == self._max_pcs:
                self._history.popitem(last=False)
            self._history[access.pc] = access.vpn, None
            return None

        last_vpn, last_stride = previous
        stride = access.vpn - last_vpn
        self._history[access.pc] = access.vpn, stride
        target = access.vpn + stride
        if stride != 0 and stride == last_stride and target >= 0:
            return target
        return None

    @property
    def metadata_entries(self) -> int:
        return len(self._history)


class MarkovPrefetcher:
    _MAX_COUNT = 255

    def __init__(self, max_sources: int = 32, max_successors: int = 4) -> None:
        _validate_integer("max_sources", max_sources, minimum=1)
        _validate_integer("max_successors", max_successors, minimum=1)
        self._max_sources = max_sources
        self._max_successors = max_successors
        self._transitions: OrderedDict[int, OrderedDict[int, int]] = OrderedDict()
        self._previous_vpn: int | None = None

    def reset(self) -> None:
        self._transitions.clear()
        self._previous_vpn = None

    def observe(self, access: Access) -> int | None:
        if self._previous_vpn is not None:
            self._record_transition(self._previous_vpn, access.vpn)
        self._previous_vpn = access.vpn
        successors = self._transitions.get(access.vpn)
        if not successors:
            return None
        return min(successors, key=lambda vpn: (-successors[vpn], vpn))

    def _record_transition(self, source: int, target: int) -> None:
        successors = self._transitions.get(source)
        if successors is None:
            if len(self._transitions) == self._max_sources:
                self._transitions.popitem(last=False)
            successors = OrderedDict()
            self._transitions[source] = successors

        if target not in successors and len(successors) == self._max_successors:
            successors.popitem(last=False)
        successors[target] = min(successors.get(target, 0) + 1, self._MAX_COUNT)

    @property
    def metadata_entries(self) -> int:
        return (
            int(self._previous_vpn is not None)
            + len(self._transitions)
            + sum(len(successors) for successors in self._transitions.values())
        )


class AdaptivePrefetcher:
    """Choose among fixed predictors using recent resolved prefetch outcomes."""

    def __init__(self, history_window: int = 32, probe_interval: int = 4) -> None:
        _validate_integer("history_window", history_window, minimum=1)
        _validate_integer("probe_interval", probe_interval, minimum=1)
        self._candidates = (NextPagePrefetcher(), PerPCStridePrefetcher(), MarkovPrefetcher())
        self._history_window = history_window
        self._probe_interval = probe_interval
        self._recent = tuple(
            deque[tuple[int, int]](maxlen=history_window) for _ in self._candidates
        )
        self._in_flight: dict[int, int] = {}
        self.reset()

    def reset(self) -> None:
        for candidate, history in zip(self._candidates, self._recent):
            candidate.reset()
            history.clear()
        self._in_flight.clear()
        self._observations = 0
        self._next_probe = 0
        self._probe_at = self._probe_interval
        self._selected: int | None = None

    def observe(self, access: Access) -> int | None:
        self._observations += 1
        targets = tuple(candidate.observe(access) for candidate in self._candidates)
        for history in self._recent:
            while history and self._observations - history[0][0] > self._history_window:
                history.popleft()

        self._selected = None
        if self._observations == self._probe_at:
            probe = self._next_probe
            self._next_probe = (probe + 1) % len(self._candidates)
            # Shift each full probe round by one access to reduce alignment with periodic traces
            self._probe_at += self._probe_interval + int(self._next_probe == 0)
            if targets[probe] is not None:
                self._selected = probe
        else:
            best_score = 0
            for index, target in enumerate(targets):
                score = sum(value for _, value in self._recent[index])
                if target is not None and score > best_score:
                    self._selected = index
                    best_score = score

        if self._selected is not None:
            return targets[self._selected]
        return None

    def on_feedback(self, event: PrefetchFeedback) -> None:
        if event.kind == "accepted":
            if self._selected is not None:
                self._in_flight[event.issued_access_index] = self._selected
                self._selected = None
            return
        if event.kind == "rejected":
            self._selected = None
            return
        candidate = self._in_flight.pop(event.issued_access_index)
        if event.kind == "evicted":
            score = -1
        elif event.saved_cycles > 0:
            score = 2 if event.kind == "timely" else 1
        else:
            return
        self._recent[candidate].append((self._observations, score))

    @property
    def metadata_entries(self) -> int:
        return (
            sum(candidate.metadata_entries for candidate in self._candidates)
            + sum(len(history) for history in self._recent)
            + len(self._in_flight)
        )
