"""Simulate a blocking stream of virtual-page accesses with optional prefetches."""

from collections import OrderedDict
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Literal, Protocol, TypeAlias


def _validate_integer(name: str, value: int, minimum: int = 0) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


@dataclass(frozen=True)
class Access:
    vpn: int
    pc: int = 0
    compute_cycles: int = 0

    def __post_init__(self) -> None:
        _validate_integer("vpn", self.vpn)
        _validate_integer("pc", self.pc)
        _validate_integer("compute_cycles", self.compute_cycles)


class Predictor(Protocol):
    def reset(self) -> None:
        """Clear observation history before an independent run."""

    def observe(self, access: Access) -> int | None:
        """Observe one completed demand and return at most one target."""


FeedbackKind: TypeAlias = Literal["accepted", "rejected", "timely", "late", "evicted", "unused", "pending"]
RejectionReason: TypeAlias = Literal["duplicate", "capacity"]


@dataclass(frozen=True)
class PrefetchFeedback:
    """An admission or outcome associated with one run-local request."""

    kind: FeedbackKind
    issued_access_index: int
    vpn: int
    cycle: int
    saved_cycles: int = 0
    rejection_reason: RejectionReason | None = None


@dataclass(frozen=True)
class SimulatorConfig:
    page_size: int = 4096
    tlb_capacity: int = 32
    prefetch_buffer_capacity: int = 8
    page_walk_latency: int = 60
    max_pending_prefetches: int = 4

    def __post_init__(self) -> None:
        _validate_integer("page_size", self.page_size, minimum=1)
        _validate_integer("tlb_capacity", self.tlb_capacity, minimum=1)
        _validate_integer("prefetch_buffer_capacity", self.prefetch_buffer_capacity)
        _validate_integer("page_walk_latency", self.page_walk_latency)
        _validate_integer("max_pending_prefetches", self.max_pending_prefetches)


@dataclass
class SimulationStats:
    demand_accesses: int = 0
    demand_tlb_hits: int = 0
    demand_tlb_misses: int = 0
    prefetch_buffer_hits: int = 0
    demands_joined_prefetches: int = 0
    demand_walks_issued: int = 0
    demand_wait_cycles: int = 0
    elapsed_cycles: int = 0
    prefetch_requests_attempted: int = 0
    prefetch_walks_issued: int = 0
    prefetch_duplicate_rejections: int = 0
    prefetch_capacity_rejections: int = 0
    prefetch_evictions: int = 0
    unused_prefetches: int = 0
    pending_prefetches: int = 0

    @property
    def prefetch_requests_rejected(self) -> int:
        return self.prefetch_duplicate_rejections + self.prefetch_capacity_rejections

    @property
    def timely_prefetches_consumed(self) -> int:
        return self.prefetch_buffer_hits

    @property
    def late_prefetches_consumed(self) -> int:
        return self.demands_joined_prefetches

    @property
    def extra_prefetch_walks(self) -> int:
        return self.prefetch_walks_issued


class _Simulation:
    def __init__(
        self,
        config: SimulatorConfig,
        on_feedback: Callable[[PrefetchFeedback], None] | None,
    ) -> None:
        self.config = config
        self.on_feedback = on_feedback
        self.events: list[PrefetchFeedback] = []
        self.stats = SimulationStats()
        self.cycle = 0
        self.tlb: OrderedDict[int, None] = OrderedDict()
        self.buffer: OrderedDict[int, int] = OrderedDict()
        self.pending: OrderedDict[int, tuple[int, int]] = OrderedDict()

    def record_feedback(
        self,
        kind: FeedbackKind,
        vpn: int,
        issued_index: int,
        cycle: int,
        saved_cycles: int = 0,
        rejection_reason: RejectionReason | None = None,
    ) -> None:
        if self.on_feedback is not None:
            self.events.append(
                PrefetchFeedback(kind, issued_index, vpn, cycle, saved_cycles, rejection_reason)
            )

    def deliver_feedback(self) -> None:
        if self.on_feedback is not None:
            for event in self.events:
                self.on_feedback(event)
            self.events.clear()

    def advance_to(self, cycle: int, joined_vpn: int | None = None) -> None:
        # With fixed latency, issue order is completion order, including ties
        while self.pending:
            vpn, (ready_cycle, issued_index) = next(iter(self.pending.items()))
            if ready_cycle > cycle:
                break
            self.pending.popitem(last=False)
            if vpn == joined_vpn:
                saved = self.config.page_walk_latency - (ready_cycle - self.cycle)
                self.record_feedback("late", vpn, issued_index, ready_cycle, saved)
                continue

            capacity = self.config.prefetch_buffer_capacity
            if capacity == 0:
                self.stats.prefetch_evictions += 1
                self.record_feedback("evicted", vpn, issued_index, ready_cycle)
                continue
            if len(self.buffer) == capacity:
                evicted_vpn, evicted_index = self.buffer.popitem(last=False)
                self.stats.prefetch_evictions += 1
                self.record_feedback("evicted", evicted_vpn, evicted_index, ready_cycle)
            self.buffer[vpn] = issued_index
        self.cycle = cycle

    def wait_until(self, cycle: int, joined_vpn: int | None = None) -> None:
        self.stats.demand_wait_cycles += cycle - self.cycle
        self.advance_to(cycle, joined_vpn)

    def access(self, access: Access) -> None:
        self.advance_to(self.cycle + access.compute_cycles)
        vpn = access.vpn
        self.stats.demand_accesses += 1
        if vpn in self.tlb:
            self.stats.demand_tlb_hits += 1
            self.tlb.move_to_end(vpn)
            return

        self.stats.demand_tlb_misses += 1
        if vpn in self.buffer:
            self.stats.prefetch_buffer_hits += 1
            issued_index = self.buffer.pop(vpn)
            self.record_feedback(
                "timely", vpn, issued_index, self.cycle, self.config.page_walk_latency
            )
        elif vpn in self.pending:
            self.stats.demands_joined_prefetches += 1
            ready_cycle, _ = self.pending[vpn]
            self.wait_until(ready_cycle, joined_vpn=vpn)
        else:
            self.stats.demand_walks_issued += 1
            self.wait_until(self.cycle + self.config.page_walk_latency)

        if len(self.tlb) == self.config.tlb_capacity:
            self.tlb.popitem(last=False)
        self.tlb[vpn] = None

    def prefetch(self, vpn: int) -> None:
        issued_index = self.stats.demand_accesses - 1
        self.stats.prefetch_requests_attempted += 1
        if vpn in self.tlb or vpn in self.buffer or vpn in self.pending:
            self.stats.prefetch_duplicate_rejections += 1
            self.record_feedback(
                "rejected", vpn, issued_index, self.cycle, rejection_reason="duplicate"
            )
            return
        if len(self.pending) == self.config.max_pending_prefetches:
            self.stats.prefetch_capacity_rejections += 1
            self.record_feedback(
                "rejected", vpn, issued_index, self.cycle, rejection_reason="capacity"
            )
            return

        self.stats.prefetch_walks_issued += 1
        self.pending[vpn] = self.cycle + self.config.page_walk_latency, issued_index
        self.record_feedback("accepted", vpn, issued_index, self.cycle)
        self.advance_to(self.cycle)

    def finish(self) -> SimulationStats:
        self.stats.elapsed_cycles = self.cycle
        self.stats.unused_prefetches = len(self.buffer)
        self.stats.pending_prefetches = len(self.pending)
        if self.on_feedback is not None:
            for vpn, issued_index in self.buffer.items():
                self.record_feedback("unused", vpn, issued_index, self.cycle)
            for vpn, (_, issued_index) in self.pending.items():
                self.record_feedback("pending", vpn, issued_index, self.cycle)
            self.deliver_feedback()
        return self.stats


def simulate(
    trace: Iterable[Access],
    config: SimulatorConfig = SimulatorConfig(),
    prefetch_plan: Mapping[int, int | None] | None = None,
    *,
    predictor: Predictor | None = None,
    on_feedback: Callable[[PrefetchFeedback], None] | None = None,
) -> SimulationStats:
    """Run from empty state and request prefetches after demands finish.

    Deliver demand-phase feedback before observing the completed access, admission
    feedback after its request and terminal accounting after the trace ends.
    """
    if predictor is not None and prefetch_plan is not None:
        raise ValueError("supply either a predictor or a scripted prefetch plan")

    plan = prefetch_plan if prefetch_plan is not None else {}
    for index, vpn in plan.items():
        _validate_integer("prefetch plan index", index)
        if vpn is not None:
            _validate_integer("prefetch target vpn", vpn)

    simulation = _Simulation(config, on_feedback)
    if predictor is not None:
        predictor.reset()
    for index, access in enumerate(trace):
        simulation.access(access)
        simulation.deliver_feedback()
        if predictor is None:
            target = plan.get(index)
        else:
            target = predictor.observe(access)
            if target is not None:
                _validate_integer("predictor target vpn", target)
        if target is not None:
            simulation.prefetch(target)
            simulation.deliver_feedback()

    return simulation.finish()
