"""Compare fixed and adaptive policies on controlled synthetic workloads."""

from argparse import ArgumentParser
from collections.abc import Iterator
from csv import DictWriter
from dataclasses import dataclass
from pathlib import Path
from random import Random

from tlb_prefetch.bandit import POLICY_SEED, WASTED_WALK_PENALTY, RewardSelector
from tlb_prefetch.predictors import (
    AdaptivePrefetcher, MarkovPrefetcher, NextPagePrefetcher, PerPCStridePrefetcher,
)
from tlb_prefetch.simulator import Access, SimulationStats, SimulatorConfig, simulate


DATA_DIR = Path(__file__).resolve().parents[1] / "results" / "data"
RANDOM_SEED = 42
LONG_SEEDS = (11, 29, 47)
HELD_OUT_SEEDS = tuple(range(101, 111))
WALK_PENALTIES = (0, 5, 20)
COMPARISON_CONFIG = SimulatorConfig(tlb_capacity=16)
LONG_GAPS = (0, 15, 40, 60, 90)
STRIDE_METADATA_BOUND = 16
MARKOV_METADATA_BOUND = 1 + 32 * (1 + 4)
CANDIDATE_METADATA_BOUND = STRIDE_METADATA_BOUND + MARKOV_METADATA_BOUND


@dataclass
class PolicyResult:
    name: str
    stats: SimulationStats
    metadata_entries: int
    baseline_walks: int
    metadata_bound: int = 0
    candidate_calls: int = 0
    value_checks_bound: int = 0

    @property
    def total_walks(self) -> int:
        return self.stats.demand_walks_issued + self.stats.prefetch_walks_issued

    @property
    def additional_walks(self) -> int:
        return self.total_walks - self.baseline_walks

    def adjusted_cost(self, penalty: int) -> int:
        return self.stats.demand_wait_cycles + penalty * self.additional_walks

    @property
    def wasted_or_unused(self) -> int:
        return (
            self.stats.prefetch_evictions
            + self.stats.unused_prefetches
            + self.stats.pending_prefetches
        )


def make_workloads(seed: int = RANDOM_SEED) -> dict[str, tuple[Access, ...]]:
    pattern = (
        10, 23, 6, 16, 2, 13, 20, 0, 8, 19, 4, 15,
        21, 12, 3, 17, 7, 22, 5, 14, 1, 18, 9, 11,
    )
    random = Random(seed)
    pages = {
        "Sequential": range(72),
        "Fixed stride": range(0, 72 * 7, 7),
        "Repeating": pattern * 3,
        "Random": [random.randrange(96) for _ in range(72)],
        "Phase change": pattern * 2 + tuple(range(24)),
    }
    gaps = (20, 60, 0)
    return {
        name: tuple(
            Access(vpn, compute_cycles=gaps[index % len(gaps)])
            for index, vpn in enumerate(vpns)
        )
        for name, vpns in pages.items()
    }


def make_irregular_region(random: Random) -> list[Access]:
    return [
        Access(random.randrange(96), random.randrange(3), random.choice(LONG_GAPS))
        for _ in range(48)
    ]


def make_long_workloads(seed: int) -> dict[str, tuple[Access, ...]]:
    random = Random(seed)
    streams = []
    for phase in range(4):
        base = 100 + 400 * phase
        for index in range(48):
            pc = index % 2
            if phase % 2 == 0:
                vpn = base + index
            else:
                vpn = base + 7 * (index // 2) + pc * 200
            streams.append(Access(vpn, pc, random.choice(LONG_GAPS)))
        streams.extend(make_irregular_region(random))

    pattern = list(range(100, 124))
    random.shuffle(pattern)
    loops = []
    for _ in range(3):
        loops.extend(
            Access(vpn, index % 3, random.choice(LONG_GAPS))
            for index, vpn in enumerate(pattern * 3)
        )
        loops.extend(make_irregular_region(random))

    return {"Streams": tuple(streams), "Loops": tuple(loops)}


def run_reward_selector(
    trace: tuple[Access, ...],
    config: SimulatorConfig,
    baseline_walks: int,
    *,
    contextual: bool,
    policy_seed: int = POLICY_SEED,
) -> PolicyResult:
    selector = RewardSelector(contextual=contextual, seed=policy_seed)
    stats = simulate(trace, config, predictor=selector, on_feedback=selector.on_feedback)
    live_bound = config.prefetch_buffer_capacity + config.max_pending_prefetches
    bound = CANDIDATE_METADATA_BOUND + live_bound + selector.value_entries + int(contextual)
    name = "Contextual" if contextual else "Reward"
    return PolicyResult(name, stats, selector.metadata_entries, baseline_walks, bound, 3, 4)


def compare(
    trace: tuple[Access, ...],
    config: SimulatorConfig = COMPARISON_CONFIG,
    *,
    include_adaptive: bool = False,
    include_reward: bool = False,
) -> tuple[PolicyResult, ...]:
    baseline = simulate(trace, config, predictor=None)
    baseline_walks = baseline.demand_walks_issued
    results = [PolicyResult("No prefetch", baseline, 0, baseline_walks)]
    policies = [
        ("Next page", NextPagePrefetcher(), 0),
        ("Per-PC stride", PerPCStridePrefetcher(), STRIDE_METADATA_BOUND),
        ("Markov", MarkovPrefetcher(), MARKOV_METADATA_BOUND),
    ]
    for name, predictor, metadata_bound in policies:
        stats = simulate(trace, config, predictor=predictor)
        results.append(PolicyResult(
            name, stats, predictor.metadata_entries, baseline_walks, metadata_bound, 1,
        ))

    live_bound = config.prefetch_buffer_capacity + config.max_pending_prefetches
    if include_adaptive:
        adaptive = AdaptivePrefetcher()
        stats = simulate(trace, config, predictor=adaptive, on_feedback=adaptive.on_feedback)
        results.append(PolicyResult(
            "Adaptive", stats, adaptive.metadata_entries, baseline_walks,
            CANDIDATE_METADATA_BOUND + live_bound + 3 * 32, 3, 3 * 32,
        ))
    if include_reward:
        for contextual in (False, True):
            results.append(run_reward_selector(trace, config, baseline_walks, contextual=contextual))
    return tuple(results)


def best_fixed_at_five(policies: tuple[PolicyResult, ...]) -> PolicyResult:
    return min(policies[1:4], key=lambda result: result.adjusted_cost(5))


def workloads() -> Iterator[tuple[str, str, int | None, tuple[Access, ...]]]:
    for name, trace in make_workloads().items():
        yield "Regression", name, RANDOM_SEED if name == "Random" else None, trace
    for split, seeds in (("Development", LONG_SEEDS), ("Held-out", HELD_OUT_SEEDS)):
        for seed in seeds:
            for name, trace in make_long_workloads(seed).items():
                yield split, name, seed, trace


CSV_COLUMNS = (
    "split", "workload", "workload_seed", "policy", "policy_seed", "demand_accesses",
    "demand_wait_cycles", "elapsed_cycles", "demand_walks", "prefetch_walks", "total_walks",
    "additional_walks", "timely_prefetches", "late_prefetches", "wasted_prefetches",
    "rejected_requests", "metadata_retained", "metadata_bound", "candidate_calls_per_demand",
    "selector_value_checks_per_demand_bound", "cost_at_0", "cost_at_5", "cost_at_20",
    "best_fixed_at_5", "best_fixed_cost_at_5",
)


def csv_row(
    split: str, name: str, seed: int | None, result: PolicyResult, best_fixed: PolicyResult,
    *, policy_seed: int = POLICY_SEED,
) -> dict[str, str | int | None]:
    stats = result.stats
    values = (
        split, name, seed, result.name, policy_seed if result.name in ("Reward", "Contextual") else None,
        stats.demand_accesses, stats.demand_wait_cycles, stats.elapsed_cycles,
        stats.demand_walks_issued, stats.prefetch_walks_issued, result.total_walks,
        result.additional_walks, stats.timely_prefetches_consumed, stats.late_prefetches_consumed,
        result.wasted_or_unused, stats.prefetch_requests_rejected, result.metadata_entries,
        result.metadata_bound, result.candidate_calls, result.value_checks_bound,
        *(result.adjusted_cost(penalty) for penalty in WALK_PENALTIES),
        best_fixed.name, best_fixed.adjusted_cost(5),
    )
    return dict(zip(CSV_COLUMNS, values, strict=True))


def print_summary(
    groups: dict[tuple[str, str], list[dict[str, str | int | None]]], sensitivity: bool,
) -> None:
    metrics = (
        "Wait", "Walks", "Extra", "Timely", "Late", "Waste", "Reject",
        "MetaMax", "Calls", "Checks",
    )
    if sensitivity:
        metrics += ("Cost@0", "Cost@5", "Cost@20")
    print(f"{'Group':<22}{'Policy':<15}" + "".join(f"{metric:>8}" for metric in metrics))
    totals = (
        "demand_wait_cycles", "total_walks", "additional_walks", "timely_prefetches",
        "late_prefetches", "wasted_prefetches", "rejected_requests",
    )
    work_bounds = (
        "metadata_bound", "candidate_calls_per_demand", "selector_value_checks_per_demand_bound",
    )
    for (group, policy), rows in groups.items():
        values = [sum(row[field] for row in rows) for field in totals]
        values.extend(rows[0][field] for field in work_bounds)
        if sensitivity:
            values.extend(sum(row[f"cost_at_{penalty}"] for row in rows)
                          for penalty in WALK_PENALTIES)
        print(f"{group:<22}{policy:<15}" + "".join(f"{value:>8}" for value in values))


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--sensitivity", action="store_true", help="Show hypothetical walk penalties")
    parser.add_argument(
        "--csv", type=Path, metavar="PATH",
        help="Write per-trace metrics to PATH, relative to results/data or absolute",
    )
    args = parser.parse_args()

    rows = []
    groups = {}
    best_fixed_total = {}
    for split, name, seed, trace in workloads():
        policies = compare(trace, include_adaptive=True, include_reward=True)
        best_fixed = best_fixed_at_five(policies)
        group = name if split == "Regression" else f"{name} {split}"
        best_fixed_total[group] = best_fixed_total.get(group, 0) + best_fixed.adjusted_cost(5)
        for result in policies:
            row = csv_row(split, name, seed, result, best_fixed)
            rows.append(row)
            groups.setdefault((group, result.name), []).append(row)

    print(COMPARISON_CONFIG)
    print(f"Development seeds {LONG_SEEDS} | Held-out seeds {HELD_OUT_SEEDS}")
    print(f"Selector seed {POLICY_SEED} | Wasted-walk reward penalty {WASTED_WALK_PENALTY}")
    print("Metrics sum over seeds. MetaMax is a record bound. Calls and Checks are per demand.")
    print_summary(groups, args.sensitivity)
    print()
    print("Retrospective best fixed cost at 5, selected separately on each trace")
    for group, cost in best_fixed_total.items():
        print(f"{group:<22}{cost:>8}")

    if args.csv:
        path = DATA_DIR / args.csv
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as output:
            csv = DictWriter(output, CSV_COLUMNS)
            csv.writeheader()
            csv.writerows(rows)


if __name__ == "__main__":
    main()
