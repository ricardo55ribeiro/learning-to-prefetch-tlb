"""Compare direct offset learning with fixed and learned selector baselines."""

from argparse import ArgumentParser
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator
from csv import DictReader
from json import dumps
from random import Random
from statistics import mean, pstdev

from experiments.compare import (
    COMPARISON_CONFIG, CSV_COLUMNS, DATA_DIR, LONG_GAPS, LONG_SEEDS, PolicyResult,
    best_fixed_at_five, compare, csv_row, run_reward_selector,
)
from experiments.finalize import make_mixed_trace, write_csv
from tlb_prefetch.direct_offset import ACTIONS, MAX_PCS, OFFSETS, DirectOffsetBandit, LastDelta
from tlb_prefetch.simulator import Access, Predictor, PrefetchFeedback, SimulatorConfig, simulate


EVALUATION_SEEDS = range(301, 306)
POLICY_SEEDS = range(10)
TRACE_LENGTH = 384
RESULT_PATH = DATA_DIR / "direct_offset.csv"
DIRECT = "Direct bandit"
LAST_DELTA = "Last delta"
DETERMINISTIC = ("No prefetch", "Next page", "Per-PC stride", "Markov", LAST_DELTA)
LEARNED = ("Reward", "Contextual", DIRECT)
CSV_FIELDS = CSV_COLUMNS + (
    "abstentions", "offset_counts", "useful_offset_counts",
    "q_values", "pc_history_bound", "live_attribution_bound",
    "cost_at_5_plus_1_per_direct_decision", "cost_at_5_plus_5_per_direct_decision",
)


def make_workloads(seed: int) -> dict[str, tuple[Access, ...]]:
    random = Random(seed)
    length = TRACE_LENGTH
    pattern = list(range(100, 124))
    random.shuffle(pattern)

    def trace(pages: Iterable[int], pc_count: int = 1) -> tuple[Access, ...]:
        return tuple(
            Access(vpn, index % pc_count, random.choice(LONG_GAPS))
            for index, vpn in enumerate(pages)
        )

    interleaved = (100 + i // 2 if i % 2 == 0 else 1000 + 7 * (i // 2) for i in range(length))
    return {
        "Sequential": trace(range(100, 100 + length)),
        "Stride +7": trace(range(100, 100 + 7 * length, 7)),
        "Interleaved": trace(interleaved, 2),
        "Irregular loop": trace(pattern * (length // len(pattern))),
        "Random": trace([random.randrange(96) for _ in range(length)], 3),
        "Mixed phase": make_mixed_trace(seed),
        "Stride +13": trace(range(100, 100 + 13 * length, 13)),
    }


class _OffsetCounts:
    """Count proposed offsets outside the predictor state."""

    def __init__(self, predictor: Predictor) -> None:
        self.predictor = predictor
        self.reset()

    def reset(self) -> None:
        self.offsets: Counter[int] = Counter()
        self.useful: Counter[int] = Counter()
        self._accepted: dict[int, int] = {}
        self._last_offset: int | None = None
        self.predictor.reset()

    def observe(self, access: Access) -> int | None:
        target = self.predictor.observe(access)
        self._last_offset = None if target is None else target - access.vpn
        if self._last_offset is not None:
            self.offsets[self._last_offset] += 1
        return target

    def on_feedback(self, event: PrefetchFeedback) -> None:
        index = event.issued_access_index
        if event.kind == "accepted":
            assert self._last_offset is not None
            self._accepted[index] = self._last_offset
        elif event.kind != "rejected":
            offset = self._accepted.pop(index)
            if event.kind in ("timely", "late"):
                self.useful[offset] += 1


def compare_trace(
    split: str, name: str, seed: int, trace: tuple[Access, ...],
    config: SimulatorConfig = COMPARISON_CONFIG,
) -> Iterator[dict[str, int | str | None]]:
    fixed = compare(trace, config)
    baseline = fixed[0].total_walks
    best_fixed = best_fixed_at_five(fixed)

    def row(
        result: PolicyResult, policy_seed: int | None = None,
        counts: _OffsetCounts | None = None, q_values: int = 0,
        pc_bound: int | None = None, attribution_bound: int | None = None,
    ) -> dict[str, int | str | None]:
        result_row = csv_row(split, name, seed, result, best_fixed)
        charge = result.stats.demand_accesses if result.name == DIRECT else 0
        result_row.update(
            policy_seed=policy_seed,
            abstentions=result.stats.demand_accesses - result.stats.prefetch_requests_attempted,
            offset_counts=dumps(dict(counts.offsets), sort_keys=True, separators=(",", ":"))
            if counts is not None else None,
            useful_offset_counts=dumps(dict(counts.useful), sort_keys=True, separators=(",", ":"))
            if counts is not None else None,
            q_values=q_values,
            pc_history_bound=pc_bound,
            live_attribution_bound=attribution_bound,
            cost_at_5_plus_1_per_direct_decision=result.adjusted_cost(5) + charge,
            cost_at_5_plus_5_per_direct_decision=result.adjusted_cost(5) + 5 * charge,
        )
        return result_row

    for result in fixed:
        yield row(result, pc_bound=16 if result.name == "Per-PC stride" else 0, attribution_bound=0)

    heuristic = LastDelta()
    control = _OffsetCounts(heuristic)
    control_stats = simulate(trace, config, predictor=control, on_feedback=control.on_feedback)
    yield row(
        PolicyResult(LAST_DELTA, control_stats, heuristic.metadata_entries, baseline, MAX_PCS),
        counts=control, pc_bound=MAX_PCS, attribution_bound=0,
    )

    attribution_bound = config.prefetch_buffer_capacity + config.max_pending_prefetches
    for policy_seed in POLICY_SEEDS:
        for contextual in (False, True):
            result = run_reward_selector(
                trace, config, baseline, contextual=contextual, policy_seed=policy_seed,
            )
            yield row(
                result, policy_seed, q_values=32 if contextual else 4,
                pc_bound=16, attribution_bound=attribution_bound,
            )
        learner = DirectOffsetBandit(seed=policy_seed)
        counted = _OffsetCounts(learner)

        def feedback(event: PrefetchFeedback) -> None:
            counted.on_feedback(event)
            learner.on_feedback(event)

        stats = simulate(trace, config, predictor=counted, on_feedback=feedback)
        bound = learner.value_entries + MAX_PCS + attribution_bound
        yield row(
            PolicyResult(DIRECT, stats, learner.metadata_entries, baseline, bound, 0, len(ACTIONS)),
            policy_seed, counted, learner.value_entries, MAX_PCS, attribution_bound,
        )


def experiment(split: str, seeds: range | tuple[int, ...]) -> Iterator[dict[str, int | str | None]]:
    for seed in seeds:
        for name, trace in make_workloads(seed).items():
            yield from compare_trace(split, name, seed, trace)


def summarize(rows: Iterable[dict[str, int | str | None]], split: str) -> None:
    costs = defaultdict(list)
    for row in rows:
        costs[row["workload"], row["policy"]].append(int(row["cost_at_5"]))
    print(f"{split} mean adjusted cost at 5 cycles per extra walk")
    names = (*DETERMINISTIC, *LEARNED)
    labels = ("None", "Next", "Stride", "Markov", "Last delta", "Reward", "Context", "Direct")
    print(f"{'Workload':<16}" + "".join(f"{label:>11}" for label in labels) + f"{'Direct SD':>11}")
    for workload in dict.fromkeys(workload for workload, _ in costs):
        values = [mean(costs[workload, policy]) for policy in names]
        spread = pstdev(costs[workload, DIRECT])
        print(f"{workload:<16}" + "".join(f"{value:>11,.0f}" for value in values) + f"{spread:>11,.0f}")


def plot_results() -> None:
    from experiments.plot_results import DARK_STYLE, FIGURE_DIR, plt
    from matplotlib.ticker import StrMethodFormatter

    direct = defaultdict(list)
    control = defaultdict(list)
    fixed = defaultdict(list)
    with RESULT_PATH.open(encoding="utf-8", newline="") as source:
        for row in DictReader(source):
            if row["split"] != "Evaluation":
                continue
            workload = row["workload"]
            if row["policy"] == DIRECT:
                direct[workload].append(int(row["cost_at_5"]))
            elif row["policy"] == LAST_DELTA:
                control[workload].append(int(row["cost_at_5"]))
            elif row["policy"] == "No prefetch":
                fixed[workload].append(int(row["best_fixed_cost_at_5"]))

    with plt.rc_context(DARK_STYLE):
        fig, ax = plt.subplots(figsize=(9, 5.3), layout="constrained")
        ax.set_title("Direct offset learning on held-out traces", loc="left", pad=13)
        labels = list(direct)
        policies = (
            ("Best fixed per trace", fixed, "#9BA8B9"),
            ("Last-delta control", control, "#F2AE66"),
            ("Direct bandit", direct, "#58A6FF"),
        )
        for index, (name, costs, color) in enumerate(policies):
            positions = [row + (index - 1) * 0.24 for row in range(len(labels))]
            ax.barh(positions, [mean(costs[label]) for label in labels], 0.21, color=color, label=name)
        ax.set_yticks(range(len(labels)), labels)
        ax.invert_yaxis()
        ax.set_xlim(left=0)
        ax.xaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
        ax.set_xlabel("Mean simulated cost at 5 cycles per extra walk")
        ax.grid(axis="x")
        ax.legend(loc="upper left", bbox_to_anchor=(0, -0.15), frameon=False, ncols=3)
        FIGURE_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(FIGURE_DIR / "direct_offset.png", dpi=180, transparent=False)
        plt.close(fig)


def fixed_offset_cost(trace: tuple[Access, ...], offset: int, baseline_walks: int) -> int:
    plan = {
        index: access.vpn + offset
        for index, access in enumerate(trace)
        if access.vpn + offset >= 0
    }
    stats = simulate(trace, COMPARISON_CONFIG, plan)
    return PolicyResult("Fixed offset", stats, 0, baseline_walks).adjusted_cost(5)


def lookahead_check() -> None:
    costs: dict[str, dict[int, list[int]]] = {}
    for seed in EVALUATION_SEEDS:
        for name, trace in make_workloads(seed).items():
            baseline_walks = simulate(trace, COMPARISON_CONFIG).demand_walks_issued
            by_offset = costs.setdefault(name, {offset: [] for offset in OFFSETS})
            for offset in OFFSETS:
                by_offset[offset].append(fixed_offset_cost(trace, offset, baseline_walks))

    print("Exploratory post-hoc fixed-offset diagnostic on the existing evaluation traces.")
    print("Mean adjusted cost at 5 cycles per extra walk "
          f"across {len(EVALUATION_SEEDS)} traces per workload.")
    print(f"{'Offset':>7}" + "".join(f"{name:>17}" for name in costs))
    for offset in OFFSETS:
        values = (mean(by_offset[offset]) for by_offset in costs.values())
        print(f"{offset:>+7d}" + "".join(f"{value:>17,.1f}" for value in values))


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    options = parser.add_mutually_exclusive_group()
    options.add_argument("--development", action="store_true", help="Run development traces only")
    options.add_argument("--plot", action="store_true", help="Plot saved evaluation results")
    options.add_argument("--sequential-check", action="store_true",
                         help="Run the post-hoc fixed +7 diagnostic on sequential traces")
    options.add_argument("--lookahead-check", action="store_true",
                         help="Run the exploratory post-hoc fixed-offset sweep")
    args = parser.parse_args()
    if args.plot:
        plot_results()
        return
    if args.lookahead_check:
        lookahead_check()
        return
    if args.sequential_check:
        costs = []
        for seed in EVALUATION_SEEDS:
            trace = make_workloads(seed)["Sequential"]
            baseline_walks = simulate(trace, COMPARISON_CONFIG).demand_walks_issued
            costs.append(fixed_offset_cost(trace, 7, baseline_walks))
        print(f"Post-hoc sequential fixed +7 costs {costs} | mean {mean(costs):,.0f}")
        return

    development = list(experiment("Development", LONG_SEEDS))
    summarize(development, "Development")
    if not args.development:
        evaluation = list(experiment("Evaluation", EVALUATION_SEEDS))
        summarize(evaluation, "Evaluation")
        write_csv(RESULT_PATH, CSV_FIELDS, development + evaluation)


if __name__ == "__main__":
    main()
