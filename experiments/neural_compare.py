"""Compare two neural offset learners with the table and fixed prefetchers."""

from argparse import ArgumentParser
from collections import defaultdict
from collections.abc import Iterator
from csv import DictReader
from json import dumps
from statistics import mean, pstdev

import numpy as np

from experiments.compare import (
    COMPARISON_CONFIG, CSV_COLUMNS, DATA_DIR, MARKOV_METADATA_BOUND, STRIDE_METADATA_BOUND,
    PolicyResult, best_fixed_at_five, csv_row,
)
from experiments.direct_compare import DIRECT, LAST_DELTA, _OffsetCounts, make_workloads
from experiments.finalize import write_csv
from tlb_prefetch.direct_offset import MAX_PCS, DirectOffsetBandit, LastDelta
from tlb_prefetch.neural_offset import (
    HIDDEN, INPUTS, INIT_SCALE, LinearOffsetLearner, TinyMLPOffsetLearner,
)
from tlb_prefetch.predictors import MarkovPrefetcher, NextPagePrefetcher, PerPCStridePrefetcher
from tlb_prefetch.simulator import Access, Predictor, PrefetchFeedback, SimulatorConfig, simulate


DEVELOPMENT_SEEDS = (211, 223, 227)
EVALUATION_SEEDS = range(401, 411)
POLICY_SEEDS = range(10)
DECISION_CHARGES = (0, 1, 5, 20)
RESULT_PATH = DATA_DIR / "neural_offset.csv"
LINEAR = "Linear"
MLP = "Tiny MLP"
FIXED_4 = "Fixed +4"
FIXED_7 = "Fixed +7"
LEARNERS = (DIRECT, LINEAR, MLP)
CSV_FIELDS = CSV_COLUMNS + (
    "model_configuration", "abstentions", "offset_counts", "useful_offset_counts",
    "decisions", "learning_updates", "peak_pending_feedback", "first_useful_after_decisions",
    "parameters", "dense_inference_macs", "selected_update_forward_macs",
    "update_gradient_multiplications", "update_parameter_subtractions",
    *(f"cost_at_5_plus_{charge}_per_neural_decision" for charge in DECISION_CHARGES),
)


class _FixedOffset:
    def __init__(self, offset: int) -> None:
        self._offset = offset

    def reset(self) -> None:
        pass

    def observe(self, access: Access) -> int | None:
        target = access.vpn + self._offset
        return target if target >= 0 else None


class _Measurements(_OffsetCounts):
    """Record policy behavior without adding state to the predictors."""

    def reset(self) -> None:
        super().reset()
        self.decisions = 0
        self.updates = 0
        self.peak_pending = 0
        self.first_useful: int | None = None

    def observe(self, access: Access) -> int | None:
        self.decisions += 1
        return super().observe(access)

    def on_feedback(self, event: PrefetchFeedback) -> None:
        super().on_feedback(event)
        if event.kind in ("timely", "late") and self.first_useful is None:
            self.first_useful = self.decisions
        if isinstance(self.predictor, (DirectOffsetBandit, LinearOffsetLearner, TinyMLPOffsetLearner)):
            self.predictor.on_feedback(event)
            self.peak_pending = max(self.peak_pending, len(self._accepted))
            if event.kind != "accepted":
                self.updates += 1


def compare_trace(
    split: str, family: str, seed: int, trace: tuple[Access, ...],
    config: SimulatorConfig = COMPARISON_CONFIG,
) -> Iterator[dict[str, int | str | None]]:
    baseline = simulate(trace, config)
    baseline_walks = baseline.demand_walks_issued
    live_bound = config.prefetch_buffer_capacity + config.max_pending_prefetches

    def run(
        name: str, predictor: Predictor, metadata_bound: int, checks: int = 0,
    ) -> tuple[PolicyResult, _Measurements]:
        measured = _Measurements(predictor)
        stats = simulate(trace, config, predictor=measured, on_feedback=measured.on_feedback)
        assert not measured._accepted
        if isinstance(predictor, (LinearOffsetLearner, TinyMLPOffsetLearner)):
            assert not predictor._pending
            if any(not np.isfinite(p).all() for p in predictor.parameters):
                raise ArithmeticError(f"Nonfinite parameters for {name} on {family}, workload seed {seed}")
            retained = predictor.parameter_count + len(predictor._history.entries)
        else:
            retained = getattr(predictor, "metadata_entries", 0)
        result = PolicyResult(name, stats, retained, baseline_walks, metadata_bound, 0, checks)
        return result, measured

    fixed = [
        (PolicyResult("No prefetch", baseline, 0, baseline_walks), None),
        run("Next page", NextPagePrefetcher(), 0),
        run("Per-PC stride", PerPCStridePrefetcher(), STRIDE_METADATA_BOUND),
        run("Markov", MarkovPrefetcher(), MARKOV_METADATA_BOUND),
        run(LAST_DELTA, LastDelta(), MAX_PCS),
        run(FIXED_4, _FixedOffset(4), 0),
        run(FIXED_7, _FixedOffset(7), 0),
    ]
    conventional_best = best_fixed_at_five(tuple(result for result, _ in fixed))

    def row(
        result: PolicyResult, measured: _Measurements | None, policy_seed: int | None = None,
    ) -> dict[str, int | str | None]:
        record = csv_row(split, family, seed, result, conventional_best)
        record["policy_seed"] = policy_seed
        stats = result.stats
        configurations = {
            DIRECT: "40x17 table, alpha=0.2, epsilon=0.1",
            LINEAR: f"{INPUTS}x16, float64, zero init, alpha=0.05, epsilon=0.1",
            MLP: f"{INPUTS}x{HIDDEN}x16 tanh, float64, hidden uniform +/-{INIT_SCALE}, "
                 "zero biases and output, alpha=0.05, epsilon=0.1",
        }
        # Count dense arithmetic even when most input features are zero
        parameters, forward, update_forward, gradient, parameter_updates = {
            LINEAR: (352, 336, 21, 22, 22),
            MLP: (624, 592, 352, 401, 369),
        }.get(result.name, (0, 0, 0, 0, 0))
        record.update(
            model_configuration=configurations.get(result.name, result.name),
            abstentions=stats.demand_accesses - stats.prefetch_requests_attempted,
            offset_counts=dumps(dict(measured.offsets), sort_keys=True) if measured else "{}",
            useful_offset_counts=dumps(dict(measured.useful), sort_keys=True) if measured else "{}",
            decisions=stats.demand_accesses,
            learning_updates=measured.updates if measured else 0,
            peak_pending_feedback=measured.peak_pending if measured else 0,
            first_useful_after_decisions=measured.first_useful if measured else None,
            parameters=parameters,
            dense_inference_macs=forward,
            selected_update_forward_macs=update_forward,
            update_gradient_multiplications=gradient,
            update_parameter_subtractions=parameter_updates,
        )
        for charge in DECISION_CHARGES:
            extra = charge * stats.demand_accesses if result.name in (LINEAR, MLP) else 0
            record[f"cost_at_5_plus_{charge}_per_neural_decision"] = result.adjusted_cost(5) + extra
        return record

    for result, measured in fixed:
        yield row(result, measured)
    for policy_seed in POLICY_SEEDS:
        table = DirectOffsetBandit(seed=policy_seed)
        result, measured = run(DIRECT, table, table.value_entries + MAX_PCS + live_bound, 17)
        yield row(result, measured, policy_seed)
        for name, model_type in ((LINEAR, LinearOffsetLearner), (MLP, TinyMLPOffsetLearner)):
            model = model_type(seed=policy_seed, page_walk_latency=config.page_walk_latency)
            result, measured = run(name, model, model.parameter_count + MAX_PCS + live_bound, 17)
            yield row(result, measured, policy_seed)


def experiment(split: str, seeds: tuple[int, ...] | range) -> Iterator[dict[str, int | str | None]]:
    for seed in seeds:
        for name, trace in make_workloads(seed).items():
            yield from compare_trace(split, name, seed, trace)


def summarize(rows: list[dict[str, int | str | None]], split: str) -> None:
    costs = defaultdict(list)
    for row in rows:
        costs[row["workload"], row["policy"]].append(row["cost_at_5"])
    print(f"\n{split} mean adjusted cost at 5 cycles per extra walk")
    labels = ("No prefetch", "Next page", "Per-PC stride", "Markov", LAST_DELTA,
              FIXED_4, FIXED_7, *LEARNERS)
    print(f"{'Workload':<17}" + "".join(f"{label:>15}" for label in labels))
    families = tuple(dict.fromkeys(family for family, _ in costs))
    for family in families:
        print(f"{family:<17}" + "".join(f"{mean(costs[family, label]):>15,.1f}" for label in labels))
    print("Population SD over workload-policy seed pairs")
    for family in families:
        values = ", ".join(f"{label} {pstdev(costs[family, label]):,.1f}" for label in LEARNERS)
        print(f"{family:<17}{values}")


def plot_results() -> None:
    from experiments.plot_results import DARK_STYLE, FIGURE_DIR, plt
    from matplotlib.ticker import StrMethodFormatter

    costs = defaultdict(list)
    with RESULT_PATH.open(encoding="utf-8", newline="") as source:
        for row in DictReader(source):
            if row["split"] == "Evaluation":
                costs[row["workload"], row["policy"]].append(int(row["cost_at_5"]))
    families = list(dict.fromkeys(family for family, _ in costs))
    policies = (
        (FIXED_4, "#C4CBD4"), (FIXED_7, "#738499"), (DIRECT, "#F2AE66"),
        (LINEAR, "#58A6FF"), (MLP, "#B5A6FA"),
    )
    with plt.rc_context(DARK_STYLE):
        fig, ax = plt.subplots(figsize=(9, 6), layout="constrained")
        ax.set_title("Direct offset learning on fresh synthetic traces", loc="left", pad=13)
        for index, (policy, color) in enumerate(policies):
            positions = [row + (index - 2) * 0.15 for row in range(len(families))]
            values = [mean(costs[family, policy]) for family in families]
            ax.barh(positions, values, 0.133, color=color, label=policy)
        ax.set_yticks(range(len(families)), families)
        ax.invert_yaxis()
        ax.set_xlim(left=0)
        ax.xaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
        ax.set_xlabel("Mean simulated cost at 5 cycles per extra walk")
        ax.grid(axis="x")
        ax.legend(loc="upper left", bbox_to_anchor=(0, -0.13), frameon=False, ncols=3)
        FIGURE_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(FIGURE_DIR / "neural_offset.png", dpi=180, transparent=False)
        plt.close(fig)


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    options = parser.add_mutually_exclusive_group()
    options.add_argument("--development", action="store_true", help="Check development traces only")
    options.add_argument("--plot", action="store_true", help="Plot saved evaluation results")
    args = parser.parse_args()
    if args.plot:
        plot_results()
        return
    print(f"NumPy {np.__version__}")
    development = list(experiment("Development", DEVELOPMENT_SEEDS))
    summarize(development, "Development")
    if not args.development:
        evaluation = list(experiment("Evaluation", EVALUATION_SEEDS))
        summarize(evaluation, "Evaluation")
        write_csv(RESULT_PATH, CSV_FIELDS, development + evaluation)


if __name__ == "__main__":
    main()
