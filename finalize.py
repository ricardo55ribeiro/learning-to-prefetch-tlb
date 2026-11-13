"""Reproduce the policy-seed sweep and the separate mixed-phase diagnostic."""

from collections.abc import Iterable, Iterator
from csv import DictWriter
from random import Random
from statistics import mean, pstdev

from compare import (
    COMPARISON_CONFIG, CSV_COLUMNS, HELD_OUT_SEEDS, LONG_GAPS, best_fixed_at_five,
    compare, csv_row, make_irregular_region, make_long_workloads, run_reward_selector,
)
from simulator import Access


POLICY_SEEDS = range(100)
DIAGNOSTIC_SEEDS = range(201, 206)
SWEEP_FIELDS = (
    "workload", "workload_seed", "policy_seed", "reward_cost_at_5",
    "contextual_cost_at_5", "best_fixed_at_5", "best_fixed_cost_at_5",
)


def sweep() -> Iterator[dict[str, int | str]]:
    for workload_seed in HELD_OUT_SEEDS:
        for name, trace in make_long_workloads(workload_seed).items():
            fixed = compare(trace)
            best_fixed = best_fixed_at_five(fixed)
            baseline = fixed[0].total_walks

            for policy_seed in POLICY_SEEDS:
                reward = run_reward_selector(
                    trace, COMPARISON_CONFIG, baseline, contextual=False, policy_seed=policy_seed,
                )
                contextual = run_reward_selector(
                    trace, COMPARISON_CONFIG, baseline, contextual=True, policy_seed=policy_seed,
                )
                yield {
                    "workload": name,
                    "workload_seed": workload_seed,
                    "policy_seed": policy_seed,
                    "reward_cost_at_5": reward.adjusted_cost(5),
                    "contextual_cost_at_5": contextual.adjusted_cost(5),
                    "best_fixed_at_5": best_fixed.name,
                    "best_fixed_cost_at_5": best_fixed.adjusted_cost(5),
                }


def make_mixed_trace(seed: int) -> tuple[Access, ...]:
    random = Random(seed)
    pattern = list(range(3000, 3024))
    random.shuffle(pattern)

    accesses = []
    for round_index in range(3):
        base = 100 + 400 * round_index
        for index in range(48):
            accesses.append(Access(base + index, index % 2, random.choice(LONG_GAPS)))
        for index in range(48):
            pc = index % 2
            vpn = base + 48 + 7 * (index // 2) + pc * 170
            accesses.append(Access(vpn, pc, random.choice(LONG_GAPS)))
        accesses.extend(
            Access(vpn, index % 3, random.choice(LONG_GAPS))
            for index, vpn in enumerate(pattern * 3)
        )
        accesses.extend(make_irregular_region(random))
    return tuple(accesses)


def diagnostic() -> Iterator[dict[str, str | int | None]]:
    for seed in DIAGNOSTIC_SEEDS:
        policies = compare(make_mixed_trace(seed), include_adaptive=True, include_reward=True)
        best_fixed = best_fixed_at_five(policies)
        for result in policies:
            yield csv_row("Diagnostic", "Mixed phase", seed, result, best_fixed)


def print_sweep(rows: list[dict[str, int | str]]) -> None:
    print("Held-out sweep | 100 paired policy seeds per trace | cost at 5")
    for name in ("Streams", "Loops"):
        family = [row for row in rows if row["workload"] == name]
        totals = {seed: [0, 0] for seed in POLICY_SEEDS}
        for row in family:
            reward = int(row["reward_cost_at_5"])
            contextual = int(row["contextual_cost_at_5"])
            pair = totals[int(row["policy_seed"])]
            pair[0] += reward
            pair[1] += contextual
        print(name)
        for index, policy in enumerate(("Reward", "Contextual")):
            scores = [pair[index] for pair in totals.values()]
            print(f"  {policy:<11} mean {mean(scores):,.2f} | SD {pstdev(scores):,.2f}"
                  f" | range {min(scores):,} to {max(scores):,}")
        wins = sum(contextual < reward for reward, contextual in totals.values())
        ties = sum(contextual == reward for reward, contextual in totals.values())
        beats_fixed = sum(
            int(row["contextual_cost_at_5"]) < int(row["best_fixed_cost_at_5"]) for row in family
        )
        print(f"  Contextual wins {wins}/{len(totals)} | ties {ties}"
              f" | beats best fixed on {beats_fixed}/{len(family)} individual runs")


def print_diagnostic(rows: list[dict[str, str | int | None]]) -> None:
    groups = {}
    for row in rows:
        groups.setdefault(row["policy"], []).append(row)
    print()
    print(f"Mixed-phase diagnostic | workload seeds {tuple(DIAGNOSTIC_SEEDS)} | policy seed 0")
    print(f"{'Policy':<16}{'Wait':>10}{'Walks':>9}{'Extra':>9}{'Cost@5':>10}")
    fields = ("demand_wait_cycles", "total_walks", "additional_walks", "cost_at_5")
    for policy, results in groups.items():
        wait, walks, extra, cost = (sum(result[field] for result in results) for field in fields)
        print(f"{policy:<16}{wait:>10,}{walks:>9,}{extra:>9,}{cost:>10,}")
    best_cost_by_seed = {row["workload_seed"]: row["best_fixed_cost_at_5"] for row in rows}
    print(f"Retrospective best fixed total at 5 {sum(best_cost_by_seed.values()):,}")


def write_csv(
    path: str, fields: tuple[str, ...], rows: Iterable[dict[str, str | int | None]],
) -> None:
    with open(path, "w", encoding="utf-8", newline="") as output:
        csv = DictWriter(output, fields)
        csv.writeheader()
        csv.writerows(rows)


def main() -> None:
    sweep_rows = list(sweep())
    diagnostic_rows = list(diagnostic())
    write_csv("seed_sweep.csv", SWEEP_FIELDS, sweep_rows)
    write_csv("mixed_phase.csv", CSV_COLUMNS, diagnostic_rows)
    print_sweep(sweep_rows)
    print_diagnostic(diagnostic_rows)


if __name__ == "__main__":
    main()
