"""Render the publication figures from the saved finalization CSV files."""

from collections import defaultdict
from csv import DictReader

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.ticker import StrMethodFormatter


def plot_seed_sweep() -> None:
    paired_differences = defaultdict(lambda: defaultdict(int))
    with open("seed_sweep.csv", encoding="utf-8", newline="") as source:
        for row in DictReader(source):
            difference = int(row["contextual_cost_at_5"]) - int(row["reward_cost_at_5"])
            paired_differences[row["workload"]][row["policy_seed"]] += difference

    figure, ax = plt.subplots(figsize=(8.5, 4.5), layout="constrained")
    ax.set_title("Effect of the policy seed on held-out cost", loc="left", pad=12)
    ax.axvline(0, color="#5d6776", linestyle="--", linewidth=1.2)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#dfe4eb")
    styles = (("Streams", "#245aa5", "-"), ("Loops", "#a14b18", "--"))
    for name, color, line in styles:
        differences = sorted(paired_differences[name].values())
        wins = sum(value < 0 for value in differences)
        cumulative = [100 * index / len(differences) for index in range(len(differences) + 1)]
        ax.step(
            [differences[0], *differences], cumulative, where="post", color=color,
            linestyle=line, linewidth=2.2, label=f"{name} ({wins}/{len(differences)} wins)",
        )
    ax.set_ylim(-3, 103)
    ax.set_yticks(range(0, 101, 25))
    ax.xaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    ax.set_xlabel("Contextual minus noncontextual cost across 10 traces (cycles)")
    ax.set_ylabel("Cumulative share of policy seeds (%)")
    ax.legend(loc="lower right", frameon=False)
    figure.savefig("seed_sensitivity.png", dpi=160)
    plt.close(figure)


def plot_mixed_phase() -> None:
    totals = defaultdict(lambda: [0, 0])
    with open("mixed_phase.csv", encoding="utf-8", newline="") as source:
        for row in DictReader(source):
            pair = totals[row["policy"]]
            pair[0] += int(row["demand_wait_cycles"])
            pair[1] += int(row["cost_at_5"]) - int(row["demand_wait_cycles"])

    names = {"Reward": "Noncontextual reward", "Contextual": "Contextual bandit"}
    labels = [names.get(name, name) for name in totals]
    wait, penalty = zip(*totals.values())

    figure, ax = plt.subplots(figsize=(8.5, 4.5), layout="constrained")
    ax.set_title("Mixed-phase diagnostic across five workload seeds", loc="left", pad=12)
    ax.set_axisbelow(True)
    ax.grid(axis="x", color="#dfe4eb")
    ax.barh(labels, wait, color="#245aa5", label="Demand waiting")
    ax.barh(labels, penalty, left=wait, color="#c78a4d", label="Extra-walk penalty")
    for index, (waiting, charge) in enumerate(zip(wait, penalty)):
        ax.annotate(
            f"{waiting + charge:,}", (waiting + charge, index), xytext=(5, 0),
            textcoords="offset points", va="center",
        )
    ax.set_xlim(0, max(sum(pair) for pair in totals.values()) * 1.13)
    ax.invert_yaxis()
    ax.xaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    ax.set_xlabel("Adjusted cost at 5 cycles per extra walk")
    ax.legend(loc="upper right", bbox_to_anchor=(1, -0.15), frameon=False, ncols=2)
    figure.savefig("mixed_phase.png", dpi=160)
    plt.close(figure)


if __name__ == "__main__":
    plot_seed_sweep()
    plot_mixed_phase()
