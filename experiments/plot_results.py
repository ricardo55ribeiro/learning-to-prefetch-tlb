"""Render the publication figures from the saved finalization CSV files."""

from collections import defaultdict
from csv import DictReader

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.ticker import StrMethodFormatter

from experiments.compare import DATA_DIR


FIGURE_DIR = DATA_DIR.parent / "figures"
TEXT = "#E6EDF3"
SECONDARY_TEXT = "#9BA8B9"
BLUE = "#58A6FF"
AMBER = "#F2AE66"
DARK_STYLE = {
    "figure.facecolor": "#0D1117",
    "axes.facecolor": "#161B22",
    "axes.edgecolor": "#30363D",
    "axes.labelcolor": TEXT,
    "axes.titlecolor": TEXT,
    "axes.titlesize": 15,
    "axes.titleweight": "bold",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.axisbelow": True,
    "text.color": TEXT,
    "xtick.color": SECONDARY_TEXT,
    "ytick.color": SECONDARY_TEXT,
    "grid.color": "#30363D",
    "grid.linewidth": 0.8,
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "figure.constrained_layout.w_pad": 0.18,
    "figure.constrained_layout.h_pad": 0.16,
}


def plot_seed_sweep() -> None:
    paired_differences = defaultdict(lambda: defaultdict(int))
    with (DATA_DIR / "seed_sweep.csv").open(encoding="utf-8", newline="") as source:
        for row in DictReader(source):
            difference = int(row["contextual_cost_at_5"]) - int(row["reward_cost_at_5"])
            paired_differences[row["workload"]][row["policy_seed"]] += difference

    figure, ax = plt.subplots(figsize=(8.5, 4.5), layout="constrained")
    ax.set_title("Effect of the policy seed on held-out cost", loc="left", pad=28)
    ax.text(0, 1.025, "Negative favors contextual", transform=ax.transAxes,
            color=SECONDARY_TEXT, va="bottom")
    ax.text(1, 1.025, "Positive favors noncontextual", transform=ax.transAxes,
            color=SECONDARY_TEXT, va="bottom", ha="right")
    ax.axvline(0, color=SECONDARY_TEXT, linestyle=":", linewidth=1.3)
    ax.grid(axis="y")
    styles = (("Streams", BLUE, "-"), ("Loops", AMBER, "--"))
    for name, color, line in styles:
        differences = sorted(paired_differences[name].values())
        wins = sum(value < 0 for value in differences)
        cumulative = [100 * index / len(differences) for index in range(len(differences) + 1)]
        ax.step(
            [differences[0], *differences], cumulative, where="post", color=color,
            linestyle=line, linewidth=2.3, label=f"{name} ({wins}/{len(differences)} contextual wins)",
        )
    ax.set_ylim(-3, 103)
    ax.set_yticks(range(0, 101, 25))
    ax.xaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    ax.set_xlabel("Contextual minus noncontextual cost across 10 traces (cycles)", labelpad=9)
    ax.set_ylabel("Cumulative share of policy seeds (%)")
    ax.legend(loc="lower right", frameon=False)
    figure.savefig(FIGURE_DIR / "seed_sensitivity.png", dpi=180, transparent=False)
    plt.close(figure)


def plot_mixed_phase() -> None:
    totals = defaultdict(lambda: [0, 0])
    with (DATA_DIR / "mixed_phase.csv").open(encoding="utf-8", newline="") as source:
        for row in DictReader(source):
            pair = totals[row["policy"]]
            pair[0] += int(row["demand_wait_cycles"])
            pair[1] += int(row["cost_at_5"]) - int(row["demand_wait_cycles"])

    names = {
        "Next page": "Next-page",
        "Adaptive": "Adaptive heuristic",
        "Reward": "Noncontextual reward",
        "Contextual": "Contextual bandit",
    }
    labels = [names.get(name, name) for name in totals]
    wait, penalty = zip(*totals.values())

    figure, ax = plt.subplots(figsize=(8.5, 4.8), layout="constrained")
    ax.set_title("Mixed-phase diagnostic", loc="left", pad=28)
    ax.text(0, 1.025, "Totals across five synthetic traces", transform=ax.transAxes,
            color=SECONDARY_TEXT, va="bottom")
    ax.grid(axis="x")
    ax.barh(labels, wait, height=0.65, color=BLUE, label="Demand waiting")
    ax.barh(labels, penalty, height=0.65, left=wait, color=AMBER, label="Extra-walk charge")
    lowest_cost = min(sum(pair) for pair in totals.values())
    for index, (waiting, charge) in enumerate(zip(wait, penalty)):
        cost = waiting + charge
        ax.annotate(
            f"{cost:,}", (cost, index), xytext=(6, 0), textcoords="offset points",
            va="center", fontweight="bold" if cost == lowest_cost else "normal",
        )
        if cost == lowest_cost:
            label = ax.get_yticklabels()[index]
            label.set_fontweight("bold")
            label.set_color(TEXT)
    ax.set_xlim(0, max(sum(pair) for pair in totals.values()) * 1.16)
    ax.invert_yaxis()
    ax.set_xticks(range(0, 200001, 50000))
    ax.xaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    ax.set_xlabel("Simulated cost (hypothetical 5 cycles per extra walk)", labelpad=9)
    ax.legend(loc="upper left", bbox_to_anchor=(0, -0.17), frameon=False, ncols=2)
    figure.savefig(FIGURE_DIR / "mixed_phase.png", dpi=180, transparent=False)
    plt.close(figure)


if __name__ == "__main__":
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    with plt.rc_context(DARK_STYLE):
        plot_seed_sweep()
        plot_mixed_phase()
