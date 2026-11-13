"""Compare two runs of the same short trace."""

from tlb_prefetch.simulator import Access, simulate


def main() -> None:
    trace = (
        Access(vpn=100, pc=1, compute_cycles=40),
        Access(vpn=200, pc=2, compute_cycles=20),
        Access(vpn=300, pc=3, compute_cycles=60),
    )
    baseline = simulate(trace)
    scripted = simulate(trace, prefetch_plan={0: 200, 1: 300})

    rows = (
        ("Demand waiting cycles", baseline.demand_wait_cycles, scripted.demand_wait_cycles),
        ("Elapsed cycles", baseline.elapsed_cycles, scripted.elapsed_cycles),
        ("Demand walks", baseline.demand_walks_issued, scripted.demand_walks_issued),
        ("Prefetch walks", baseline.extra_prefetch_walks, scripted.extra_prefetch_walks),
        ("Timely prefetches", baseline.timely_prefetches_consumed, scripted.timely_prefetches_consumed),
        ("Late prefetches", baseline.late_prefetches_consumed, scripted.late_prefetches_consumed),
    )
    print(f"{'Metric':<24}{'No prefetch':>12}{'Scripted':>12}")
    for label, without_prefetch, with_prefetch in rows:
        print(f"{label:<24}{without_prefetch:>12}{with_prefetch:>12}")

    print()
    print("The demand for VPN 200 arrives at cycle 120. Its prefetch finishes at 160.")
    print("The demand for VPN 300 arrives at cycle 220, when its prefetch finishes.")


if __name__ == "__main__":
    main()
