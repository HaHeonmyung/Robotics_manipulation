"""Combine play_diagnose.py result files into the numbers the assignment asks for.

Usage:
    python3 coursework/toy2box-bc/aggregate.py results/baseline/eval_seed101.json results/baseline/eval_seed102.json

Prints success rate per training seed, mean +/- std over seeds, and the failure
stage distribution (counts and share of failures) per seed and pooled.
"""

import json
import sys

import statistics

STAGES = ["approach", "grasp", "transport", "place"]


def main(paths):
    rates = []
    pooled = {s: 0 for s in STAGES}
    pooled_fail = 0
    print(f"{'file':<40s} {'succ':>5s} {'rate':>6s}  " + "  ".join(f"{s:>9s}" for s in STAGES))
    for p in paths:
        r = json.load(open(p))
        rates.append(r["success_rate"])
        sc = r["stage_counts"]
        n_fail = r["num_rollouts"] - r["successes"]
        pooled_fail += n_fail
        cells = []
        for s in STAGES:
            c = sc.get(s, 0)
            pooled[s] += c
            cells.append(f"{c:3d}({c / n_fail:4.0%})" if n_fail else f"{c:3d}")
        print(f"{p[-40:]:<40s} {r['successes']:>5d} {r['success_rate']:>6.2f}  " + "  ".join(f"{c:>9s}" for c in cells))
    print()
    print(f"success rate over {len(rates)} seed(s): {statistics.mean(rates):.3f} +/- {(statistics.pstdev(rates) if len(rates) > 1 else 0.0):.3f}")
    if pooled_fail:
        print("failure stages pooled: " + ", ".join(f"{s} {pooled[s]} ({pooled[s] / pooled_fail:.0%})" for s in STAGES))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1:])
