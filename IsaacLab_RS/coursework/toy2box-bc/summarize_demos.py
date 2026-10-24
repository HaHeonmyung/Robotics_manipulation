"""Print the per-demo health check from the Week 08 slides (p.35, p.40).

Usage (from IsaacLab_RS):
    ./isaaclab.sh -p coursework/toy2box-bc/summarize_demos.py datasets/team03_kim_s1.hdf5 [more.hdf5 ...]

Columns: length in steps and seconds (20 Hz), gripper switches (2 = one grasp, one
release), idle ratio (no movement key pressed), success flag. Demos that break the
recommended ranges from slide 40 are marked.
"""

import sys

import h5py
import numpy as np


def summarize(path: str) -> None:
    f = h5py.File(path, "r")
    D = f["data"]
    demos = sorted(D, key=lambda k: int(k.split("_")[1]))
    print(f"\n{path}")
    print(f"episodes: {len(demos)} | total: {D.attrs.get('total', 'n/a')}")
    lens, flags = [], 0
    for k in demos:
        a = D[k]["actions"][:]
        n = int(D[k].attrs["num_samples"])
        ok = int(D[k].attrs.get("success", 1))
        sw = int((np.diff(a[:, 6]) != 0).sum())
        idle = float((a[:, :6] == 0).all(1).mean())
        marks = []
        if n / 20 > 40:
            marks.append("long")
        if sw >= 4:
            marks.append("regrasp")
        if idle > 0.30:
            marks.append("idle")
        if ok != 1:
            marks.append("NOT-SUCCESS")
        flags += bool(marks)
        lens.append(n / 20)
        print(f"{k:<8s} len={n:4d} ({n / 20:5.1f}s) sw={sw:2d} idle={idle:4.0%} ok={ok}  {' '.join(marks)}")
    if lens:
        print(f"length s: mean {np.mean(lens):.1f} min {np.min(lens):.1f} max {np.max(lens):.1f} | flagged: {flags}/{len(demos)}")
    f.close()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    for p in sys.argv[1:]:
        summarize(p)
