"""Cap long idle runs in recorded demos (controlled-experiment data for axis A).

An idle step is one whose six motion commands are all zero. Every run of
consecutive idle steps longer than --keep is shortened to its first --keep
steps, except runs that touch the --protect steps around a gripper switch,
which are left as they are (the pause before closing / after opening is
physically needed). All per-step arrays of a demo (actions, obs/*,
processed_actions, states/*, anything with num_samples rows) are cut with the
same mask, so the file keeps the recorder's layout. Demo count, order and
initial states are unchanged; the source file is never modified.

Usage:
    python trim_idle.py IN.hdf5 OUT.hdf5 [--keep 5] [--protect 10]
"""

import argparse

import h5py
import numpy as np


def keep_mask(actions: np.ndarray, keep: int, protect: int, threshold: float = 0.0) -> np.ndarray:
    n = len(actions)
    # threshold 0 = exactly zero (keyboard demos); Mimic output carries noise, so use e.g. 0.03 there
    idle = (np.abs(actions[:, :6]) <= threshold).all(axis=1)
    switches = np.flatnonzero(np.diff(actions[:, 6]) != 0) + 1
    protected = np.zeros(n, dtype=bool)
    for s in switches:
        protected[max(0, s - protect) : min(n, s + protect + 1)] = True
    mask = np.ones(n, dtype=bool)
    i = 0
    while i < n:
        if not idle[i]:
            i += 1
            continue
        j = i
        while j < n and idle[j]:
            j += 1
        if (j - i) > keep and not protected[i:j].any():
            mask[i + keep : j] = False
        i = j
    return mask


def cut_group(src: h5py.Group, dst: h5py.Group, mask: np.ndarray, n: int) -> None:
    for k, v in src.items():
        if isinstance(v, h5py.Group):
            g = dst.create_group(k)
            for ak, av in v.attrs.items():
                g.attrs[ak] = av
            cut_group(v, g, mask, n)
        else:
            data = v[()]
            if isinstance(data, np.ndarray) and data.ndim >= 1 and data.shape[0] == n:
                data = data[mask]
            dst.create_dataset(k, data=data)
    for ak, av in src.attrs.items():
        dst.attrs[ak] = av


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--keep", type=int, default=5, help="idle steps kept per run (default 5 = 0.25 s)")
    ap.add_argument("--protect", type=int, default=10, help="steps around a gripper switch never trimmed")
    ap.add_argument("--threshold", type=float, default=0.0, help="|motion action| at or below this counts as idle")
    a = ap.parse_args()

    total_in = total_out = 0
    with h5py.File(a.src, "r") as fi, h5py.File(a.dst, "w") as fo:
        go = fo.create_group("data")
        for k, v in fi["data"].attrs.items():
            go.attrs[k] = v
        for name in sorted(fi["data"], key=lambda s: int(s.split("_")[1])):
            demo = fi["data"][name]
            actions = demo["actions"][()]
            n = len(actions)
            mask = keep_mask(actions, a.keep, a.protect, a.threshold)
            g = go.create_group(name)
            cut_group(demo, g, mask, n)
            # initial_state is copied untouched by cut_group (its arrays have 1 row)
            g.attrs["num_samples"] = int(mask.sum())
            total_in += n
            total_out += int(mask.sum())
            idle_in = (np.abs(actions[:, :6]) <= a.threshold).all(1).mean()
            idle_out = (np.abs(actions[mask][:, :6]) <= a.threshold).all(1).mean()
            print(f"{name:<8s} {n:4d} -> {int(mask.sum()):4d} steps  idle {idle_in:4.0%} -> {idle_out:4.0%}")
        go.attrs["total"] = total_out
        go.attrs["trim_idle"] = f"keep={a.keep} protect={a.protect} threshold={a.threshold} src={a.src}"
    print(f"total {total_in} -> {total_out} steps ({1 - total_out / total_in:.0%} removed)")


if __name__ == "__main__":
    main()
