"""Append grasp-window segments to a Mimic training set (experiment: pre-grasp oversampling).

For every demo, the steps from --before steps before the first gripper close to --after
steps after it are copied out as an extra short demo, --copies times (obs, actions,
states cut with the same mask; initial_state copied from the parent). The original demos
are kept unchanged, so only the share of grasp-window samples changes.

Usage: python grasp_window.py IN.hdf5 OUT.hdf5 [--before 60] [--after 20] [--copies 2] [--stats-only]
"""
import argparse, sys
import h5py, numpy as np
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from trim_idle import cut_group
from edit_idle import first_close

ap = argparse.ArgumentParser(); ap.add_argument("src"); ap.add_argument("dst", nargs="?")
ap.add_argument("--before", type=int, default=60); ap.add_argument("--after", type=int, default=20)
ap.add_argument("--copies", type=int, default=2); ap.add_argument("--stats-only", action="store_true")
a = ap.parse_args()
tot = win = 0; n_demos = 0
with h5py.File(a.src, "r") as fi:
    names = sorted(fi["data"], key=lambda s: int(s.split("_")[1]))
    fo = None if a.stats_only else h5py.File(a.dst, "w")
    if fo is not None:
        go = fo.create_group("data")
        for k, v in fi["data"].attrs.items():
            if k != "total": go.attrs[k] = v
    k_out = 0
    for name in names:
        d = fi["data"][name]; act = d["actions"][()]; n = len(act)
        c = first_close(act); s0 = max(0, c - a.before); s1 = min(n, c + a.after + 1)
        tot += n; win += (s1 - s0) if c < n else 0; n_demos += 1
        if fo is None: continue
        fi.copy(d, go, name=f"demo_{k_out}"); k_out += 1
        if c >= n: continue
        mask = np.zeros(n, dtype=bool); mask[s0:s1] = True
        for _ in range(a.copies):
            g = go.create_group(f"demo_{k_out}"); cut_group(d, g, mask, n)
            g.attrs["num_samples"] = int(mask.sum()); g.attrs["grasp_window_of"] = name; k_out += 1
    if fo is not None:
        total = sum(int(go[k].attrs["num_samples"]) for k in go)
        go.attrs["total"] = total
        go.attrs["grasp_window"] = f"before={a.before} after={a.after} copies={a.copies} src={a.src}"
        fo.close()
        print(f"wrote {a.dst}: {k_out} demos, {total} steps")
print(f"{n_demos} demos, {tot} steps, grasp window steps {win} ({win/tot:.1%}); after x{a.copies} copies -> {(win*(1+a.copies))/(tot+win*a.copies):.1%}")
