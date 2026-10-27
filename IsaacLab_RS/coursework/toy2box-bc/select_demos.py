"""Build one training HDF5 from several session files, leaving chosen demos out.

Usage:
    python select_demos.py OUT.hdf5 FILE1.hdf5[:skip,skip] FILE2.hdf5 ...

Example (skip demo_2 of session 3 and demo_4 of session 4):
    python select_demos.py datasets/team03_v1_20.hdf5 \
        datasets/team03_hahm_s1.hdf5 datasets/team03_hahm_s2.hdf5 \
        datasets/team03_hahm_s3.hdf5:2 datasets/team03_hahm_s4.hdf5:4 datasets/team03_hahm_s5.hdf5

Demos are copied unchanged (actions, obs, states, initial_state, attrs) and
renumbered demo_0.. in order. File attrs (env_args) come from the first input;
data.attrs["total"] is recomputed. The source files are never modified.
"""

import sys

import h5py


def main(out, specs):
    if len(specs) < 1:
        print(__doc__)
        sys.exit(1)
    with h5py.File(out, "w") as fo:
        go = fo.create_group("data")
        n = 0
        total = 0
        origin = []
        for i, spec in enumerate(specs):
            path, _, skip = spec.partition(":")
            skip = {int(s) for s in skip.split(",") if s}
            with h5py.File(path, "r") as fi:
                if i == 0:
                    for k, v in fi["data"].attrs.items():
                        if k != "total":
                            go.attrs[k] = v
                demos = sorted(fi["data"], key=lambda k: int(k.split("_")[1]))
                for name in demos:
                    idx = int(name.split("_")[1])
                    if idx in skip:
                        print(f"skip  {path}:{name}")
                        continue
                    fi.copy(fi["data"][name], go, name=f"demo_{n}")
                    total += int(fi["data"][name].attrs["num_samples"])
                    origin.append(f"demo_{n} <- {path}:{name}")
                    n += 1
        go.attrs["total"] = total
        go.attrs["source"] = "\n".join(origin)
    print(f"wrote {out}: {n} demos, {total} steps")
    for line in origin:
        print("  " + line)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
