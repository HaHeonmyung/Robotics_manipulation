#!/bin/bash
cd ~/Robotics_manipulation/IsaacLab_RS
source ~/anaconda3/etc/profile.d/conda.sh && conda activate lerobot-arena
export PYTHONUNBUFFERED=1
# condition 1: 78 human demos, Mimic 1000
coursework/toy2box-bc/run_baseline.sh datasets/multi78.hdf5 multi78_1000 1000 > coursework/toy2box-bc/results/multi78_1000.log 2>&1
# condition 2: first 500 of the same generated demos
python - <<PY
import h5py
with h5py.File("datasets/multi78_1000_mimic_1000.hdf5","r") as fi, h5py.File("datasets/multi78_500_from1000.hdf5","w") as fo:
    go=fo.create_group("data")
    for k,v in fi["data"].attrs.items():
        if k!="total": go.attrs[k]=v
    names=sorted(fi["data"], key=lambda s:int(s.split("_")[1]))[:500]; tot=0
    for n in names:
        fi.copy(fi["data"][n], go, name=n); tot+=int(fi["data"][n].attrs["num_samples"])
    go.attrs["total"]=tot; go.attrs["subset"]="first 500 of multi78_1000_mimic_1000.hdf5"
    print("subset", len(names), tot)
PY
coursework/toy2box-bc/run_baseline.sh datasets/multi78_500_from1000.hdf5 multi78_500 500 --skip-mimic > coursework/toy2box-bc/results/multi78_500.log 2>&1
echo CHAIN_DONE > coursework/toy2box-bc/results/multi78_done
