#!/bin/bash
# dose-response point: random 750 of the multi78 Mimic 1000 (seed 0), same training/eval
cd ~/Robotics_manipulation/IsaacLab_RS
R=coursework/toy2box-bc/results
while [ ! -e $R/rep_done ]; do sleep 120; done
sleep 60
source ~/anaconda3/etc/profile.d/conda.sh && conda activate lerobot-arena
export PYTHONUNBUFFERED=1
python - <<PY > $R/src_stats/build_rand750.log 2>&1
import h5py, random
random.seed(0)
with h5py.File("datasets/multi78_1000_mimic_1000.hdf5","r") as fi, h5py.File("datasets/multi78_rand750.hdf5","w") as fo:
    go=fo.create_group("data")
    for k,v in fi["data"].attrs.items():
        if k!="total": go.attrs[k]=v
    names=sorted(fi["data"], key=lambda s:int(s.split("_")[1])); pick=sorted(random.sample(names,750), key=lambda s:int(s.split("_")[1])); tot=0
    for i,n in enumerate(pick):
        fi.copy(fi["data"][n], go, name=f"demo_{i}"); tot+=int(fi["data"][n].attrs["num_samples"])
    go.attrs["total"]=tot; go.attrs["subset"]="random 750 (seed 0) of multi78_1000_mimic_1000.hdf5"
    print("rand750", len(pick), tot)
PY
coursework/toy2box-bc/run_baseline.sh datasets/multi78_rand750.hdf5 rand750 750 --skip-mimic > $R/rand750.log 2>&1
echo R750_DONE > $R/rand750_done
