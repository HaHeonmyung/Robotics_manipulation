#!/bin/bash
# After the filt70/rand70/ckpt chain: rand500 (outlier check) then multi20_1000 (operator diversity vs demo count)
cd ~/Robotics_manipulation/IsaacLab_RS
R=coursework/toy2box-bc/results
while [ ! -e $R/filt_done ]; do sleep 120; done
sleep 60
source ~/anaconda3/etc/profile.d/conda.sh && conda activate lerobot-arena
export PYTHONUNBUFFERED=1
python - <<PY > $R/src_stats/build_rand500.log 2>&1
import h5py, random
random.seed(0)
with h5py.File("datasets/multi78_1000_mimic_1000.hdf5","r") as fi, h5py.File("datasets/multi78_rand500.hdf5","w") as fo:
    go=fo.create_group("data")
    for k,v in fi["data"].attrs.items():
        if k!="total": go.attrs[k]=v
    names=sorted(fi["data"], key=lambda s:int(s.split("_")[1])); pick=sorted(random.sample(names,500), key=lambda s:int(s.split("_")[1])); tot=0
    for i,n in enumerate(pick):
        fi.copy(fi["data"][n], go, name=f"demo_{i}"); tot+=int(fi["data"][n].attrs["num_samples"])
    go.attrs["total"]=tot; go.attrs["subset"]="random 500 (seed 0) of multi78_1000_mimic_1000.hdf5: "+" ".join(pick)
    print("rand500", len(pick), tot)
PY
coursework/toy2box-bc/run_baseline.sh datasets/multi78_rand500.hdf5 rand500 500 --skip-mimic > $R/rand500.log 2>&1
python - <<PY > $R/src_stats/build_multi20.log 2>&1
import h5py, random, collections
random.seed(0)
f=h5py.File("datasets/multi78.hdf5","r"); src=f["data"].attrs["source"].split(chr(10)); f.close()
groups=collections.defaultdict(list)
for line in src:
    d, rest = line.split(" <- "); groups[rest.split(":")[0]].append(int(d.split("_")[1]))
keep=[]
for g in sorted(groups): keep += random.sample(groups[g],5)
skip=sorted(set(range(78))-set(keep))
print("groups", {g:len(v) for g,v in groups.items()}); print("keep", sorted(keep)); print("skip", ",".join(map(str,skip)))
open("$R/src_stats/multi20_skip.txt","w").write(",".join(map(str,skip)))
PY
SKIP=$(cat $R/src_stats/multi20_skip.txt)
python -I coursework/toy2box-bc/select_demos.py datasets/multi20.hdf5 datasets/multi78.hdf5:$SKIP >> $R/src_stats/build_multi20.log 2>&1
coursework/toy2box-bc/run_baseline.sh datasets/multi20.hdf5 multi20_1000 1000 > $R/multi20_1000.log 2>&1
echo AFTER_DONE > $R/after_filt_done
