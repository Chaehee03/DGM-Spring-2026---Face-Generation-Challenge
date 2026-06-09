#!/bin/bash
#SBATCH --job-name=dgm_align_i4
#SBATCH --partition=R3090
#SBATCH --qos=new_default
#SBATCH --array=0-3
#SBATCH --exclude=soda-03
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --output=/workspace/chsong/dgm-face/logs/%A_%a.out
#SBATCH --error=/workspace/chsong/dgm-face/logs/%A_%a.err

export PYTHONUNBUFFERED=1

source /workspace/chsong/miniconda3/bin/activate stylegan

export LD_LIBRARY_PATH=$(
python - <<'PY'
import site
from pathlib import Path

paths = []
for sp in site.getsitepackages():
    root = Path(sp) / "nvidia"
    if root.exists():
        for libdir in root.glob("*/lib"):
            if libdir.exists():
                paths.append(str(libdir))
print(":".join(paths))
PY
):$LD_LIBRARY_PATH

cd /workspace/chsong/dgm-face

echo "Start: $(date)"
echo "Array job: $SLURM_ARRAY_JOB_ID"
echo "Task id: $SLURM_ARRAY_TASK_ID"
echo "Node: $SLURM_NODELIST"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"

nvidia-smi

python - <<'PY'
import onnxruntime as ort
print("ORT version:", ort.__version__, flush=True)
print("ORT providers:", ort.get_available_providers(), flush=True)
assert "CUDAExecutionProvider" in ort.get_available_providers(), "CUDAExecutionProvider not available"
PY

python scripts/align_faces_insight.py \
  --input-dir /data/chsong/dgm-face/frames_raw_3f_0205080 \
  --out-dir /data/chsong/dgm-face/faces_256_insight_3f_std \
  --meta-path /data/chsong/dgm-face/metadata/aligned_faces_insight_3f_std_shard${SLURM_ARRAY_TASK_ID}.csv \
  --size 256 \
  --det-size 640,640 \
  --ctx-id 0 \
  --no-sort \
  --flush-every 1000 \
  --shard-id ${SLURM_ARRAY_TASK_ID} \
  --num-shards 4

echo "Done shard ${SLURM_ARRAY_TASK_ID}: $(date)"
