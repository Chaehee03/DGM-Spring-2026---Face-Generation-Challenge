#!/bin/bash
#SBATCH --job-name=dgm_align_i5
#SBATCH --partition=R3090
#SBATCH --qos=new_default
#SBATCH --nodelist=soda-02
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --output=/workspace/chsong/dgm-face/logs/%j.out
#SBATCH --error=/workspace/chsong/dgm-face/logs/%j.err

export PYTHONUNBUFFERED=1
source /workspace/chsong/miniconda3/bin/activate stylegan

cd /workspace/chsong/dgm-face

echo "Start: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

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
  --meta-path /data/chsong/dgm-face/metadata/aligned_faces_insight_3f_std.csv \
  --size 256 \
  --det-size 640,640 \
  --ctx-id 0 \
  --no-sort \
  --flush-every 1000

echo "Aligned face count:"
find /data/chsong/dgm-face/faces_256_insight_3f_std -name "*.png" | wc -l

echo "Done: $(date)"
