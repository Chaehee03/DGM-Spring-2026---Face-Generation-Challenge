#!/bin/bash
#SBATCH --job-name=dgm_hf_dl
#SBATCH --partition=R3090
#SBATCH --qos=new_default
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=2-00:00:00
#SBATCH --output=/workspace/chsong/dgm-face/logs/%j.out
#SBATCH --error=/workspace/chsong/dgm-face/logs/%j.err

export PYTHONUNBUFFERED=1
export HF_HUB_ENABLE_HF_TRANSFER=1

source /workspace/chsong/miniconda3/bin/activate stylegan

mkdir -p /data/chsong/dgm-face/celebvhq_hf
cd /data/chsong/dgm-face/celebvhq_hf

echo "Start: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"

python - <<'PY'
from huggingface_hub import snapshot_download

local_dir = "/data/chsong/dgm-face/celebvhq_hf"

snapshot_download(
    repo_id="SwayStar123/CelebV-HQ",
    repo_type="dataset",
    local_dir=local_dir,
    local_dir_use_symlinks=False,
    resume_download=True,
)
print("Downloaded to", local_dir)
PY

echo "Done: $(date)"
du -sh /data/chsong/dgm-face/celebvhq_hf || true
find /data/chsong/dgm-face/celebvhq_hf -type f | head
