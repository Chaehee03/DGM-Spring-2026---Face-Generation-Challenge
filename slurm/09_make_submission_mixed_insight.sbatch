#!/bin/bash
#SBATCH --job-name=dgm_make_sub
#SBATCH --partition=R3090
#SBATCH --qos=new_default
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=0-01:00:00
#SBATCH --output=/workspace/chsong/dgm-face/logs/%j.out
#SBATCH --error=/workspace/chsong/dgm-face/logs/%j.err

set -e
export PYTHONUNBUFFERED=1

source /workspace/chsong/miniconda3/bin/activate stylegan
cd /workspace/chsong/dgm-face

BASE=/data/chsong/dgm-face/generated_candidates_insight_sweep
OUTROOT=/data/chsong/dgm-face/submissions/mixed_insight_v1
IMGDIR=${OUTROOT}/images
ZIP=${OUTROOT}/submission.zip
MANIFEST=${OUTROOT}/manifest.csv

mkdir -p "$IMGDIR"
rm -f "$ZIP" "$MANIFEST"
find "$IMGDIR" -type f -delete

python - <<'PY'
from pathlib import Path
import random
import shutil
import csv
import zipfile

random.seed(2026)

base = Path("/data/chsong/dgm-face/generated_candidates_insight_sweep")
outroot = Path("/data/chsong/dgm-face/submissions/mixed_insight_v1")
outdir = outroot / "images"
zip_path = outroot / "submission.zip"
manifest = outroot / "manifest.csv"

spec = [
    ("0.65", "psi_0_65", 300),
    ("0.80", "psi_0_80", 500),
    ("1.00", "psi_1_00", 200),
]

outdir.mkdir(parents=True, exist_ok=True)
outroot.mkdir(parents=True, exist_ok=True)

for p in outdir.glob("*.png"):
    p.unlink()

rows = []
global_idx = 0

for psi, psi_dir, need in spec:
    files = sorted((base / psi_dir).rglob("*.png"))
    print(f"{psi_dir}: found {len(files)} files")

    if len(files) < need:
        raise RuntimeError(f"Not enough files in {psi_dir}: need {need}, found {len(files)}")

    chosen = sorted(random.sample(files, need))

    for src in chosen:
        dst_name = f"img_{global_idx:04d}.png"
        dst = outdir / dst_name
        shutil.copy2(src, dst)
        rows.append([dst_name, psi, str(src)])
        global_idx += 1

print(f"total selected: {global_idx}")

with open(manifest, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["submission_name", "psi", "source_path"])
    writer.writerows(rows)

if zip_path.exists():
    zip_path.unlink()

# Write images directly at zip root, no nested folders.
with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
    for img in sorted(outdir.glob("*.png")):
        z.write(img, arcname=img.name)

with zipfile.ZipFile(zip_path, "r") as z:
    names = z.namelist()
    imgs = [x for x in names if x.lower().endswith(".png")]
    nested = [x for x in imgs if "/" in x.strip("/")]

print(f"zip: {zip_path}")
print(f"zip image count: {len(imgs)}")
print(f"nested image count: {len(nested)}")
print(f"zip size MB: {zip_path.stat().st_size / 1024 / 1024:.2f}")

if len(imgs) != 1000:
    raise RuntimeError(f"Expected 1000 images, got {len(imgs)}")
if nested:
    raise RuntimeError(f"Zip contains nested image paths: {nested[:5]}")
PY

echo "Image count:"
find "$IMGDIR" -maxdepth 1 -type f -name "*.png" | wc -l

echo "Zip:"
ls -lh "$ZIP"

echo "Done: $(date)"
