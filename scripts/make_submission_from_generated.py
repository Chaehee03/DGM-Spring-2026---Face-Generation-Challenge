import argparse
import csv
import shutil
from pathlib import Path


def collect_images(d, n):
    d = Path(d)
    imgs = sorted((d / "images").glob("*.png"))
    if len(imgs) < n:
        raise RuntimeError(f"Not enough images in {d}: {len(imgs)} < {n}")
    return imgs[:n]


def parse_seed_from_name(name):
    stem = Path(name).stem
    digits = ''.join([c if c.isdigit() else ' ' for c in stem]).split()
    return digits[-1] if digits else ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--psi065", required=True)
    ap.add_argument("--psi080", required=True)
    ap.add_argument("--psi100", required=True)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    out = Path(args.outdir)
    img_out = out / "images"
    img_out.mkdir(parents=True, exist_ok=True)

    selected = []
    selected += [(p, "0.65", Path(args.psi065)) for p in collect_images(args.psi065, 300)]
    selected += [(p, "0.80", Path(args.psi080)) for p in collect_images(args.psi080, 500)]
    selected += [(p, "1.00", Path(args.psi100)) for p in collect_images(args.psi100, 200)]

    meta_rows = []

    for i, (src, psi, src_dir) in enumerate(selected):
        dst_name = f"img_{i:04d}.png"
        dst = img_out / dst_name
        shutil.copy2(src, dst)

        ckpt_file = src_dir / "checkpoint.txt"
        ckpt = ckpt_file.read_text().strip() if ckpt_file.exists() else ""

        meta_rows.append({
            "filename": dst_name,
            "source_file": str(src),
            "seed_guess": parse_seed_from_name(src.name),
            "truncation_psi": psi,
            "checkpoint": ckpt,
        })

    with open(out / "submission_metadata.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=meta_rows[0].keys())
        writer.writeheader()
        writer.writerows(meta_rows)

    print(f"Saved images: {len(selected)}")
    print(f"Output dir: {out}")


if __name__ == "__main__":
    main()
