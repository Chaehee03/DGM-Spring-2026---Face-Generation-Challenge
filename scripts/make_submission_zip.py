import argparse
import zipfile
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--output-zip", required=True)
    args = ap.parse_args()

    input_dir = Path(args.input_dir)
    output_zip = Path(args.output_zip)
    output_zip.parent.mkdir(parents=True, exist_ok=True)

    imgs = sorted([p for p in input_dir.glob("*.png")])
    print(f"Found images: {len(imgs)}")

    if len(imgs) != 1000:
        raise RuntimeError(f"Need exactly 1000 png images, but found {len(imgs)}")

    with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for p in imgs:
            z.write(p, arcname=p.name)  # root level

    print(f"Saved zip: {output_zip}")

if __name__ == "__main__":
    main()
