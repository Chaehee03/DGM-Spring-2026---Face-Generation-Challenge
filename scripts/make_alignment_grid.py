from pathlib import Path
from PIL import Image
import argparse
import random

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)
    files = sorted(Path(args.input_dir).glob("*.png"))
    if len(files) == 0:
        raise RuntimeError("No aligned images found")

    sample = random.sample(files, min(args.n, len(files)))

    cell = 256
    cols = 10
    rows = (len(sample) + cols - 1) // cols

    grid = Image.new("RGB", (cols * cell, rows * cell), "white")

    for i, p in enumerate(sample):
        img = Image.open(p).convert("RGB").resize((cell, cell))
        x = (i % cols) * cell
        y = (i // cols) * cell
        grid.paste(img, (x, y))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    grid.save(args.out, quality=95)
    print(args.out)

if __name__ == "__main__":
    main()
