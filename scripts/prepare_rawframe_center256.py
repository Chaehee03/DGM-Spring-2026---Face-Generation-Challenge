import argparse
from pathlib import Path
from PIL import Image, ImageOps

def center_square_resize(im, size=256):
    im = ImageOps.exif_transpose(im).convert("RGB")
    w, h = im.size
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    im = im.crop((left, top, left + side, top + side))
    im = im.resize((size, size), Image.Resampling.LANCZOS)
    return im

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--shard", type=int, required=True)
    ap.add_argument("--num-shards", type=int, required=True)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--quality", type=int, default=95)
    args = ap.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)

    imgs = sorted([
        p for p in src.iterdir()
        if p.suffix.lower() in [".jpg", ".jpeg", ".png"]
    ])

    imgs = [p for i, p in enumerate(imgs) if i % args.num_shards == args.shard]
    print(f"Shard {args.shard}/{args.num_shards}: {len(imgs)} images")

    for i, p in enumerate(imgs):
        out = dst / f"{p.stem}.jpg"
        if out.exists() and out.stat().st_size > 0:
            continue
        try:
            with Image.open(p) as im:
                im = center_square_resize(im, args.size)
                im.save(out, quality=args.quality, subsampling=0, optimize=True)
        except Exception as e:
            print(f"FAIL {p}: {e}")

        if (i + 1) % 1000 == 0:
            print(f"processed {i+1}/{len(imgs)}")

if __name__ == "__main__":
    main()
