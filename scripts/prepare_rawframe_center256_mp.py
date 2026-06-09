import argparse
from pathlib import Path
from PIL import Image, ImageOps
from concurrent.futures import ProcessPoolExecutor, as_completed
import os

def center_square_resize(im, size=256):
    im = ImageOps.exif_transpose(im).convert("RGB")
    w, h = im.size
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    im = im.crop((left, top, left + side, top + side))
    im = im.resize((size, size), Image.LANCZOS)
    return im

def process_one(args):
    p, dst, size, quality = args
    out = dst / f"{p.stem}.jpg"
    if out.exists() and out.stat().st_size > 0:
        return "skip", str(p)

    try:
        with Image.open(p) as im:
            im = center_square_resize(im, size)
            im.save(out, quality=quality, subsampling=0, optimize=True)
        return "ok", str(p)
    except Exception as e:
        return "fail", f"{p}: {e}"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--workers", type=int, default=32)
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

    print("src:", src, flush=True)
    print("dst:", dst, flush=True)
    print("input images:", len(imgs), flush=True)
    print("workers:", args.workers, flush=True)
    print("pid:", os.getpid(), flush=True)

    ok = skip = fail = 0
    tasks = [(p, dst, args.size, args.quality) for p in imgs]

    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(process_one, t) for t in tasks]
        for i, fut in enumerate(as_completed(futs), 1):
            status, msg = fut.result()
            if status == "ok":
                ok += 1
            elif status == "skip":
                skip += 1
            else:
                fail += 1
                print("FAIL:", msg, flush=True)

            if i % 1000 == 0:
                print(f"processed={i}/{len(tasks)} ok={ok} skip={skip} fail={fail}", flush=True)

    out_count = len(list(dst.glob("*.jpg")))
    print(f"FINAL ok={ok} skip={skip} fail={fail}", flush=True)
    print("output jpg count:", out_count, flush=True)

    if out_count < 100000:
        raise SystemExit(f"Too few output images: {out_count}")

if __name__ == "__main__":
    main()
