import argparse
import csv
import os
import time
from pathlib import Path

import cv2
from tqdm import tqdm

import onnxruntime as ort
from insightface.app import FaceAnalysis
from insightface.utils import face_align


def log(msg):
    print(msg, flush=True)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", default="/data/chsong/dgm-face/frames_raw_3f_0205080")
    ap.add_argument("--out-dir", default="/data/chsong/dgm-face/faces_256_insight_3f_std")
    ap.add_argument("--meta-path", default="/data/chsong/dgm-face/metadata/aligned_faces_insight_3f_std.csv")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--det-size", default="640,640")
    ap.add_argument("--ctx-id", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--no-sort", action="store_true")
    ap.add_argument("--flush-every", type=int, default=1000)
    ap.add_argument("--shard-id", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    return ap.parse_args()


def iter_images(input_dir: Path, limit: int = 0):
    count = 0
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        for p in input_dir.rglob(ext):
            yield p
            count += 1
            if limit and count >= limit:
                return


def select_largest_face(faces):
    if len(faces) == 0:
        return None

    def area(face):
        x1, y1, x2, y2 = face.bbox
        return max(0.0, (x2 - x1) * (y2 - y1))

    return max(faces, key=area)


def write_rows(meta_path, rows, write_header=False):
    mode = "w" if write_header else "a"
    with open(meta_path, mode, newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow([
                "input_frame",
                "output_face",
                "status",
                "bbox",
                "det_score",
                "num_faces_or_error",
                "shard_id",
                "num_shards",
            ])
        writer.writerows(rows)


def main():
    args = parse_args()
    t0 = time.time()

    input_dir = Path(args.input_dir)
    out_dir = Path(args.out_dir)
    meta_path = Path(args.meta_path)

    out_dir.mkdir(parents=True, exist_ok=True)
    meta_path.parent.mkdir(parents=True, exist_ok=True)

    if args.num_shards < 1:
        raise ValueError("--num-shards must be >= 1")
    if not (0 <= args.shard_id < args.num_shards):
        raise ValueError("--shard-id must satisfy 0 <= shard_id < num_shards")

    det_w, det_h = map(int, args.det_size.split(","))

    log("=== environment ===")
    log(f"python pid: {os.getpid()}")
    log(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")
    log(f"ORT version: {ort.__version__}")
    log(f"ORT providers: {ort.get_available_providers()}")

    if args.ctx_id >= 0 and "CUDAExecutionProvider" not in ort.get_available_providers():
        raise RuntimeError("CUDAExecutionProvider is not available. Refusing to run CPU alignment.")

    log("=== listing images ===")

    # For smoke test with --limit and --no-sort, stop early.
    # For full sharded run, list all images so shard assignment is deterministic.
    if args.limit and args.limit > 0 and args.no_sort and args.num_shards == 1:
        image_paths = list(iter_images(input_dir, limit=args.limit))
    else:
        image_paths = list(iter_images(input_dir, limit=0))
        if not args.no_sort:
            image_paths = sorted(image_paths)
        if args.limit and args.limit > 0:
            image_paths = image_paths[:args.limit]

    before_shard = len(image_paths)

    if args.num_shards > 1:
        image_paths = [
            p for i, p in enumerate(image_paths)
            if i % args.num_shards == args.shard_id
        ]

    log(f"Input dir: {input_dir}")
    log(f"Found frames before shard: {before_shard}")
    log(f"Shard: {args.shard_id}/{args.num_shards}")
    log(f"Found frames for this shard: {len(image_paths)}")
    log(f"Output dir: {out_dir}")
    log(f"Meta path: {meta_path}")
    log(f"Output size: {args.size}")
    log(f"Det size: {(det_w, det_h)}")
    log(f"CTX ID: {args.ctx_id}")
    log(f"Listing time: {time.time() - t0:.1f}s")

    log("=== loading InsightFace ===")
    app = FaceAnalysis(
        name="buffalo_l",
        allowed_modules=["detection"],
        providers=["CUDAExecutionProvider"],
    )

    log("=== preparing InsightFace models ===")
    app.prepare(ctx_id=args.ctx_id, det_size=(det_w, det_h))
    log("=== InsightFace ready ===")

    write_rows(meta_path, [], write_header=True)

    rows_buf = []
    ok = 0
    no_face = 0
    err = 0
    exists = 0

    loop_start = time.time()

    for idx, img_path in enumerate(tqdm(image_paths, dynamic_ncols=True), start=1):
        out_name = img_path.stem + ".png"
        out_path = out_dir / out_name

        if out_path.exists() and not args.overwrite:
            rows_buf.append([str(img_path), str(out_path), "exists", "", "", "", args.shard_id, args.num_shards])
            ok += 1
            exists += 1
        else:
            try:
                img_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)

                if img_bgr is None:
                    rows_buf.append([str(img_path), str(out_path), "read_fail", "", "", "", args.shard_id, args.num_shards])
                    err += 1
                else:
                    faces = app.get(img_bgr)

                    if len(faces) == 0:
                        rows_buf.append([str(img_path), str(out_path), "no_face", "", "", "", args.shard_id, args.num_shards])
                        no_face += 1
                    else:
                        face = select_largest_face(faces)

                        bbox = face.bbox.astype(int).tolist()
                        score = float(face.det_score) if hasattr(face, "det_score") else -1.0

                        aligned = face_align.norm_crop(
                            img_bgr,
                            landmark=face.kps,
                            image_size=args.size,
                        )

                        ok_write = cv2.imwrite(str(out_path), aligned)

                        if not ok_write:
                            rows_buf.append([str(img_path), str(out_path), "write_fail", bbox, score, "", args.shard_id, args.num_shards])
                            err += 1
                        else:
                            rows_buf.append([str(img_path), str(out_path), "ok", bbox, score, len(faces), args.shard_id, args.num_shards])
                            ok += 1

            except Exception as e:
                rows_buf.append([str(img_path), str(out_path), "error", "", "", repr(e), args.shard_id, args.num_shards])
                err += 1

        if idx % args.flush_every == 0:
            write_rows(meta_path, rows_buf, write_header=False)
            rows_buf.clear()

            elapsed = time.time() - loop_start
            rate = idx / max(elapsed, 1e-6)
            log(
                f"[progress] shard={args.shard_id}/{args.num_shards} "
                f"processed={idx}/{len(image_paths)} "
                f"ok_or_exists={ok} exists={exists} no_face={no_face} err={err} "
                f"rate={rate:.2f} img/s"
            )

    if rows_buf:
        write_rows(meta_path, rows_buf, write_header=False)

    log("=== done ===")
    log(f"Shard: {args.shard_id}/{args.num_shards}")
    log(f"Aligned/existing: {ok}")
    log(f"Existing skipped: {exists}")
    log(f"No face: {no_face}")
    log(f"Errors: {err}")
    log(f"Metadata saved: {meta_path}")
    log(f"Total time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
