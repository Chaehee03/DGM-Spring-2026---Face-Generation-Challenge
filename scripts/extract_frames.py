import argparse
import cv2
import csv
from pathlib import Path
from multiprocessing import Pool
from tqdm import tqdm


def parse_fracs(s: str):
    vals = [float(x) for x in s.split(",")]
    for v in vals:
        if not (0.0 < v < 1.0):
            raise ValueError(f"Frame fraction must be between 0 and 1: {v}")
    return vals


def extract_one(args_tuple):
    video_path, out_dir, frac_positions, jpeg_quality, overwrite = args_tuple

    rows = []
    cap = cv2.VideoCapture(str(video_path))

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))

    if total <= 0:
        cap.release()
        return rows

    stem = video_path.stem

    for j, frac in enumerate(frac_positions):
        idx = int(total * frac)
        idx = max(0, min(idx, total - 1))

        out_name = f"{stem}_p{int(frac * 100):02d}_f{idx:06d}.jpg"
        out_path = out_dir / out_name

        if out_path.exists() and not overwrite:
            rows.append([str(video_path), stem, frac, idx, fps, str(out_path), "exists"])
            continue

        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()

        if not ret or frame is None:
            rows.append([str(video_path), stem, frac, idx, fps, str(out_path), "read_fail"])
            continue

        ok = cv2.imwrite(
            str(out_path),
            frame,
            [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality],
        )

        status = "ok" if ok else "write_fail"
        rows.append([str(video_path), stem, frac, idx, fps, str(out_path), status])

    cap.release()
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--video-dir",
        default="/data/chsong/dgm-face/videos",
        help="Directory containing extracted CelebV-HQ mp4 files",
    )
    ap.add_argument(
        "--out-dir",
        default="/data/chsong/dgm-face/frames_raw_3f_0205080",
        help="Output directory for raw extracted frames",
    )
    ap.add_argument(
        "--meta-path",
        default="/data/chsong/dgm-face/metadata/extracted_frames_3f_0205080.csv",
        help="CSV metadata path",
    )
    ap.add_argument(
        "--fracs",
        default="0.2,0.5,0.8",
        help="Comma-separated temporal positions within each clip",
    )
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--jpeg-quality", type=int, default=95)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    video_dir = Path(args.video_dir)
    out_dir = Path(args.out_dir)
    meta_path = Path(args.meta_path)
    frac_positions = parse_fracs(args.fracs)

    out_dir.mkdir(parents=True, exist_ok=True)
    meta_path.parent.mkdir(parents=True, exist_ok=True)

    videos = []
    for ext in ["*.mp4", "*.mov", "*.avi", "*.mkv"]:
        videos.extend(video_dir.rglob(ext))
    videos = sorted(videos)

    print(f"Video dir: {video_dir}")
    print(f"Found videos: {len(videos)}")
    print(f"Output dir: {out_dir}")
    print(f"Metadata: {meta_path}")
    print(f"Frame fractions: {frac_positions}")
    print(f"Expected raw frames if all succeed: {len(videos) * len(frac_positions)}")

    tasks = [
        (v, out_dir, frac_positions, args.jpeg_quality, args.overwrite)
        for v in videos
    ]

    all_rows = []
    with Pool(args.workers) as pool:
        for rows in tqdm(pool.imap_unordered(extract_one, tasks), total=len(tasks)):
            all_rows.extend(rows)

    with open(meta_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "video_path",
            "video_stem",
            "frac_position",
            "frame_idx",
            "fps",
            "frame_path",
            "status",
        ])
        writer.writerows(all_rows)

    ok_count = sum(1 for r in all_rows if r[-1] in ("ok", "exists"))
    fail_count = len(all_rows) - ok_count

    print(f"Total attempted frames: {len(all_rows)}")
    print(f"Saved/existing frames: {ok_count}")
    print(f"Failed frames: {fail_count}")
    print(f"Metadata saved: {meta_path}")


if __name__ == "__main__":
    main()
