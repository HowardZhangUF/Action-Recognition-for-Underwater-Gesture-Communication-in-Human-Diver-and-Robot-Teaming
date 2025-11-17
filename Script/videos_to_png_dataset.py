# videos_to_png_dataset_nocli.py
# Convert a folder of videos into PNG frames with your dataset layout:
# <DATASET_ROOT>/<CLASS_NAME>/<clip_id>/pose/0.png, 1.png, ...

from pathlib import Path
import cv2
import os

# ========= EDIT THESE =========
VIDEOS_DIR   = Path(r"C:\Users\zhangzihao\Downloads\NONE")
DATASET_ROOT = Path(r"C:\Users\zhangzihao\Action-Recognition-for-Underwater-Gesture-Communication-in-Human-Diver-and-Robot-Teaming\Image_data\horizontal")
CLASS_NAME   = "NONE"   # target class folder name
EVERY_N_FRAMES = 3      # save every Nth frame (ignored if TARGET_FPS > 0)
TARGET_FPS     = 30.0   # sample to this fps if > 0 (uses native_fps / TARGET_FPS)
RESIZE_W, RESIZE_H = 0, 0  # set to 224,224 if you want to resize; keep 0,0 to keep original
MAX_FRAMES_PER_VIDEO = 0   # 0 = no cap
# ==============================

VIDEO_EXTS = (".mp4", ".mov", ".avi", ".mkv", ".webm", ".MP4", ".MOV", ".AVI", ".MKV", ".WEBM")


def list_videos(videos_dir: Path):
    files = []
    for ext in VIDEO_EXTS:
        files.extend(videos_dir.glob(f"*{ext}"))
    return sorted(files)


def get_next_clip_index(class_dir: Path) -> int:
    """Find next numeric subfolder name to use (1..N)."""
    indices = []
    if class_dir.exists():
        for p in class_dir.iterdir():
            if p.is_dir() and p.name.isdigit():
                try:
                    indices.append(int(p.name))
                except ValueError:
                    pass
    return (max(indices) + 1) if indices else 1


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def extract_frames(
    video_path: Path,
    out_pose_dir: Path,
    every_n_frames: int = 1,
    target_fps: float = 0.0,
    resize_w: int = 0,
    resize_h: int = 0,
    max_frames: int = 0,
) -> int:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[WARN] Failed to open: {video_path}")
        return 0

    native_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    step = max(1, int(every_n_frames))
    if target_fps and native_fps > 0:
        step = max(1, round(native_fps / float(target_fps)))

    saved = 0
    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_idx % step == 0:
            if resize_w > 0 and resize_h > 0:
                frame = cv2.resize(frame, (resize_w, resize_h), interpolation=cv2.INTER_AREA)

            out_path = out_pose_dir / f"{saved}.png"
            cv2.imwrite(str(out_path), frame)
            saved += 1

            if max_frames > 0 and saved >= max_frames:
                break

        frame_idx += 1

    cap.release()
    return saved


def main():
    # Basic checks
    if not VIDEOS_DIR.exists():
        raise FileNotFoundError(f"--videos_dir not found: {VIDEOS_DIR}")
    ensure_dir(DATASET_ROOT)

    class_dir = DATASET_ROOT / CLASS_NAME
    ensure_dir(class_dir)

    vids = list_videos(VIDEOS_DIR)
    if not vids:
        raise RuntimeError(f"No video files found in {VIDEOS_DIR}")

    clip_index = get_next_clip_index(class_dir)

    print(f"Found {len(vids)} videos in: {VIDEOS_DIR}")
    print(f"Writing to class folder: {class_dir}")
    print(f"Starting clip index: {clip_index}")
    print(f"Sampling: {'TARGET_FPS=' + str(TARGET_FPS) if TARGET_FPS > 0 else 'EVERY_N_FRAMES=' + str(EVERY_N_FRAMES)}")
    if RESIZE_W > 0 and RESIZE_H > 0:
        print(f"Resize: {RESIZE_W}x{RESIZE_H}")
    if MAX_FRAMES_PER_VIDEO > 0:
        print(f"Cap frames/video: {MAX_FRAMES_PER_VIDEO}")

    total_frames = 0
    for v in vids:
        clip_folder = class_dir / str(clip_index)
        pose_dir = clip_folder / "pose"
        ensure_dir(pose_dir)

        print(f"\n[{clip_index}] {v.name}")
        saved = extract_frames(
            v,
            pose_dir,
            every_n_frames=EVERY_N_FRAMES,
            target_fps=TARGET_FPS,
            resize_w=RESIZE_W,
            resize_h=RESIZE_H,
            max_frames=MAX_FRAMES_PER_VIDEO,
        )
        print(f"  -> saved {saved} frames to {pose_dir}")
        total_frames += saved
        clip_index += 1

    print(f"\nDone. Total frames saved: {total_frames}")
    print(f"Class created/updated: {class_dir}")


if __name__ == "__main__":
    main()
