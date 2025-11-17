import os
import argparse
import csv
from collections import deque, Counter

import cv2
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from torchvision.models import ResNet50_Weights


# ----------------------------
# Colors (cycled if classes > 11)
# ----------------------------
BASE_COLORS = [
    (245, 117, 16),
    (117, 245, 16),
    (16, 117, 245),
    (255, 0, 0),
    (0, 255, 255),
    (255, 0, 255),
    (128, 0, 128),
    (128, 128, 0),
    (0, 128, 128),
    (50, 100, 50),
    (100, 50, 150),
]

def make_colors(n):
    if n <= len(BASE_COLORS):
        return BASE_COLORS[:n]
    # cycle
    colors = []
    for i in range(n):
        colors.append(BASE_COLORS[i % len(BASE_COLORS)])
    return colors


# --------------------------------------
# 1) LABELS (match ImageFolder ordering)
# --------------------------------------
def get_class_names_from_folder(data_path: str):
    names = []
    for d in os.listdir(data_path):
        full = os.path.join(data_path, d)
        if os.path.isdir(full) and not d.startswith('.'):
            names.append(d)
    names = sorted(names)
    if not names:
        raise ValueError(f"No class folders found under: {data_path}")
    return names


# --------------------------------------
# 2) MODEL
# --------------------------------------
def build_resnet(num_classes, device, weights_path):
    # Same architecture as training
    model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def make_transform(img_size=224):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


# --------------------------------------
# 3) UI: probability bar panel (like your example)
# --------------------------------------
def prob_viz(probs, actions, input_frame, colors, bar_max=300):
    """
    Draw horizontal bars for each class, like your demo.
    bar_max controls the max bar width in pixels (100 in your snippet).
    """
    output_frame = input_frame.copy()
    h, w = output_frame.shape[:2]

    y0 = 60
    row_h = 40
    x0 = 10
    # optional background strip
    panel_h = y0 + row_h * len(actions) + 20
    panel_w = min(bar_max + 220, w - 20)
    cv2.rectangle(output_frame, (0, y0 - 30), (panel_w, panel_h), (0, 0, 0), -1)
    cv2.addWeighted(output_frame[y0 - 30:panel_h, 0:panel_w], 0.6,
                    input_frame[y0 - 30:panel_h, 0:panel_w], 0.4, 0,
                    output_frame[y0 - 30:panel_h, 0:panel_w])

    limit = min(len(probs), len(actions), len(colors))
    for i in range(limit):
        action = actions[i]
        c = colors[i]
        prob = float(probs[i])
        bar_w = int(bar_max * prob)

        y1 = y0 + i * row_h
        y2 = y1 + 30
        cv2.rectangle(output_frame, (x0, y1), (x0 + bar_w, y2), c, -1)

        txt = f"{action} {prob*100:.1f}%"
        cv2.putText(output_frame, txt, (x0 + 5, y1 + 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255),
                    2, cv2.LINE_AA)
    return output_frame


def draw_sentence_strip(frame_bgr, sentence_text):
    out = frame_bgr.copy()
    strip_h = 40
    cv2.rectangle(out, (0, 0), (out.shape[1], strip_h), (245, 117, 16), -1)
    cv2.putText(out, sentence_text, (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255),
                2, cv2.LINE_AA)
    return out


# --------------------------------------
# 4) MAIN
# --------------------------------------
def main():
    parser = argparse.ArgumentParser("ResNet50 video inference with bar overlay")
    parser.add_argument("--video_in", type=str, required=True, help="Input video path")
    parser.add_argument("--video_out", type=str, default="annotated.mp4", help="Output annotated video path")
    parser.add_argument("--weights", type=str, default="resnet50_diver_action_model.pth", help="Trained weights .pth")
    parser.add_argument("--data_path", type=str, required=True, help="Root used for training (class folders)")
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--threshold", type=float, default=0.6, help="Min confidence to add to sentence")
    parser.add_argument("--sentence_max", type=int, default=5, help="Max tokens to keep in sentence strip")
    parser.add_argument("--ema_alpha", type=float, default=0.2, help="EMA smoothing [0 disables]")
    parser.add_argument("--vote_window", type=int, default=0, help="Majority vote window (0 disables)")
    parser.add_argument("--bar_max", type=int, default=300, help="Max width of probability bars (px)")
    parser.add_argument("--display", action="store_true", help="Show live window (press q to quit)")
    parser.add_argument("--save_csv", type=str, default="video_predictions.csv",
                        help="CSV path to save per-frame predictions")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    # Labels & colors
    class_names = get_class_names_from_folder(args.data_path)
    num_classes = len(class_names)
    colors = make_colors(num_classes)

    # Model
    device = torch.device(args.device)
    model = build_resnet(num_classes, device, args.weights)
    transform = make_transform(args.img_size)

    # Video I/O
    cap = cv2.VideoCapture(args.video_in)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {args.video_in}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # or 'avc1' if needed on your platform
    writer = cv2.VideoWriter(args.video_out, fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open VideoWriter for: {args.video_out}")

    # Smoothing
    ema_probs = None
    alpha = float(args.ema_alpha)
    use_ema = alpha > 0.0
    votes = deque(maxlen=max(1, args.vote_window)) if args.vote_window > 0 else None

    # Sentence (stable label memory)
    sentence = []

    # CSV rows
    csv_rows = []
    frame_idx = 0

    with torch.inference_mode():
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break

            # Preprocess
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(frame_rgb)
            x = transform(pil).unsqueeze(0).to(device)

            # Forward
            logits = model(x)  # [1, C]
            probs = F.softmax(logits, dim=1).squeeze(0).detach().cpu().numpy()

            # EMA
            use_probs = probs
            if use_ema:
                if ema_probs is None:
                    ema_probs = probs.copy()
                else:
                    ema_probs = alpha * probs + (1.0 - alpha) * ema_probs
                use_probs = ema_probs

            # Majority vote (on top-1 labels)
            top1_idx_now = int(np.argmax(use_probs))
            if votes is not None:
                votes.append(top1_idx_now)
                top1_idx = Counter(votes).most_common(1)[0][0]
            else:
                top1_idx = top1_idx_now

            top1_label = class_names[top1_idx]
            top1_conf = float(use_probs[top1_idx])

            # Draw probability bars (like your sample)
            overlay = prob_viz(use_probs, class_names, frame_bgr, colors, bar_max=args.bar_max)

            # Sentence logic (like your sample)
            if top1_conf > args.threshold:
                if not sentence or (top1_label != sentence[-1]):
                    sentence.append(top1_label)
            if len(sentence) > args.sentence_max:
                sentence = sentence[-args.sentence_max:]

            # Draw sentence strip
            sentence_text = " | ".join(sentence)
            overlay = draw_sentence_strip(overlay, sentence_text)

            writer.write(overlay)

            # CSV log
            ts = frame_idx / fps
            top3_idx = np.argsort(use_probs)[-3:][::-1]
            csv_rows.append({
                "frame_idx": frame_idx,
                "timestamp_sec": f"{ts:.3f}",
                "top1_label": top1_label,
                "top1_index": top1_idx,
                "top1_confidence": f"{top1_conf:.6f}",
                "top3_labels": "|".join([class_names[i] for i in top3_idx]),
                "top3_confidences": "|".join([f"{float(use_probs[i]):.6f}" for i in top3_idx]),
            })

            if args.display:
                try:
                    cv2.imshow("ResNet Action Recognition", overlay)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                except cv2.error:
                    pass

            frame_idx += 1

    cap.release()
    writer.release()
    try:
        cv2.destroyAllWindows()
    except cv2.error:
        pass

    # Save CSV
    if args.save_csv:
        fieldnames = ["frame_idx", "timestamp_sec", "top1_label", "top1_index",
                      "top1_confidence", "top3_labels", "top3_confidences"]
        with open(args.save_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(csv_rows)

    # Quick summary
    if csv_rows:
        counts = Counter([r["top1_label"] for r in csv_rows])
        maj = counts.most_common(1)[0][0]
        print(f"Done. Frames: {len(csv_rows)} | Majority label: {maj}")
    print(f"Annotated video: {args.video_out}")
    if args.save_csv:
        print(f"Per-frame CSV: {args.save_csv}")


if __name__ == "__main__":
    main()
