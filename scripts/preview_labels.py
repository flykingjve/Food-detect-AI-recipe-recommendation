"""
視覺化確認標註：隨機抽取訓練圖片，繪製 Bounding Box，存成 preview/label_preview.jpg
用法：
  python scripts\\preview_labels.py            # 預設 16 張，4x4 grid
  python scripts\\preview_labels.py --n 16
"""
import yaml
import random
import argparse
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # 無視窗環境也能存圖
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import font_manager
from PIL import Image

BASE_DIR = Path(__file__).parent.parent.resolve()
TRAIN_IMAGES = BASE_DIR / "train" / "images"
TRAIN_LABELS = BASE_DIR / "train" / "labels"
DATA_YAML = BASE_DIR / "data.yaml"
PREVIEW_DIR = BASE_DIR / "preview"

# 每個類別一個固定顏色（tab20 取色，依 class id 決定）
CMAP = plt.get_cmap("tab20")


def load_names():
    if DATA_YAML.exists():
        with open(DATA_YAML, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        names = data.get("names", [])
        if isinstance(names, dict):
            names = [names[k] for k in sorted(names)]
        return names
    return []


def color_for(cls_id):
    return CMAP(cls_id % 20)


def read_labels(label_path: Path):
    """讀取 YOLO 格式標註，回傳 [(cls_id, cx, cy, w, h), ...]（皆為正規化值）"""
    boxes = []
    if not label_path.exists():
        return boxes
    with open(label_path, encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 5:
                continue
            cls_id = int(float(parts[0]))
            cx, cy, w, h = (float(p) for p in parts[1:5])
            boxes.append((cls_id, cx, cy, w, h))
    return boxes


def main(n=16):
    if not TRAIN_IMAGES.exists():
        print(f"❌ 找不到 train/images：{TRAIN_IMAGES}")
        return

    imgs = (list(TRAIN_IMAGES.glob("*.jpg"))
            + list(TRAIN_IMAGES.glob("*.jpeg"))
            + list(TRAIN_IMAGES.glob("*.png")))
    if not imgs:
        print(f"❌ train/images 找不到圖片：{TRAIN_IMAGES}")
        return

    names = load_names()
    n = min(n, len(imgs))
    sample = random.sample(imgs, n)

    cols = int(math.ceil(math.sqrt(n)))
    rows = int(math.ceil(n / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    axes = axes.flatten() if n > 1 else [axes]

    for ax in axes:
        ax.axis("off")

    for ax, img_path in zip(axes, sample):
        img = Image.open(img_path).convert("RGB")
        W, H = img.size
        ax.imshow(img)
        ax.set_title(img_path.name, fontsize=8)

        label_path = TRAIN_LABELS / (img_path.stem + ".txt")
        for cls_id, cx, cy, w, h in read_labels(label_path):
            x = (cx - w / 2) * W
            y = (cy - h / 2) * H
            bw, bh = w * W, h * H
            color = color_for(cls_id)
            rect = patches.Rectangle((x, y), bw, bh, linewidth=2,
                                     edgecolor=color, facecolor="none")
            ax.add_patch(rect)
            label = names[cls_id] if 0 <= cls_id < len(names) else str(cls_id)
            ax.text(x, max(y - 4, 0), label, fontsize=8, color="white",
                    bbox=dict(facecolor=color, edgecolor="none", pad=1, alpha=0.8))

    PREVIEW_DIR.mkdir(exist_ok=True)
    out_path = PREVIEW_DIR / "label_preview.jpg"
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)

    print(f"✅ 已抽取 {n} 張圖片繪製標註")
    print(f"📸 預覽圖已存至：{out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="預覽訓練標註")
    parser.add_argument("--n", default=16, type=int, help="抽取張數（預設 16）")
    args = parser.parse_args()
    main(args.n)
