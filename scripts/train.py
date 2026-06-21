"""
YOLO11 本地訓練腳本
用法：
  python scripts\\train.py                          # 預設 yolo11s，100 epochs
  python scripts\\train.py --model n --epochs 30    # CPU 快速測試
  python scripts\\train.py --model m --batch 8      # 大模型低 batch
"""
from ultralytics import YOLO
from pathlib import Path
import torch, shutil, argparse

BASE_DIR = Path(__file__).parent.parent.resolve()
DATASET_YAML = BASE_DIR / "dataset.yaml"
WEIGHTS_DIR = BASE_DIR / "weights"
PROJECT_DIR = BASE_DIR / "runs"
RUN_NAME = "yolo11_ingredients_v1"


def get_device():
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory // (1024 ** 3)
        print(f"✅ GPU：{name}（VRAM：{vram} GB）")
        return 0
    print("⚠️  未偵測到 GPU，使用 CPU 訓練")
    return "cpu"


def suggest_batch(device, requested_batch):
    """根據 VRAM 建議 batch size"""
    if device == "cpu":
        return min(requested_batch, 4)
    vram = torch.cuda.get_device_properties(0).total_memory // (1024 ** 3)
    if vram >= 16: return min(requested_batch, 32)
    if vram >= 12: return min(requested_batch, 16)
    if vram >= 10: return min(requested_batch, 16)
    if vram >= 8:  return min(requested_batch, 8)
    return min(requested_batch, 4)


def train(model_size="s", epochs=100, batch=16):
    if not DATASET_YAML.exists():
        print("❌ 找不到 dataset.yaml")
        print("👉 請先執行：python scripts\\prepare_yaml.py")
        return

    device = get_device()
    batch = suggest_batch(device, batch)

    if device == "cpu":
        print(f"⚠️  CPU 模式：batch 調整為 {batch}")
        if epochs > 30:
            print(f"⚠️  CPU 模式：建議 epochs 不超過 30（目前設定：{epochs}）")

    model_name = f"yolo11{model_size}.pt"
    print(f"\n🚀 開始訓練 {model_name}")
    print(f"   epochs={epochs} | batch={batch} | device={device}")
    print(f"   資料集：{DATASET_YAML}\n")

    model = YOLO(model_name)
    results = model.train(
        data=str(DATASET_YAML),
        epochs=epochs,
        imgsz=640,
        batch=batch,
        optimizer="auto",
        patience=20,
        device=device,
        project=str(PROJECT_DIR),
        name=RUN_NAME,
        exist_ok=True,
    )

    # 訓練完自動複製 best.pt 到 weights/
    best_src = PROJECT_DIR / RUN_NAME / "weights" / "best.pt"
    best_dst = WEIGHTS_DIR / "best.pt"
    WEIGHTS_DIR.mkdir(exist_ok=True)

    if best_src.exists():
        shutil.copy2(best_src, best_dst)
        print(f"\n✅ 訓練完成！")
        print(f"📁 best.pt → {best_dst}")
        print(f"📊 訓練紀錄 → {PROJECT_DIR / RUN_NAME}")
        print(f"\n👉 下一步：python scripts\\evaluate.py")
    else:
        print(f"⚠️  找不到 best.pt，請確認訓練是否完成")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLO11 本地訓練")
    parser.add_argument("--model", default="s", choices=["n", "s", "m", "l", "x"],
                        help="模型大小：n=最快, s=推薦, m=最準（default: s）")
    parser.add_argument("--epochs", default=100, type=int,
                        help="訓練輪數（CPU 建議 30，GPU 建議 100）")
    parser.add_argument("--batch", default=16, type=int,
                        help="Batch size（會依 VRAM 自動調整上限）")
    args = parser.parse_args()
    train(args.model, args.epochs, args.batch)
