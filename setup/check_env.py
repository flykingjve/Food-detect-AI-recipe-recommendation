"""
YOLO11 環境檢查
用法：python setup\\check_env.py

逐項驗證訓練所需環境，印出 ✅ / ❌ / ⚠️，最後統計通過項數。
"""
import sys
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent.resolve()


def count_images(folder: Path) -> int:
    if not folder.exists():
        return 0
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")
    total = 0
    for ext in exts:
        total += len(list(folder.glob(ext)))
    return total


def main():
    print("================================================")
    print(" YOLO11 環境檢查")
    print("================================================")

    passed = 0
    total = 10
    gpu_available = False

    # 1. Python 版本
    py = sys.version_info
    if py.major == 3 and py.minor >= 9:
        print(f"✅ Python {py.major}.{py.minor}.{py.micro}")
        passed += 1
    else:
        print(f"❌ Python {py.major}.{py.minor}.{py.micro}（建議 3.11）")

    # 2. ultralytics
    try:
        import ultralytics
        print(f"✅ ultralytics {ultralytics.__version__}")
        passed += 1
    except Exception:
        print("❌ ultralytics 未安裝（pip install -r requirements_training.txt）")

    # 3. torch
    torch = None
    try:
        import torch
        print(f"✅ torch {torch.__version__}")
        passed += 1
    except Exception:
        print("❌ torch 未安裝")

    # 4. GPU
    if torch is not None and torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory // (1024 ** 3)
        print(f"✅ GPU 可用：{name}（VRAM: {vram} GB）")
        gpu_available = True
        passed += 1
    else:
        print("❌ GPU 不可用（將使用 CPU）")

    # 5-7. train / valid / test images
    for split, label in (("train", "train/images"),
                         ("valid", "valid/images"),
                         ("test", "test/images")):
        folder = BASE_DIR / split / "images"
        n = count_images(folder)
        if folder.exists() and n > 0:
            print(f"✅ {label} 資料夾存在（{n} 張圖片）")
            passed += 1
        else:
            print(f"❌ {label} 資料夾不存在或無圖片")

    # 8. data.yaml
    if (BASE_DIR / "data.yaml").exists():
        print("✅ data.yaml 存在")
        passed += 1
    else:
        print("❌ data.yaml 不存在")

    # 9. dataset.yaml
    dataset_yaml = BASE_DIR / "dataset.yaml"
    if dataset_yaml.exists():
        print("✅ dataset.yaml 存在（已生成絕對路徑版）")
        passed += 1
    else:
        print("❌ dataset.yaml 不存在")

    # 10. weights/
    if (BASE_DIR / "weights").exists():
        print("✅ weights/ 資料夾存在")
        passed += 1
    else:
        print("❌ weights/ 資料夾不存在")

    print()
    print(f"通過：{passed}/{total}")
    print("================================================")

    # 額外提示
    if not gpu_available:
        print("⚠️  未偵測到 GPU")
        print("   CPU 訓練 100 epochs 預估需要 6～12 小時")
        print("   建議改用：python scripts\\train.py --model n --epochs 30 --batch 4")

    if not dataset_yaml.exists():
        print("❌ 找不到 dataset.yaml")
        print("   請先執行：python scripts\\prepare_yaml.py")


if __name__ == "__main__":
    main()
