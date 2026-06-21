"""
評估訓練好的 YOLO11 模型
用法：python scripts\\evaluate.py
"""
from ultralytics import YOLO
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent.resolve()
DATASET_YAML = BASE_DIR / "dataset.yaml"
BEST_PT = BASE_DIR / "weights" / "best.pt"


def main():
    if not BEST_PT.exists():
        print("❌ 找不到 weights/best.pt")
        print("👉 請先執行：python scripts\\train.py")
        return

    if not DATASET_YAML.exists():
        print("❌ 找不到 dataset.yaml")
        print("👉 請先執行：python scripts\\prepare_yaml.py")
        return

    print(f"📊 載入模型：{BEST_PT}")
    model = YOLO(str(BEST_PT))
    metrics = model.val(data=str(DATASET_YAML))

    names = list(model.names.values())
    n_classes = len(names)

    # ap50 長度可能 < 類別數（例如某類別在驗證集無樣本），補 0.0 避免 zip 截斷漏印
    ap50 = list(metrics.box.ap50)
    if len(ap50) < n_classes:
        ap50 = ap50 + [0.0] * (n_classes - len(ap50))

    print("\n" + "=" * 50)
    print(" 評估結果（NutriLens 食材偵測）")
    print("=" * 50)
    print(f"mAP@0.5       : {metrics.box.map50:.3f}")
    print(f"mAP@0.5:0.95  : {metrics.box.map:.3f}")
    print(f"Precision     : {metrics.box.mp:.3f}")
    print(f"Recall        : {metrics.box.mr:.3f}")
    print()
    print("各類別 AP@0.5：")
    for name, ap in zip(names, ap50):
        bar = "█" * int(ap * 20)
        status = "✅" if ap >= 0.6 else "⚠️ "
        print(f"  {status} {name:<12}: {ap:.3f}  {bar}")
    print("=" * 50)
    if metrics.box.map50 >= 0.6:
        print("✅ mAP@0.5 達標（>= 0.6），模型可使用")
    else:
        print("⚠️  mAP@0.5 未達 0.6，建議增加 epochs 或補充訓練資料")


if __name__ == "__main__":
    main()
