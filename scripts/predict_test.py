"""
用圖片測試推論
用法：
  python scripts\\predict_test.py                         # 自動從 test 取一張
  python scripts\\predict_test.py --image path\\to\\img.jpg
  python scripts\\predict_test.py --conf 0.3              # 調整信心閾值
"""
from ultralytics import YOLO
from pathlib import Path
import random, argparse

BASE_DIR = Path(__file__).parent.parent.resolve()
BEST_PT = BASE_DIR / "weights" / "best.pt"
TEST_IMAGES = BASE_DIR / "test" / "images"
PREVIEW_DIR = BASE_DIR / "preview"


def main(image_path=None, conf=0.4):
    if not BEST_PT.exists():
        print("❌ 找不到 weights/best.pt")
        print("👉 請先執行：python scripts\\train.py")
        return

    # 若未指定圖片，自動從 test/images 隨機取一張
    if image_path is None:
        imgs = list(TEST_IMAGES.glob("*.jpg")) + list(TEST_IMAGES.glob("*.png"))
        if not imgs:
            print(f"❌ test/images 找不到圖片：{TEST_IMAGES}")
            return
        image_path = random.choice(imgs)
        print(f"🎲 隨機選取測試圖片：{image_path.name}")

    model = YOLO(str(BEST_PT))
    results = model(str(image_path), conf=conf)
    result = results[0]

    # 印出偵測結果
    print(f"\n🔍 推論結果（conf >= {conf}）")
    print(f"   圖片：{Path(image_path).name}")
    if len(result.boxes) == 0:
        print("   未偵測到任何食材")
    else:
        names = model.names
        for box in result.boxes:
            cls_id = int(box.cls)
            conf_v = float(box.conf)
            name = names[cls_id]
            print(f"   ✅ {name:<12} 信心值：{conf_v:.2f}")

    # 儲存結果圖片
    PREVIEW_DIR.mkdir(exist_ok=True)
    out_path = PREVIEW_DIR / "predict_result.jpg"
    result.save(filename=str(out_path))
    print(f"\n📸 結果圖已存至：{out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default=None, help="圖片路徑（不填則自動從 test 隨機選）")
    parser.add_argument("--conf", default=0.4, type=float, help="信心閾值（預設 0.4）")
    args = parser.parse_args()
    main(args.image, args.conf)
