"""
讀取 data.yaml，生成帶絕對路徑的 dataset.yaml
Windows 路徑中的反斜線會自動處理為正斜線
"""
import yaml
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent.resolve()
SRC_YAML = BASE_DIR / "data.yaml"
DEST_YAML = BASE_DIR / "dataset.yaml"


def main():
    if not SRC_YAML.exists():
        print(f"❌ 找不到 data.yaml：{SRC_YAML}")
        return

    with open(SRC_YAML, encoding="utf-8") as f:
        src = yaml.safe_load(f)

    # 取得類別資訊
    nc = src.get("nc", 0)
    names = src.get("names", [])

    # 建立帶絕對路徑的 dataset.yaml
    # Windows 路徑轉為正斜線，避免 YOLO 解析錯誤
    dataset = {
        "path": str(BASE_DIR).replace("\\", "/"),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "nc": nc,
        "names": names,
    }

    with open(DEST_YAML, "w", encoding="utf-8") as f:
        yaml.dump(dataset, f, allow_unicode=True, default_flow_style=False, sort_keys=False)

    print(f"✅ dataset.yaml 已生成：{DEST_YAML}")
    print(f"📁 資料集路徑：{BASE_DIR}")
    print(f"🏷️  類別數量：{nc}")
    print(f"🏷️  類別列表：{', '.join(names) if isinstance(names, list) else names}")
    print()
    print("👉 下一步：python setup\\check_env.py")


if __name__ == "__main__":
    main()
