# YOLO11 食材偵測模型 — 本地訓練

使用已就緒的 NutriLens 食材資料集，在本地訓練 YOLO11，產出 `best.pt`。

- **資料集**：13 類別，train 21,544 / valid 1,746 / test 711 張
- **本機 GPU**：NVIDIA RTX 3080（10 GB）→ 訓練 batch 自動上限 16

---

## 前置需求

> ⚠️ **訓練需要資料集，本庫未包含**（檔案龐大未上傳）。
> 重新訓練前，請先到 [NutriLens Food Ingredients Detection](https://universe.roboflow.com/nutrilens-qvsz6/food-ingredients-detection-nxe34)（CC BY 4.0）
> 下載 **YOLOv8 格式**資料集，解壓後將 `train/`、`valid/`、`test/` 與 `data.yaml` 放回專案根目錄。
>
> 另需 **Python 3.11 以上**（安裝時於 Windows 勾選「Add python.exe to PATH」）。
> 只是想**使用**模型（不重新訓練）的話，本庫已附 `weights/best.pt`，請改看 [README_backend.md](README_backend.md)。

---

## 快速開始（重新訓練）

```bat
:: Step 1：安裝環境（只需執行一次）
setup\install.bat

:: Step 2：啟動虛擬環境
venv\Scripts\activate

:: Step 3：生成 dataset.yaml
python scripts\prepare_yaml.py

:: Step 4：確認環境正常
python setup\check_env.py

:: Step 5：預覽確認標註
python scripts\preview_labels.py

:: Step 6：開始訓練
:: 有 GPU（本機 RTX 3080，batch 自動限 16）
python scripts\train.py --model s --epochs 100 --batch 16

:: 僅有 CPU
python scripts\train.py --model n --epochs 30 --batch 4

:: Step 7：評估模型
python scripts\evaluate.py

:: Step 8：測試推論
python scripts\predict_test.py
```

> 已啟動過虛擬環境後，每次只需 `venv\Scripts\activate` 即可直接跑 scripts。

---

## 模型大小選擇

| 模型 | 情境 | 訓練時間（GPU） | 訓練時間（CPU） |
|---|---|---|---|
| yolo11n | CPU 測試、快速驗證 | 30 分鐘 | 2～4 小時 |
| yolo11s | 專題推薦（GPU 8GB+） | 1～2 小時 | 6～10 小時 |
| yolo11m | 精度優先（GPU 12GB+） | 2～4 小時 | 不建議 |

> 本機 RTX 3080（10 GB）建議用 `yolo11s`，batch 會自動限制為 16。
> 21,544 張訓練圖跑 100 epoch 約需數小時，請預留時間。

---

## 資料集類別與 LABEL_MAP 對應

`dataset.yaml` 的 `names` 為**字母順序**，以下 LABEL_MAP 已對應本資料集實際 class id，
訓練完成後可直接複製到後端 `app/main.py`：

```python
# class id 以 dataset.yaml 的 names 順序為準（本資料集為字母順序）
LABEL_MAP = {
    0: "Beef",
    1: "Cabbage",
    2: "Carrot",
    3: "Chicken",
    4: "Cucumber",
    5: "Egg",
    6: "Eggplant",
    7: "Leek",
    8: "Onion",
    9: "Pork",
    10: "Potato",
    11: "Radish",
    12: "Tomato",
}
```

> ⚠️ **重要：** 若日後更換資料集或版本，class id 仍以 `dataset.yaml` 的 `names` 欄位為準。
> 執行 `python scripts\prepare_yaml.py` 後請打開 `dataset.yaml` 確認順序，再填入 `app/main.py`。

---

## 專案結構

```
ai food detect project\
├── train/ valid/ test/        ← 資料集（未隨庫上傳，需從 Roboflow 下載放回）
├── data.yaml                  ← 原始設定（隨資料集下載）
├── dataset.yaml               ← prepare_yaml.py 生成（絕對路徑版）
├── requirements_training.txt
├── README_training.md
├── setup\
│   ├── install.bat            # 建立 venv + 安裝套件
│   └── check_env.py           # 環境與 GPU 檢查
├── scripts\
│   ├── prepare_yaml.py        # 生成 dataset.yaml
│   ├── preview_labels.py      # 標註視覺化預覽
│   ├── train.py               # 訓練主程式
│   ├── evaluate.py            # 評估 mAP / per-class AP
│   └── predict_test.py        # 推論測試
├── weights\                   # best.pt 產出位置
├── preview\                   # 預覽 / 推論結果圖
└── runs\                      # 訓練紀錄（自動生成）
```

---

*資料集：NutriLens Food Ingredients Detection（CC BY 4.0）*
*<https://universe.roboflow.com/nutrilens-qvsz6/food-ingredients-detection-nxe34>*
