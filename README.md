# 智慧食材辨識系統 🍳

上傳食材照片或開啟攝影機即時辨識，系統用 **YOLO11** 偵測食材，依現有食材推薦家常食譜，並由 **三層 LLM 容錯鏈** 生成親切的料理建議。

## 功能

- 📁 **上傳圖片辨識** — 上傳食材照片，取得辨識結果與推薦食譜
- 📷 **攝影機即時辨識** — 開啟鏡頭即時顯示 YOLO Bounding Box，對準後拍照分析
- 🍲 **食譜推薦** — cosine 相似度 + 主料加權 + 缺料懲罰，依相符度排序
- 🤖 **AI 料理建議** — Groq → Gemini → 靜態模板，三層容錯永不失敗

## 技術棧

| 層 | 技術 |
|---|---|
| 偵測 | YOLO11s (Ultralytics)，13 類食材，mAP@0.5 ≈ 0.905 |
| 後端 | FastAPI + Uvicorn |
| 前端 | Streamlit（上傳 + 攝影機即時辨識） |
| 推薦 | scikit-learn cosine 相似度 |
| LLM | Groq `llama-3.1-8b-instant` / Gemini `gemini-3.1-flash-lite` / 靜態模板 |

## 快速開始（從 GitHub clone）

> 本庫**已包含訓練好的模型 `weights/best.pt`**，clone 後即可直接執行偵測、推薦與前端，**無需重新訓練、也不需下載資料集**。

```bat
:: 1. 取得專案
git clone https://github.com/flykingjve/Food-detect-AI-recipe-recommendation.git
cd Food-detect-AI-recipe-recommendation

:: 2. 建立並啟動虛擬環境（需 Python 3.11 以上）
python -m venv venv
venv\Scripts\activate

:: 3. 安裝套件
pip install -r requirements_backend.txt
::（可選）GPU 加速：改裝對應 CUDA 版 torch，例如 CUDA 12.4：
:: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

:: 4. 設定金鑰（複製範本後編輯填入自己的金鑰）
copy .env.example .env
:: 編輯 .env 填入 GEMINI_API_KEY 與 GROQ_API_KEY

:: 5. （可選）生成食譜資料庫；庫內已附 data/recipes.json，略過亦可
python scripts\generate_recipes.py

:: 6. 啟動後端
uvicorn app.main:app --reload --port 8000

:: 7. 另開終端機，啟動前端（先 venv\Scripts\activate）
streamlit run frontend.py --server.port 8501
```

開啟 <http://localhost:8501> 即可使用。

> macOS / Linux 的啟動方式為 `source venv/bin/activate`。

詳細說明見 **[使用說明手冊.pdf](使用說明手冊.pdf)**、[README_backend.md](README_backend.md)、[README_training.md](README_training.md)。

## 專案結構

```
app/                FastAPI 後端（main / recommendation / llm_service / cache）
scripts/            訓練、評估、食譜生成等腳本
data/recipes.json   食譜資料庫
weights/best.pt     YOLO11 訓練完成模型
frontend.py         Streamlit 前端
setup/              環境安裝與檢查
```

### 本庫包含 / 未包含

| 已包含（clone 即有） | 未包含（需自行準備） |
|---|---|
| 全部程式碼（app / scripts / frontend） | `.env` 金鑰 → 由 `.env.example` 複製後填入自己的金鑰 |
| 訓練好的模型 `weights/best.pt` | 虛擬環境 `venv/` → 自行 `python -m venv venv` |
| 食譜資料庫 `data/recipes.json` | 訓練資料集 `train/ valid/ test/` → 僅「重新訓練」時才需要 |
| 設定檔 `data.yaml` / `dataset.yaml` | — |

> 因已附 `best.pt` 與 `recipes.json`，一般使用（偵測 + 推薦 + 前端）clone 後安裝套件即可，**不需資料集**。
> 只有要**重新訓練**模型時，才需到 [NutriLens Food Ingredients Detection](https://universe.roboflow.com/nutrilens-qvsz6/food-ingredients-detection-nxe34)（CC BY 4.0）下載資料集放回 `train/ valid/ test/`，詳見 [README_training.md](README_training.md)。

## 授權與資料來源

資料集：NutriLens Food Ingredients Detection（CC BY 4.0）
