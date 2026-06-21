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

## 快速開始

```bat
:: 1. 啟動虛擬環境並安裝套件
setup\venv\Scripts\activate
pip install -r requirements_backend.txt

:: 2. 設定金鑰（複製範本後填入）
copy .env.example .env

:: 3. （可選）生成食譜資料庫
python scripts\generate_recipes.py

:: 4. 啟動後端
uvicorn app.main:app --reload --port 8000

:: 5. 另開終端機啟動前端
streamlit run frontend.py --server.port 8501
```

開啟 <http://localhost:8501> 即可使用。

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

> **注意**：訓練資料集（`train/`、`valid/`、`test/`）與 `.env` 金鑰未隨庫上傳。
> 資料集可至 [NutriLens Food Ingredients Detection](https://universe.roboflow.com/nutrilens-qvsz6/food-ingredients-detection-nxe34)（CC BY 4.0）下載。

## 授權與資料來源

資料集：NutriLens Food Ingredients Detection（CC BY 4.0）
