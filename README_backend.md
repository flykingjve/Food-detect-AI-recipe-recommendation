# 智慧食材辨識系統 — 後端

FastAPI + YOLO11 + 規則式推薦引擎 + 三層 LLM 容錯鏈 + Streamlit 前端。

> **環境需求**：Python 3.11 以上（已在 3.14.6 驗證）。以下指令請在**專案根目錄**執行。
> 本庫已附訓練好的 `weights/best.pt`，clone 後安裝套件即可執行，不需重新訓練或下載資料集。

---

## 1. 建立環境並安裝套件

```bat
:: 建立並啟動虛擬環境（首次）
python -m venv venv
venv\Scripts\activate

:: 安裝後端套件
pip install -r requirements_backend.txt
```

> 之後每次使用只需 `venv\Scripts\activate`。
> requirements 會裝 fastapi、uvicorn、streamlit、scikit-learn、groq、google-genai、ultralytics 等；
> 預設安裝的是 CPU 版 torch。若要 GPU 加速，另裝對應 CUDA 版（例 CUDA 12.4）：
> `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124`

## 2. 設定 API Key

`.env` 未隨庫上傳，請從範本複製後填入**自己的**金鑰：

```bat
copy .env.example .env
:: 用記事本開啟 .env，填入 Gemini 和 Groq 的 API Key
```

申請（皆免費）：
- Gemini：<https://aistudio.google.com> → Get API Key（金鑰格式為 `AIzaSy...`）
- Groq：<https://console.groq.com> → API Keys（金鑰格式為 `gsk_...`）

> ⚠️ 若 AI 料理建議一直走靜態模板、或 `generate_recipes.py` 失敗，多半是 Gemini 金鑰無效，請重新申請 `AIzaSy...` 格式金鑰。

### 使用模型

| 用途 | 模型 |
|---|---|
| 離線食譜生成（`generate_recipes.py`） | `gemini-3.5-flash` |
| 線上 L1 主力（`llm_service.py`） | Groq `llama-3.1-8b-instant` |
| 線上 L2 備援（`llm_service.py`） | `gemini-3.1-flash-lite` |
| 線上 L3（永不失敗） | 靜態模板 |

> **配額說明**：各模型的免費額度、RPM、RPD **以官方 Dashboard 即時顯示為準**，本文件不寫死固定數字。
> - Gemini：<https://ai.dev/rate-limit> ／ AI Studio Dashboard
> - Groq：<https://console.groq.com> Dashboard

## 3. 生成食譜資料庫（一次性，需有效 Gemini 金鑰）

```bat
python scripts\generate_recipes.py
```

> 未執行時 `data\recipes.json` 為空陣列，系統仍可啟動與偵測，只是推薦清單為空。
> 若生成中途出現 `429 RESOURCE_EXHAUSTED`，表示當日該模型免費配額已用盡（以官方 Dashboard 為準）；腳本會存下已成功的部分，待配額重置後再執行即可續補。

## 4. 啟動後端

```bat
uvicorn app.main:app --reload --port 8000
```

啟動成功會看到：
```
✅ YOLO11 已載入：...\weights\best.pt
✅ 食譜資料庫：N 道
```

- API 文件：<http://localhost:8000/docs>
- 健康檢查：<http://localhost:8000/health>

## 5. 啟動前端（另開一個終端機）

```bat
venv\Scripts\activate
streamlit run frontend.py --server.port 8501
```

開啟瀏覽器：<http://localhost:8501>

> 手機同網路測試：改用 `streamlit run frontend.py --server.address 0.0.0.0 --server.port 8501`，手機連 `http://<電腦IP>:8501`。

### 前端兩種模式

| 分頁 | 用途 |
|---|---|
| 📁 上傳圖片 | 上傳食材照片 → 後端辨識 + 推薦食譜 + AI 建議 |
| 📷 攝影機即時辨識 | 開啟鏡頭即時顯示 YOLO Bounding Box → 拍照 → 完整 AI 分析 |

**攝影機分頁操作：**
1. 勾選「▶️ 開啟即時辨識」→ 畫面即時顯示框與食材名稱
2. 對準食材後按「📸 拍照並分析」→ 顯示擷取畫面、推薦食譜、AI 建議
3. 想再次即時辨識，重新勾選 checkbox 即可

> - 需本機有可用 webcam；若顯示「無法開啟攝影機」，請關閉其他占用鏡頭的程式（視訊軟體等）後重試。
> - 多顆鏡頭可用「攝影機編號」欄位切換（預設 0）。
> - 攝影機即時辨識需 Streamlit 與鏡頭在同一台電腦（本機 localhost 成立；遠端連線無法存取本機鏡頭）。

---

## LABEL_MAP（與 dataset.yaml 一致，不可修改）

```python
LABEL_MAP = {
    0: "Beef",      1: "Cabbage",   2: "Carrot",
    3: "Chicken",   4: "Cucumber",  5: "Egg",
    6: "Eggplant",  7: "Leek",      8: "Onion",
    9: "Pork",      10: "Potato",   11: "Radish",
    12: "Tomato",
}
```

---

## 系統流程

```
使用者上傳圖片
       ↓
Streamlit（8501）POST 圖片 → FastAPI（8000）
       ↓
YOLO11（weights/best.pt）偵測 conf=0.4
       ↓
RecipeRecommender → cosine + 加權評分 → Top-5
       ↓
快取檢查 → 命中回傳 / 未命中呼叫 LLM
       ↓
三層 LLM：Groq → Gemini → 靜態模板（永不失敗）
       ↓
JSON 回傳 Streamlit 顯示
```

## 常見調整

| 項目 | 做法 |
|---|---|
| 食材常漏偵 | 把 `app/main.py` 的 `conf=0.4` 改為 `conf=0.3` |
| LLM 說明改英文 | 修改 `app/llm_service.py` 的 `_make_prompt` 提示詞 |
| 手機測試 | Streamlit 加 `--server.address 0.0.0.0` |
