"""FastAPI 主程式 — lifespan 啟動模式"""
from contextlib import asynccontextmanager
from pathlib import Path
import io

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
from PIL import Image

from app.recommendation import RecipeRecommender
from app.llm_service import generate_recommendation_text
from app import cache

# ── 路徑設定（從本檔案位置往上推算專案根目錄）──────────────────
BASE_DIR = Path(__file__).parent.parent.resolve()
WEIGHTS_PT = BASE_DIR / "weights" / "best.pt"
RECIPES_JSON = BASE_DIR / "data" / "recipes.json"

# ── LABEL_MAP（與 dataset.yaml names 字母順序一致）──────────────
LABEL_MAP = {
    0: "Beef",      1: "Cabbage",   2: "Carrot",
    3: "Chicken",   4: "Cucumber",  5: "Egg",
    6: "Eggplant",  7: "Leek",      8: "Onion",
    9: "Pork",      10: "Potato",   11: "Radish",
    12: "Tomato",
}

state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """啟動時載入模型與食譜；關閉時釋放資源"""
    if not WEIGHTS_PT.exists():
        raise FileNotFoundError(f"找不到 best.pt：{WEIGHTS_PT}")
    state["detector"] = YOLO(str(WEIGHTS_PT))
    state["recommender"] = RecipeRecommender(RECIPES_JSON)
    print(f"✅ YOLO11 已載入：{WEIGHTS_PT}")
    print(f"✅ 食譜資料庫：{len(state['recommender'].recipes)} 道")
    yield
    state.clear()


app = FastAPI(title="智慧食材辨識系統", version="1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/detect-and-recommend")
async def detect_and_recommend(file: UploadFile = File(...)):
    # 讀取圖片
    try:
        img = Image.open(io.BytesIO(await file.read())).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="無法讀取圖片")

    # YOLO11 偵測
    results = state["detector"](img, conf=0.4)
    detected = list({
        LABEL_MAP[int(c)]
        for c in results[0].boxes.cls.tolist()
        if int(c) in LABEL_MAP
    })

    # 推薦引擎
    recs = state["recommender"].recommend(detected, top_k=5)

    # LLM 容錯鏈（若食譜庫為空則跳過）
    llm_summary = ""
    if recs:
        top = recs[0]
        key = cache.make_key(detected, top.get("recipe_id", top["name"]))
        llm_summary = cache.get(key)
        if llm_summary is None:
            llm_summary = generate_recommendation_text(detected, top, top["missing"])
            cache.put(key, llm_summary)

    return {
        "detected_ingredients": detected,
        "recommendations": recs,
        "llm_summary": llm_summary,
    }


@app.get("/health")
def health():
    rec = state.get("recommender")
    return {
        "status": "ok",
        "model": str(WEIGHTS_PT.name),
        "recipes_count": len(rec.recipes) if rec else 0,
    }
