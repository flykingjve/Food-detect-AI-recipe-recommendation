"""Streamlit 前端：上傳圖片 / 攝影機即時辨識 + 拍照分析"""
import time
from pathlib import Path

import cv2
import requests
import streamlit as st
from PIL import Image

st.set_page_config(page_title="智慧食材辨識系統", layout="centered")
st.title("🍳 智慧食材辨識與食譜推薦")
st.caption("上傳食材照片或開啟攝影機即時辨識，立即獲得料理建議")

BACKEND_URL = "http://localhost:8000"
WEIGHTS_PT = Path(__file__).parent / "weights" / "best.pt"


@st.cache_resource
def load_model():
    """前端本地載入 YOLO 供即時預覽（GPU 自動）"""
    from ultralytics import YOLO
    return YOLO(str(WEIGHTS_PT))


def analyze_bytes(jpg: bytes):
    """把 jpg bytes 上傳後端做完整分析，回傳 dict；失敗回 None"""
    try:
        resp = requests.post(
            f"{BACKEND_URL}/detect-and-recommend",
            files={"file": ("img.jpg", jpg, "image/jpeg")},
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.ConnectionError:
        st.error("❌ 無法連線後端，請確認 FastAPI 已啟動（uvicorn app.main:app --port 8000）")
        return None
    except Exception as e:
        st.error(f"❌ 發生錯誤：{e}")
        return None


def render_results(data):
    """顯示辨識食材 / AI 建議 / 推薦食譜"""
    # 偵測結果
    st.subheader("🥕 辨識到的食材")
    if data["detected_ingredients"]:
        st.success("、".join(data["detected_ingredients"]))
    else:
        st.warning("未偵測到食材（請確認圖片清晰、食材明顯）")

    # LLM 建議
    if data.get("llm_summary"):
        st.subheader("🤖 AI 料理建議")
        st.info(data["llm_summary"])

    # 推薦食譜
    st.subheader("📋 推薦食譜（依相符度排序）")
    if not data["recommendations"]:
        st.warning("目前食譜資料庫為空，請先執行 python scripts\\generate_recipes.py")
    else:
        for i, r in enumerate(data["recommendations"], 1):
            with st.expander(f"{i}. {r['name']}（相符度 {r['match_pct']}%）"):
                c1, c2 = st.columns(2)
                c1.write(f"**主食材：** {'、'.join(r.get('main_ingredients', []))}")
                c1.write(f"**時間：** {r.get('time_minutes', '—')} 分鐘")
                c1.write(f"**難度：** {r.get('difficulty', '—')}")
                if r["missing"]:
                    c2.warning(f"缺少：{'、'.join(r['missing'])}")
                else:
                    c2.success("食材完整，可以直接做！")
                if r.get("description"):
                    st.caption(r["description"])


tab_upload, tab_cam = st.tabs(["📁 上傳圖片", "📷 攝影機即時辨識"])

# ── 分頁一：上傳圖片 ───────────────────────────────────────────
with tab_upload:
    uploaded = st.file_uploader("上傳食材圖片", type=["jpg", "jpeg", "png"])
    if uploaded:
        st.image(Image.open(uploaded), use_container_width=True)
        with st.spinner("辨識中..."):
            data = analyze_bytes(uploaded.getvalue())
        if data:
            render_results(data)

# ── 分頁二：攝影機即時辨識 + 拍照分析 ──────────────────────────
with tab_cam:
    st.write("勾選「開啟即時辨識」後，畫面會即時顯示 YOLO Bounding Box；對準食材後按「拍照並分析」。")
    cam_idx = st.number_input("攝影機編號（多顆鏡頭時可調）", min_value=0, max_value=8, value=0, step=1)
    run = st.checkbox("▶️ 開啟即時辨識")
    snap = st.button("📸 拍照並分析")

    if snap and st.session_state.get("last_jpg"):
        # 拍照優先：顯示擷取畫面並送後端分析（畫面凍結在結果）
        st.image(st.session_state["last_jpg"], caption="擷取畫面", use_container_width=True)
        with st.spinner("分析中..."):
            data = analyze_bytes(st.session_state["last_jpg"])
        if data:
            render_results(data)

    elif run:
        model = load_model()
        # placeholder 必須在 while 迴圈外宣告，否則每幀疊加
        frame_ph = st.empty()
        info_ph = st.empty()

        cap = cv2.VideoCapture(int(cam_idx), cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap.release()
            st.error(f"❌ 無法開啟攝影機（編號 {int(cam_idx)}），請確認鏡頭已連接且未被其他程式占用。")
        else:
            try:
                while cap.isOpened() and run:
                    ret, frame = cap.read()
                    if not ret:
                        st.warning("⚠️ 讀取攝影機畫面失敗")
                        break

                    results = model(frame, conf=0.4, verbose=False)
                    annotated = results[0].plot()  # BGR numpy array（含框與標籤）
                    names = sorted({
                        model.names[int(c)]
                        for c in results[0].boxes.cls.tolist()
                    })

                    # 存原始幀（未疊框）供後端重新分析
                    success, buf = cv2.imencode(".jpg", frame)
                    if success:
                        st.session_state["last_jpg"] = buf.tobytes()

                    frame_ph.image(annotated, channels="BGR", use_container_width=True)
                    info_ph.success("偵測中：" + ("、".join(names) if names else "（目前無食材）"))

                    time.sleep(0.01)  # 讓 Streamlit event loop 有機會處理 UI 事件
            finally:
                cap.release()

    else:
        st.info("勾選「開啟即時辨識」啟動攝影機；對準食材後按「拍照並分析」。")
