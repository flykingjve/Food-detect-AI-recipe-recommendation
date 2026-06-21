# -*- coding: utf-8 -*-
"""產生「智慧食材辨識系統 使用說明手冊」PDF（繁體中文，含技術架構）

用法：python build_manual.py  ->  產出 使用說明手冊.pdf
需求：reportlab（pip install reportlab）
"""
from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle,
    Preformatted, HRFlowable,
)

BASE = Path(__file__).parent.resolve()
OUT = BASE / "使用說明手冊.pdf"

# 中文字型（reportlab 內建 CID 字型，支援繁中；不含 emoji 故全文不使用 emoji）
CJK = "STSong-Light"
pdfmetrics.registerFont(UnicodeCIDFont(CJK))

NAVY = colors.HexColor("#1F3A5F")
BLUE = colors.HexColor("#2E6FB7")
GREY = colors.HexColor("#555555")
LIGHT = colors.HexColor("#EEF2F7")
CODEBG = colors.HexColor("#F4F4F4")

styles = getSampleStyleSheet()


def S(name, **kw):
    base = kw.pop("parent", styles["Normal"])
    return ParagraphStyle(name, parent=base, fontName=CJK, **kw)


st_title = S("zTitle", fontSize=24, leading=30, textColor=NAVY,
             alignment=TA_CENTER, spaceAfter=6)
st_sub = S("zSub", fontSize=12, leading=18, textColor=GREY,
           alignment=TA_CENTER)
st_h1 = S("zH1", fontSize=16, leading=22, textColor=NAVY, spaceBefore=14,
          spaceAfter=8)
st_h2 = S("zH2", fontSize=12.5, leading=18, textColor=BLUE, spaceBefore=10,
          spaceAfter=4)
st_body = S("zBody", fontSize=10.5, leading=16, spaceAfter=5)
st_bullet = S("zBullet", fontSize=10.5, leading=15, leftIndent=14,
              spaceAfter=2)
st_note = S("zNote", fontSize=9.5, leading=14, textColor=GREY, leftIndent=8)
st_code = ParagraphStyle("zCode", fontName="Courier", fontSize=9, leading=12.5,
                         backColor=CODEBG, borderColor=colors.HexColor("#DDDDDD"),
                         borderWidth=0.5, borderPadding=6, textColor=colors.HexColor("#1A1A1A"),
                         spaceBefore=3, spaceAfter=8)
st_cell = S("zCell", fontSize=9.5, leading=13)
st_cellb = S("zCellB", fontSize=9.5, leading=13, textColor=colors.white)

story = []


def h1(t): story.append(Paragraph(t, st_h1))
def h2(t): story.append(Paragraph(t, st_h2))
def p(t): story.append(Paragraph(t, st_body))
def bullet(t): story.append(Paragraph("• " + t, st_bullet))
def note(t): story.append(Paragraph("※ " + t, st_note))
def code(t): story.append(Preformatted(t, st_code))
def sp(h=6): story.append(Spacer(1, h))


def table(data, widths, header=True):
    rows = []
    for ri, row in enumerate(data):
        style = st_cellb if (header and ri == 0) else st_cell
        rows.append([Paragraph(str(c), style) for c in row])
    t = Table(rows, colWidths=widths, repeatRows=1 if header else 0)
    ts = [
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CCCCCC")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT]),
    ]
    if header:
        ts.append(("BACKGROUND", (0, 0), (-1, 0), NAVY))
    t.setStyle(TableStyle(ts))
    story.append(t)
    sp(8)


# ───────────────────────── 封面 ─────────────────────────
sp(120)
story.append(Paragraph("智慧食材辨識系統", st_title))
story.append(Paragraph("使用說明手冊", st_title))
sp(10)
story.append(Paragraph("YOLO11 食材偵測 × 規則式推薦 × 三層 LLM 容錯鏈", st_sub))
story.append(Paragraph("FastAPI 後端 + Streamlit 前端（含攝影機即時辨識）", st_sub))
sp(40)
story.append(Paragraph("完整操作手冊 + 技術架構", st_sub))
story.append(PageBreak())

# ───────────────────────── 一、系統簡介 ─────────────────────────
h1("一、系統簡介")
p("本系統可辨識照片或攝影機畫面中的食材，並依現有食材推薦合適的家常食譜，再由 AI 生成親切的料理建議。")
p("核心流程：使用者提供食材影像 → YOLO11 偵測食材 → 推薦引擎比對食譜 → LLM 生成推薦說明 → 顯示結果。")
h2("主要功能")
bullet("<b>上傳圖片辨識</b>：上傳食材照片，取得辨識結果與推薦食譜。")
bullet("<b>攝影機即時辨識</b>：開啟鏡頭即時顯示 YOLO Bounding Box，對準後拍照分析。")
bullet("<b>食譜推薦</b>：依食材相符度排序，標示缺少的主食材。")
bullet("<b>AI 料理建議</b>：三層容錯鏈確保永遠有回應。")

# ───────────────────────── 二、系統架構 ─────────────────────────
h1("二、系統架構")
p("系統分為前端（Streamlit）、後端（FastAPI）、模型與資料三大部分，資料流如下：")
code(
"使用者上傳圖片 / 攝影機拍照\n"
"        |\n"
"Streamlit 前端 (port 8501)  --- POST 影像 --->  FastAPI 後端 (port 8000)\n"
"                                                      |\n"
"                                  YOLO11 (weights/best.pt) 偵測食材 conf=0.4\n"
"                                                      |\n"
"                                  推薦引擎: cosine 相似度 + 主料加權 + 缺料懲罰\n"
"                                                      |\n"
"                                  快取檢查 -> 命中直接回傳 / 未命中呼叫 LLM\n"
"                                                      |\n"
"                                  三層 LLM: Groq -> Gemini -> 靜態模板\n"
"                                                      |\n"
"                                  JSON 回傳 -> Streamlit 顯示結果"
)
h2("元件對照")
table([
    ["元件", "技術", "說明"],
    ["前端", "Streamlit", "兩分頁：上傳圖片 / 攝影機即時辨識"],
    ["後端 API", "FastAPI + Uvicorn", "/detect-and-recommend、/health"],
    ["偵測模型", "YOLO11s (Ultralytics)", "13 類食材，mAP@0.5 約 0.905"],
    ["推薦引擎", "scikit-learn cosine", "食材向量相似度 + 加權評分"],
    ["LLM 服務", "Groq / Gemini / 模板", "三層容錯，永不失敗"],
    ["快取", "記憶體 FIFO", "相同食材組合不重複呼叫 LLM"],
], [90, 130, 250])

# ───────────────────────── 三、環境需求與安裝 ─────────────────────────
h1("三、環境需求與安裝")
h2("環境需求")
bullet("作業系統：Windows 11")
bullet("Python：3.11 以上（本機已驗證 3.14.6）")
bullet("GPU（建議）：NVIDIA GPU + CUDA（本機 RTX 3080 10GB）")
bullet("攝影機功能：需本機有可用 webcam")
note("虛擬環境位於 setup\\venv，所有指令請先啟動它。")
h2("安裝步驟")
p("1. 啟動虛擬環境：")
code("setup\\venv\\Scripts\\activate")
p("2. 安裝後端套件（torch / ultralytics 已安裝，僅補裝其餘）：")
code("pip install -r requirements_backend.txt")
p("3. 設定 API 金鑰（.env 已建立；如需自行設定）：")
code("copy .env.example .env\n:: 編輯 .env 填入 GEMINI_API_KEY 與 GROQ_API_KEY")

# ───────────────────────── 四、啟動系統 ─────────────────────────
h1("四、啟動系統")
p("需開兩個終端機，先啟動後端，再啟動前端。")
h2("終端機 A — 後端")
code("uvicorn app.main:app --reload --port 8000")
p("成功會看到 “YOLO11 已載入” 與 “食譜資料庫：N 道”。")
bullet("API 文件：http://localhost:8000/docs")
bullet("健康檢查：http://localhost:8000/health")
h2("終端機 B — 前端")
code("setup\\venv\\Scripts\\activate\nstreamlit run frontend.py --server.port 8501")
p("瀏覽器開啟 http://localhost:8501 即可使用。")
note("手機同網路測試：加上 --server.address 0.0.0.0，手機連 http://<電腦IP>:8501。")

# ───────────────────────── 五、功能操作 ─────────────────────────
h1("五、功能操作")
h2("分頁一：上傳圖片")
bullet("點選「上傳食材圖片」，選擇 jpg / jpeg / png 檔。")
bullet("系統顯示辨識到的食材、AI 料理建議與推薦食譜（依相符度排序）。")
bullet("每道食譜可展開查看主食材、所需時間、難度與缺少的食材。")
h2("分頁二：攝影機即時辨識")
bullet("（可選）設定「攝影機編號」，多顆鏡頭時切換，預設 0。")
bullet("勾選「開啟即時辨識」：畫面即時顯示 YOLO Bounding Box 與當前偵測到的食材名稱。")
bullet("對準食材後按「拍照並分析」：擷取當下畫面送後端，顯示推薦食譜與 AI 建議。")
bullet("想再次即時辨識，重新勾選「開啟即時辨識」即可。")
note("即時畫面只顯示食材；完整推薦與 AI 建議在拍照後才產生。")

# ───────────────────────── 六、食譜資料庫 ─────────────────────────
h1("六、食譜資料庫生成")
p("推薦功能需要食譜資料庫 data/recipes.json。可用 Gemini 批次生成：")
code("python scripts\\generate_recipes.py")
bullet("資料庫為空時系統仍可啟動與偵測，只是推薦清單為空。")
bullet("若出現 429（配額用盡），腳本會存下已成功的部分，待配額重置後再執行續補。")
note("各模型免費額度、RPM、RPD 以官方 Dashboard 即時顯示為準。")

# ───────────────────────── 七、技術架構詳解 ─────────────────────────
h1("七、技術架構詳解")

h2("7.1 YOLO11 食材偵測")
p("使用 Ultralytics YOLO11s，輸入影像、conf=0.4，輸出框與類別。類別 id 對照（與 dataset.yaml 字母順序一致）：")
table([
    ["id", "食材", "id", "食材", "id", "食材"],
    ["0", "Beef 牛肉", "5", "Egg 蛋", "10", "Potato 馬鈴薯"],
    ["1", "Cabbage 高麗菜", "6", "Eggplant 茄子", "11", "Radish 蘿蔔"],
    ["2", "Carrot 紅蘿蔔", "7", "Leek 蒜苗/蔥", "12", "Tomato 番茄"],
    ["3", "Chicken 雞肉", "8", "Onion 洋蔥", "", ""],
    ["4", "Cucumber 小黃瓜", "9", "Pork 豬肉", "", ""],
], [30, 120, 30, 120, 35, 105])

h2("7.2 推薦引擎")
p("將食譜的食材轉成 0/1 向量，計算使用者食材與各食譜的 cosine 相似度，再加權：")
bullet("主料加權：命中主食材每項 +0.15")
bullet("缺料懲罰：每缺一項主食材 -0.10")
bullet("接近獎勵：僅缺 1 項主食材 +0.05")
p("最終分數排序取前 5 名，並標示缺少的主食材與相符度百分比。")

h2("7.3 三層 LLM 容錯鏈")
p("依序嘗試，前一層失敗自動降到下一層，確保永遠有回應：")
table([
    ["層級", "服務 / 模型", "角色"],
    ["L1", "Groq llama-3.1-8b-instant", "主力，速度快"],
    ["L2", "Gemini gemini-3.1-flash-lite", "備援"],
    ["L3", "靜態模板", "永不失敗的保底文字"],
], [60, 240, 170])
p("離線食譜生成另用 Gemini gemini-3.5-flash。")
note("模型配額（RPM / RPD / 免費額度）一律以官方 Dashboard 即時顯示為準。")

h2("7.4 快取機制")
p("以「排序後的食材組合 + 食譜 id」為鍵的記憶體 FIFO 快取（上限 500），相同情境不重複呼叫 LLM，加快回應並節省配額。")

# ───────────────────────── 八、常見問題排除 ─────────────────────────
h1("八、常見問題排除")
table([
    ["問題", "可能原因與解法"],
    ["check_env 顯示無 GPU / torch 為 CPU 版",
     "安裝了 CPU 版 PyTorch。重裝 CUDA 版：pip install torch --index-url https://download.pytorch.org/whl/cu124"],
    ["攝影機「無法開啟」",
     "鏡頭未連接或被其他程式占用；關閉視訊軟體後重試，或調整「攝影機編號」。"],
    ["前端顯示「無法連線後端」",
     "FastAPI 未啟動。先在終端機 A 執行 uvicorn app.main:app --port 8000。"],
    ["AI 建議一直是制式句 / 食譜生成 429",
     "Gemini 金鑰無效或配額用盡。確認金鑰為 AIzaSy... 格式；配額以官方 Dashboard 為準，重置後再試。"],
    ["即時畫面顏色不對（紅藍對調）",
     "顯示需指定 channels=BGR（已於 frontend.py 處理）。"],
    ["食材常漏偵測",
     "將 app/main.py 的 conf=0.4 調低為 0.3。"],
], [150, 320])

# ───────────────────────── 九、專案結構 ─────────────────────────
h1("九、專案結構（重點）")
code(
"ai food detect project\\\n"
"  app\\               後端：main.py / recommendation.py / llm_service.py / cache.py\n"
"  scripts\\           generate_recipes.py（食譜生成）\n"
"  data\\recipes.json  食譜資料庫\n"
"  weights\\best.pt    YOLO11 訓練完成模型\n"
"  frontend.py        Streamlit 前端（上傳 + 攝影機）\n"
"  setup\\venv\\        Python 虛擬環境\n"
"  requirements_backend.txt / .env / README_backend.md"
)

sp(10)
story.append(HRFlowable(width="100%", color=colors.HexColor("#CCCCCC")))
sp(4)
story.append(Paragraph(
    "資料集：NutriLens Food Ingredients Detection（CC BY 4.0）", st_note))


# ───────────────────────── 頁碼 ─────────────────────────
def on_page(canvas, doc):
    canvas.saveState()
    canvas.setFont(CJK, 8)
    canvas.setFillColor(GREY)
    canvas.drawCentredString(A4[0] / 2, 12 * mm,
                             f"智慧食材辨識系統 使用說明手冊　-　{doc.page}")
    canvas.restoreState()


doc = SimpleDocTemplate(
    str(OUT), pagesize=A4,
    leftMargin=20 * mm, rightMargin=20 * mm,
    topMargin=18 * mm, bottomMargin=20 * mm,
    title="智慧食材辨識系統 使用說明手冊",
)
doc.build(story, onFirstPage=on_page, onLaterPages=on_page)
print(f"OK -> {OUT}  ({OUT.stat().st_size} bytes)")
