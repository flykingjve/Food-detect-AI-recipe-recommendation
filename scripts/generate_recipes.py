"""食譜批次生成：Gemini Flash + 指數退避重試"""
from google import genai
from google.genai import types
from pathlib import Path
from dotenv import load_dotenv
import json, time, os

load_dotenv()
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
MODEL = "gemini-3.5-flash"

BASE_DIR = Path(__file__).parent.parent.resolve()

# 13 類食材（對應 LABEL_MAP）
INGREDIENTS = [
    "Beef", "Cabbage", "Carrot", "Chicken", "Cucumber",
    "Egg", "Eggplant", "Leek", "Onion", "Pork",
    "Potato", "Radish", "Tomato",
    # 常用調味料（不需偵測但可入食譜）
    "garlic", "ginger", "soy sauce", "sesame oil", "salt", "pepper",
]

PROMPT_TEMPLATE = """Generate {count} distinct home-style recipes using ingredients from this list: {ings}

Rules:
- name: traditional Chinese dish name in Traditional Chinese (繁體中文)
- description: one sentence in Traditional Chinese
- main_ingredients and total_ingredients: use EXACT ingredient names from the list above
- difficulty: one of "easy" / "medium" / "hard"
- time_minutes: integer"""

# 用 response_schema 強制輸出結構，確保回傳為合法 JSON 陣列（recipe_id 由程式另行編號）
RECIPE_SCHEMA = types.Schema(
    type=types.Type.ARRAY,
    items=types.Schema(
        type=types.Type.OBJECT,
        properties={
            "name": types.Schema(type=types.Type.STRING),
            "main_ingredients": types.Schema(
                type=types.Type.ARRAY, items=types.Schema(type=types.Type.STRING)),
            "sub_ingredients": types.Schema(
                type=types.Type.ARRAY, items=types.Schema(type=types.Type.STRING)),
            "total_ingredients": types.Schema(
                type=types.Type.ARRAY, items=types.Schema(type=types.Type.STRING)),
            "difficulty": types.Schema(type=types.Type.STRING),
            "time_minutes": types.Schema(type=types.Type.INTEGER),
            "description": types.Schema(type=types.Type.STRING),
        },
        required=["name", "main_ingredients", "sub_ingredients",
                  "total_ingredients", "difficulty", "time_minutes", "description"],
    ),
)


def generate_batch(n: int = 10, retries: int = 3) -> list[dict]:
    prompt = PROMPT_TEMPLATE.format(
        count=n,
        ings=", ".join(INGREDIENTS),
    )
    for attempt in range(retries):
        try:
            resp = client.models.generate_content(
                model=MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    max_output_tokens=8192,
                    temperature=0.7,
                    response_mime_type="application/json",
                    response_schema=RECIPE_SCHEMA,
                ),
            )
            # 偵測因 token 上限被截斷的情況，給出清楚訊息
            cand = resp.candidates[0] if resp.candidates else None
            if cand is not None and str(getattr(cand, "finish_reason", "")).endswith("MAX_TOKENS"):
                raise ValueError("輸出被 max_output_tokens 截斷，請降低 batch 數量")
            return json.loads(resp.text)
        except Exception as e:
            wait = 5 * (2 ** attempt)   # 5s / 10s / 20s
            print(f"⚠️  重試 {attempt+1}/{retries}（{e}），等待 {wait}s")
            time.sleep(wait)
    raise RuntimeError("批次生成連續失敗")


def main(total: int = 200, batch: int = 10) -> None:
    recipes: list[dict] = []
    batches = total // batch
    failed = 0

    for i in range(batches):
        print(f"📦 第 {i+1}/{batches} 批...")
        try:
            data = generate_batch(batch)
        except RuntimeError as e:
            failed += 1
            print(f"⚠️  第 {i+1} 批跳過（{e}）")
            continue
        for j, r in enumerate(data):
            r["recipe_id"] = f"R{len(recipes)+j+1:03d}"
        recipes.extend(data)
        time.sleep(4)   # 避免 RPM 超限

    out = BASE_DIR / "data" / "recipes.json"
    out.parent.mkdir(exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(recipes, f, ensure_ascii=False, indent=2)
    print(f"✅ 完成！共 {len(recipes)} 道食譜 → {out}")
    if failed:
        print(f"⚠️  有 {failed} 批失敗，已存入成功的部分；可再次執行補足數量")


if __name__ == "__main__":
    main()
