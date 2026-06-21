"""三層 LLM 容錯鏈：Groq -> Gemini Flash-Lite -> 靜態模板"""
import os
from groq import Groq
from google import genai
from dotenv import load_dotenv

load_dotenv()

_groq = Groq(api_key=os.environ.get("GROQ_API_KEY", ""))
_gemini = genai.Client(api_key=os.environ.get("GEMINI_API_KEY", ""))

GROQ_MODEL = "llama-3.1-8b-instant"
GEMINI_MODEL = "gemini-3.1-flash-lite"


def _make_prompt(detected: list[str], recipe: dict, missing: list[str]) -> str:
    return (
        f"使用者冰箱有：{'、'.join(detected)}\n"
        f"最推薦食譜：{recipe['name']}\n"
        f"還差食材：{'、'.join(missing) if missing else '食材齊全'}\n"
        f"請用親切繁體中文給一句 80 字內的推薦說明。"
    )


def _try_groq(prompt: str) -> str:
    r = _groq.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=120,
        temperature=0.7,
        timeout=5,
    )
    return r.choices[0].message.content


def _try_gemini(prompt: str) -> str:
    return _gemini.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
    ).text


def _static(recipe: dict, missing: list[str]) -> str:
    if missing:
        return f"推薦您製作{recipe['name']}，只需再補充{'、'.join(missing)}即可！"
    return f"食材齊全！{recipe['name']}是今天的最佳選擇。"


def generate_recommendation_text(
    detected: list[str],
    recipe: dict,
    missing: list[str],
) -> str:
    prompt = _make_prompt(detected, recipe, missing)
    for fn in (_try_groq, _try_gemini):   # L1 -> L2
        try:
            return fn(prompt)
        except Exception:
            continue
    return _static(recipe, missing)       # L3 永不失敗
