"""規則式推薦引擎：cosine similarity + 主料加權 + 缺料懲罰"""
import json
import numpy as np
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity


class RecipeRecommender:
    def __init__(self, path: str | Path):
        with open(path, encoding="utf-8") as f:
            self.recipes = json.load(f)

        # 建立食材詞彙表
        all_ing = set()
        for r in self.recipes:
            all_ing.update(r.get("total_ingredients", []))
        self.vocab = sorted(all_ing)
        self.idx = {v: i for i, v in enumerate(self.vocab)}

        # 預先建立所有食譜向量
        self.vectors = np.array([
            self._vec(r.get("total_ingredients", []))
            for r in self.recipes
        ])

    def _vec(self, ings: list[str]) -> np.ndarray:
        v = np.zeros(len(self.vocab))
        for i in ings:
            if i in self.idx:
                v[self.idx[i]] = 1.0
        return v

    def recommend(self, detected: list[str], top_k: int = 5) -> list[dict]:
        if not self.recipes or not self.vocab:
            return []

        uv = self._vec(detected).reshape(1, -1)
        scores = cosine_similarity(uv, self.vectors)[0]

        results = []
        for i, r in enumerate(self.recipes):
            main_ings = r.get("main_ingredients", [])
            missing = [x for x in main_ings if x not in detected]
            main_bonus = sum(0.15 for x in main_ings if x in detected)
            miss_pen = len(missing) * 0.10
            near_bonus = 0.05 if len(missing) == 1 else 0.0
            final = scores[i] + main_bonus - miss_pen + near_bonus

            results.append({
                **r,
                "score": round(float(final), 3),
                "missing": missing,
                "match_pct": round(float(scores[i]) * 100, 1),
            })

        return sorted(results, key=lambda x: x["score"], reverse=True)[:top_k]
