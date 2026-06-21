"""記憶體 FIFO 快取：相同食材組合不重複呼叫 LLM"""

_store: dict[str, str] = {}
MAX = 500


def make_key(detected: list[str], recipe_id: str) -> str:
    return "|".join(sorted(detected)) + "::" + recipe_id


def get(key: str) -> str | None:
    return _store.get(key)


def put(key: str, value: str) -> None:
    if len(_store) >= MAX:
        _store.pop(next(iter(_store)))
    _store[key] = value
