#!/usr/bin/env python3
import json
import os
import sys
from pathlib import Path
from typing import List, Tuple

import faiss
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

# --- I/O encoding guards -----------------------------------------------------
# Try to force stdin/stdout to UTF-8 and ignore invalid bytes from the terminal
try:
    sys.stdin.reconfigure(encoding="utf-8", errors="ignore")
    sys.stdout.reconfigure(encoding="utf-8", errors="ignore")
except Exception:
    pass

VECTOR_DIR = Path(os.getenv("VECTOR_DIR", "vector_store"))
INDEX_PATH = VECTOR_DIR / "index.faiss"
META_PATH = VECTOR_DIR / "meta.jsonl"
MANIFEST_PATH = VECTOR_DIR / "manifest.json"

# Response language (default Russian as requested)
RAG_RESPONSE_LANG = os.getenv("RAG_RESPONSE_LANG", "ru").lower()

# —— Text hygiene ——

def scrub(text: str) -> str:
    if text is None:
        return ""
    return text.encode("utf-8", errors="ignore").decode("utf-8", errors="ignore")


def safe_input(prompt: str) -> str:
    """Robust input that survives bad terminal encodings."""
    try:
        return input(prompt)
    except UnicodeDecodeError:
        # Fallback: read raw bytes and decode leniently
        sys.stdout.write(prompt)
        sys.stdout.flush()
        data = sys.stdin.buffer.readline()
        return data.decode("utf-8", errors="ignore")

# —— Retrieval helpers ——

def load_index_and_meta():
    if not INDEX_PATH.exists() or not META_PATH.exists():
        raise SystemExit("Vector store not found. Run ingest_md.py first.")
    index = faiss.read_index(str(INDEX_PATH))
    meta = []
    with META_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            meta.append(json.loads(line))
    manifest = {}
    if MANIFEST_PATH.exists():
        try:
            manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        except Exception:
            manifest = {}
    return index, meta, manifest


def search(index, meta, query_vector: np.ndarray, k: int = 5) -> List[Tuple[float, dict]]:
    q = query_vector.astype("float32")[None, :]
    faiss.normalize_L2(q)
    scores, ids = index.search(q, k)
    out = []
    for score, idx in zip(scores[0].tolist(), ids[0].tolist()):
        if idx == -1:
            continue
        out.append((float(score), meta[idx]))
    return out


# —— Embedding selection (symmetry with ingest) ——
try:
    from sentence_transformers import SentenceTransformer
    _has_st = True
except Exception:
    _has_st = False

_embedder_cache = None

def get_query_embedder(manifest: dict):
    """
    Возвращает callable(text)->np.ndarray, совместимый с моделью инжеста.
    Приоритет:
      1) RAG_EMBED_MODEL_OVERRIDE (точное имя HF или 'openai:<model>')
      2) manifest['embedding_model']
      3) дефолты
    """
    embed_id = manifest.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2")
    override_name = os.getenv("RAG_EMBED_MODEL_OVERRIDE", "").strip() or embed_id
    override_mode = os.getenv("RAG_QUERY_EMBEDDER", "").lower().strip()  # 'st' | 'openai'

    def st_embedder_factory(model_name: str):
        def _f(text: str) -> np.ndarray:
            nonlocal model_name
            global _embedder_cache
            if _embedder_cache is None:
                from sentence_transformers import SentenceTransformer
                _embedder_cache = SentenceTransformer(model_name)
            # e5-модели рекомендуют префиксы
            qtext = f"query: {text}" if "e5" in model_name.lower() else text
            vec = _embedder_cache.encode([qtext], convert_to_numpy=True, normalize_embeddings=True)[0]
            return vec.astype("float32")
        return _f

    def openai_embedder_factory(model_name: str):
        def _f(text: str) -> np.ndarray:
            client = OpenAI()
            emb = client.embeddings.create(model=model_name, input=text)
            return np.array(emb.data[0].embedding, dtype="float32")
        return _f

    # Явный оверрайд режимом
    if override_mode == "openai" and os.getenv("OPENAI_API_KEY"):
        model_name = override_name.split(":", 1)[1] if override_name.startswith("openai:") else "text-embedding-3-small"
        return openai_embedder_factory(model_name)
    if override_mode == "st":
        return st_embedder_factory(override_name)

    # Авто-распознавание по имени
    if override_name.startswith("openai:") and os.getenv("OPENAI_API_KEY"):
        return openai_embedder_factory(override_name.split(":", 1)[1])

    # По умолчанию — SentenceTransformers для любого HF id
    return st_embedder_factory(override_name)

# —— LLM call ——

def call_llm(question: str, context_blocks: List[str]) -> str:
    load_dotenv()
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    question = scrub(question)
    context_blocks = [scrub(c) for c in context_blocks]

    if not os.getenv("OPENAI_API_KEY"):
        joined = "\n\n".join(context_blocks) if context_blocks else "(контекст не найден)"
        return f"[DEV MODE / RU]\nВопрос: {question}\n\nКонтекст:\n{joined}\n\nОтвет (заглушка): Установите OPENAI_API_KEY для реальных ответов."

    client = OpenAI()
    system = (
        "Ты — полезный ассистент по документации. Отвечай на РУССКОМ ЯЗЫКЕ. "
        "Опирайся только на переданные CONTEXT-фрагменты; если ответа нет в контексте, так и скажи и предложи, где его искать. "
        "Добавляй краткие ссылки на источники в формате (source: filename.md#chunk)."
    )
    if RAG_RESPONSE_LANG.startswith("ru"):
        user_prefix = "Вопрос"
        ctx_label = "КОНТЕКСТ"
    else:
        user_prefix = "Question"
        ctx_label = "CONTEXT"

    context_text = "\n\n".join(context_blocks)

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": f"{user_prefix}: {question}\n\n{ctx_label}:\n{context_text}"},
    ]

    resp = client.chat.completions.create(model=model, messages=messages, temperature=0.2)
    return resp.choices[0].message.content.strip()



## Modified format_citations to generate wiki links
def format_citations(hits: List[Tuple[float, dict]]) -> List[str]:
    """Формирует список ссылок на wiki.yandex.ru из путей doc_path."""
    cites = []
    cwd = os.getcwd()
    for score, m in hits:
        raw_path = Path(m["doc_path"]).as_posix()

        # 1. Удаляем текущую директорию
        if raw_path.startswith(cwd):
            raw_path = raw_path[len(cwd):]

        # 2. Удаляем префикс /var/out/yandex-wiki-catalog
        raw_path = raw_path.replace("/var/out/yandex-wiki-catalog", "")

        # 3. Удаляем расширение .md
        if raw_path.endswith(".md"):
            raw_path = raw_path[:-3]

        # 4. Удаляем _index, если это финальный элемент пути
        raw_path = raw_path.replace("/_index", "")

        # 5. Удаляем ведущие слэши
        raw_path = raw_path.lstrip("/")

        # 6. Склеиваем базовый URL
        wiki_url = f"https://wiki.yandex.ru/{raw_path}"

        cites.append(f"[{wiki_url}]( {wiki_url} ) (score={score:.3f})")

    # Убираем дубликаты, сохраняя порядок
    seen = set()
    uniq = []
    for c in cites:
        if c not in seen:
            seen.add(c)
            uniq.append(c)
    return uniq

def main():
    load_dotenv()
    index, meta, manifest = load_index_and_meta()
    embedder = get_query_embedder(manifest)

    print("\n🔍 Markdown RAG Agent — введите вопрос (или 'exit')\n")
    while True:
        try:
            q = safe_input("Вы: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nПока!")
            break
        if not q:
            continue
        if q.lower() in {"exit", "quit", ":q", "выход"}:
            print("Пока!")
            break

        q = scrub(q)
        qvec = embedder(q)

        if qvec.shape[0] != index.d:
            print(f"[WARN] Размерность запроса {qvec.shape[0]} != размерности индекса {index.d}. Переиндексируйте или задайте RAG_QUERY_EMBEDDER.")
            continue

        hits = search(index, meta, qvec, k=6)
        blocks = [h[1]["text"] for h in hits]
        citations = format_citations(hits)

        answer = call_llm(q, blocks)
        print("\nАссистент:\n" + answer + "\n")
        if citations:
            print("Источники:")
            for c in citations:
                print(" - ", c)
        print()


if __name__ == "__main__":
    main()
