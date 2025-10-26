import os
import json
from pathlib import Path
from typing import List, Tuple

import faiss
import numpy as np
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

VECTOR_DIR = Path(os.getenv("VECTOR_DIR", "vector_store"))
INDEX_PATH = VECTOR_DIR / "index.faiss"
META_PATH = VECTOR_DIR / "meta.jsonl"
MANIFEST_PATH = VECTOR_DIR / "manifest.json"

RAG_RESPONSE_LANG = os.getenv("RAG_RESPONSE_LANG", "ru").lower()
RAG_QUERY_TRANSLATE = os.getenv("RAG_QUERY_TRANSLATE", "").lower()  # "en" чтобы переводить запрос

# ---- utils reused ----

def scrub(text: str) -> str:
    if text is None:
        return ""
    return text.encode("utf-8", errors="ignore").decode("utf-8", errors="ignore")

@st.cache_resource(show_spinner=False)
def load_index_and_meta():
    if not INDEX_PATH.exists() or not META_PATH.exists():
        st.stop()
    index = faiss.read_index(str(INDEX_PATH))
    meta = [json.loads(x) for x in META_PATH.read_text(encoding="utf-8").splitlines() if x.strip()]
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

try:
    from sentence_transformers import SentenceTransformer
    _has_st = True
except Exception:
    _has_st = False

_embedder_cache = None

def get_query_embedder(manifest: dict):
    embed_id = manifest.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2")
    override_name = os.getenv("RAG_EMBED_MODEL_OVERRIDE", "").strip() or embed_id
    override_mode = os.getenv("RAG_QUERY_EMBEDDER", "").lower().strip()  # 'st' | 'openai'

    def st_embedder_factory(model_name: str):
        def _f(text: str) -> np.ndarray:
            nonlocal model_name
            global _embedder_cache
            if _embedder_cache is None:
                _embedder_cache = SentenceTransformer(model_name)
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

    if override_mode == "openai" and os.getenv("OPENAI_API_KEY"):
        model_name = override_name.split(":", 1)[1] if override_name.startswith("openai:") else "text-embedding-3-small"
        return openai_embedder_factory(model_name)
    if override_mode == "st":
        return st_embedder_factory(override_name)

    if override_name.startswith("openai:") and os.getenv("OPENAI_API_KEY"):
        return openai_embedder_factory(override_name.split(":", 1)[1])

    return st_embedder_factory(override_name)


def maybe_translate_query(text: str) -> str:
    text = scrub(text)
    if RAG_QUERY_TRANSLATE == "en" and os.getenv("OPENAI_API_KEY"):
        client = OpenAI()
        out = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            messages=[
                {"role": "system", "content": "Translate the user's query into concise English suitable for semantic search. Return only the translation."},
                {"role": "user", "content": text},
            ],
            temperature=0,
        )
        return scrub(out.choices[0].message.content.strip())
    return text


def call_llm(question: str, context_blocks: List[str]) -> str:
    load_dotenv()
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    question = scrub(question)
    context_blocks = [scrub(c) for c in context_blocks]

    if not os.getenv("OPENAI_API_KEY"):
        joined = "\n\n".join(context_blocks) if context_blocks else "(контекст не найден)"
        return f"[DEV MODE / RU]\nВопрос: {question}\n\nКонтекст:\n{joined}\n\nОтвет (заглушка)"

    client = OpenAI()
    system = (
        "Ты — полезный ассистент по документации. Отвечай на РУССКОМ ЯЗЫКЕ. "
        "Опирайся только на CONTEXT; если ответа нет, явно скажи об этом. "
        "Добавляй ссылки на источники в формате (source: filename.md#chunk)."
    )
    context_text = "\n\n".join(context_blocks)

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": f"Вопрос: {question}\n\nКОНТЕКСТ:\n{context_text}"},
    ]

    resp = client.chat.completions.create(model=model, messages=messages, temperature=0.2)
    return resp.choices[0].message.content.strip()


# ---- UI ----
st.set_page_config(page_title="RAG Wiki Assistant", layout="wide")
st.title("🔎 RAG Wiki Assistant")

with st.sidebar:
    st.header("Настройки")
    top_k = 6
    translate = st.checkbox("Переводить запрос RU→EN", value=(RAG_QUERY_TRANSLATE == "en"))
    if translate:
        os.environ["RAG_QUERY_TRANSLATE"] = "en"
    else:
        os.environ["RAG_QUERY_TRANSLATE"] = ""

index, meta, manifest = load_index_and_meta()
embedder = get_query_embedder(manifest)

if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

q = st.chat_input("Введите вопрос по документации…")
if q:
    st.session_state.messages.append({"role": "user", "content": q})
    with st.chat_message("user"):
        st.markdown(q)

    q_for_embed = maybe_translate_query(q)
    qvec = embedder(q_for_embed)
    if qvec.shape[0] != index.d:
        st.error(f"Размерность запроса {qvec.shape[0]} != индекса {index.d}. Переиндексируйте или задайте RAG_EMBED_MODEL_OVERRIDE.")
    else:
        hits = search(index, meta, qvec, k=top_k)
        blocks = [h[1]["text"] for h in hits]
        answer = call_llm(q, blocks)

        # Ссылки: используем вашу format_citations из app.py-версии
        # Чтобы не дублировать код здесь, сформируем ссылки прямо тут
        from pathlib import Path as _Path
        cites = []
        cwd = os.getcwd()
        for score, m in hits:
            raw_path = _Path(m["doc_path"]).as_posix()
            if raw_path.startswith(cwd):
                raw_path = raw_path[len(cwd):]
            raw_path = raw_path.replace("/var/out/yandex-wiki-catalog", "")
            if raw_path.endswith(".md"):
                raw_path = raw_path[:-3]
            raw_path = raw_path.replace("/_index", "").lstrip("/")
            wiki_url = f"https://wiki.yandex.ru/{raw_path}"
            cites.append((wiki_url, score))

        with st.chat_message("assistant"):
            st.markdown(answer)
            if cites:
                st.markdown("**Источники:**")
                for url, sc in cites:
                    st.markdown(f"- [{url}]({url}) (score={sc:.3f})")
        st.session_state.messages.append({"role": "assistant", "content": answer})
