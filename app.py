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

# Try to force stdin/stdout to UTF-8 to avoid surrogate issues from the terminal
try:
    sys.stdin.reconfigure(encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

VECTOR_DIR = Path(os.getenv("VECTOR_DIR", "vector_store"))
INDEX_PATH = VECTOR_DIR / "index.faiss"
META_PATH = VECTOR_DIR / "meta.jsonl"

# —— Text hygiene ——

def scrub(text: str) -> str:
    """Remove surrogate code points / invalid bytes that can break JSON encoders.
    Keeps everything else as UTF-8. """
    if text is None:
        return ""
    # Round-trip through bytes, dropping invalids
    return text.encode("utf-8", errors="ignore").decode("utf-8", errors="ignore")

# —— Retrieval helpers ——

def load_index_and_meta():
    if not INDEX_PATH.exists() or not META_PATH.exists():
        raise SystemExit("Vector store not found. Run ingest_md.py first.")
    index = faiss.read_index(str(INDEX_PATH))
    meta = []
    with META_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            meta.append(json.loads(line))
    return index, meta


def search(index, meta, query_vector: np.ndarray, k: int = 5) -> List[Tuple[float, dict]]:
    # Make sure query is 2D float32
    q = query_vector.astype("float32")[None, :]
    faiss.normalize_L2(q)
    scores, ids = index.search(q, k)
    out = []
    for score, idx in zip(scores[0].tolist(), ids[0].tolist()):
        if idx == -1:
            continue
        out.append((float(score), meta[idx]))
    return out


# —— Simple embedding with OpenAI (optional) or a local fallback ——
try:
    from sentence_transformers import SentenceTransformer
    _has_st = True
except Exception:
    _has_st = False

_embedder_cache = None

def embed_query(text: str) -> np.ndarray:
    text = scrub(text)
    # Prefer OpenAI embedding if key exists
    if os.getenv("OPENAI_API_KEY"):
        client = OpenAI()
        emb = client.embeddings.create(model="text-embedding-3-small", input=text)
        vec = np.array(emb.data[0].embedding, dtype="float32")
        return vec
    # Fallback to local ST model (same as ingest default)
    if _has_st:
        global _embedder_cache
        if _embedder_cache is None:
            _embedder_cache = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        vec = _embedder_cache.encode([text], convert_to_numpy=True, normalize_embeddings=True)[0]
        return vec.astype("float32")
    raise RuntimeError("No embedding available. Set OPENAI_API_KEY or install sentence-transformers.")


# —— LLM call ——

def call_llm(question: str, context_blocks: List[str]) -> str:
    load_dotenv()
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    question = scrub(question)
    context_blocks = [scrub(c) for c in context_blocks]

    if not os.getenv("OPENAI_API_KEY"):
        # Offline/dev-friendly fallback: a rules-based response
        joined = "\n\n".join(context_blocks) if context_blocks else "(no relevant context)"
        return f"[DEV MODE]\nQuestion: {question}\n\nContext:\n{joined}\n\nAnswer (mock): This is a placeholder answer. Configure OPENAI_API_KEY for real responses."

    client = OpenAI()
    system = (
        "You are a helpful documentation assistant. "
        "Use the provided CONTEXT snippets to answer precisely. "
        "If the answer is not in context, say so and suggest where it might be. "
        "Cite sources as (source: filename.md#chunk)."
    )
    context_text = "\n\n".join(context_blocks)

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": f"Question: {question}\n\nCONTEXT:\n{context_text}"},
    ]

    resp = client.chat.completions.create(model=model, messages=messages, temperature=0.2)
    return resp.choices[0].message.content.strip()


# —— Pretty formatting ——

def format_citations(hits: List[Tuple[float, dict]]) -> List[str]:
    cites = []
    for score, m in hits:
        path = Path(m["doc_path"])  # absolute path saved in ingest
        cites.append(f"{path.name}#chunk{m['chunk_id']} (score={score:.3f})")
    # Deduplicate while preserving order
    seen = set()
    uniq = []
    for c in cites:
        if c not in seen:
            seen.add(c)
            uniq.append(c)
    return uniq


def main():
    load_dotenv()
    index, meta = load_index_and_meta()

    print("\n🔍 Markdown RAG Agent — type your question (or 'exit')\n")
    while True:
        try:
            q = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break
        if not q:
            continue
        if q.lower() in {"exit", "quit", ":q"}:
            print("Bye!")
            break

        q = scrub(q)
        qvec = embed_query(q)
        hits = search(index, meta, qvec, k=6)
        blocks = [h[1]["text"] for h in hits]
        citations = format_citations(hits)

        answer = call_llm(q, blocks)
        print("\nAssistant:\n" + answer + "\n")
        if citations:
            print("Sources:")
            for c in citations:
                print(" - ", c)
        print()


if __name__ == "__main__":
    main()
