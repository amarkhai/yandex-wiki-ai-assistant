#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

DEFAULT_EMBED_MODEL = "intfloat/multilingual-e5-base"
# Recommended for RU/Multilingual:
#   - "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2" (384-dim, lightweight (EN-centric))
#   - "intfloat/multilingual-e5-base" (768-dim) — strong, but not lightweight

def read_markdown_files(docs_dir: Path, glob_pattern: str):
    paths = sorted(docs_dir.glob(glob_pattern))
    for p in paths:
        if p.is_file() and p.suffix.lower() == ".md":
            try:
                text = p.read_text(encoding="utf-8", errors="ignore")
                yield p, text
            except Exception as e:
                print(f"[WARN] Could not read {p}: {e}")


def simple_md_clean(text: str) -> str:
    return "\n".join(line.rstrip() for line in text.strip().splitlines())


def chunk_text(text: str, chunk_size: int = 900, chunk_overlap: int = 120):
    text = text.strip()
    if not text:
        return []
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        chunk = text[start:end]
        chunks.append(chunk)
        if end == n:
            break
        start = max(0, end - chunk_overlap)
    return chunks


def build_index(vectors: np.ndarray):
    d = vectors.shape[1]
    index = faiss.IndexFlatIP(d)
    faiss.normalize_L2(vectors)
    index.add(vectors)
    return index


def main():
    ap = argparse.ArgumentParser(description="Ingest Markdown files into a FAISS vector store")
    ap.add_argument("--docs_dir", type=Path, required=True, help="Directory containing .md files")
    ap.add_argument("--out_dir", type=Path, default=Path("vector_store"), help="Output dir for FAISS + metadata")
    ap.add_argument("--glob", dest="glob_pattern", default="**/*.md", help="Glob to find Markdown files")
    ap.add_argument("--chunk_size", type=int, default=900, help="Chunk size in characters")
    ap.add_argument("--chunk_overlap", type=int, default=120, help="Overlap in characters between chunks")
    ap.add_argument("--embed_model", default=DEFAULT_EMBED_MODEL, help="Sentence-Transformers model identifier")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Loading embed model: {args.embed_model}")
    embedder = SentenceTransformer(args.embed_model)

    metadata = []
    all_chunks = []

    print("[INFO] Scanning markdown files…")
    for path, raw in tqdm(list(read_markdown_files(args.docs_dir, args.glob_pattern))):
        cleaned = simple_md_clean(raw)
        chunks = chunk_text(cleaned, chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)
        for i, ch in enumerate(chunks):
            meta = {
                "id": len(metadata),
                "doc_path": str(path.resolve()),
                "chunk_id": i,
                "text": ch,
            }
            metadata.append(meta)
            all_chunks.append(ch)

    if not all_chunks:
        raise SystemExit("No markdown chunks found. Check your --docs_dir and --glob pattern.")

    print(f"[INFO] Embedding {len(all_chunks)} chunks…")
    embeddings = embedder.encode(all_chunks, batch_size=64, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)

    print("[INFO] Building FAISS index…")
    index = build_index(embeddings.copy())

    index_path = args.out_dir / "index.faiss"
    faiss.write_index(index, str(index_path))

    meta_path = args.out_dir / "meta.jsonl"
    with meta_path.open("w", encoding="utf-8") as f:
        for m in metadata:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")

    manifest = {
        "embedding_model": args.embed_model,  # read by app.py to match query embedder
        "num_vectors": len(all_chunks),
        "chunk_size": args.chunk_size,
        "chunk_overlap": args.chunk_overlap,
        "glob": args.glob_pattern,
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"[OK] Wrote {index_path} and {meta_path}. Vectors: {len(all_chunks)}. Dim: {embeddings.shape[1]}")


if __name__ == "__main__":
    main()
