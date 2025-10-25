## How to Run

1. **Install deps** (recommend a fresh venv):

   ```bash
   python -m venv .venv && source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
   pip install -r requirements.txt
   ```

2. **Create your vector DB** from Markdown files:

   ```bash
   python ingest_md.py --docs_dir var/export --out_dir vector_store
   ```

   Common options:

   * `--glob "**/*.md"` to control file discovery.
   * `--chunk_size 900 --chunk_overlap 120` to tune chunking.
   * `--embed_model sentence-transformers/all-MiniLM-L6-v2` to switch models.

3. **Set your OpenAI key** (optional but recommended):

   * Copy `.env.example` → `.env` and fill `OPENAI_API_KEY`.
   * Optionally set `OPENAI_MODEL` (defaults to `gpt-4o-mini`).

4. **Run the agent**:

   ```bash
   python app.py
   ```

   Then ask questions like: *“What does the onboarding doc say about local setup?”*

---