"""Chunk handbook files and write embeddings into Postgres + pgvector."""

from __future__ import annotations

import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sqlalchemy import func, select, text
from tqdm import tqdm

from app.config import CORPUS_DIR, EMBED_API_KEY, EMBED_BASE_URL, EMBED_MODEL
from app.database import SessionLocal, init_db
from app.models.index import Document as DocumentRow

FRONT_MATTER = re.compile(r"^---\n(.*?)\n---\n", re.S)
SOURCE_COMMENT = re.compile(r"<!--\s*source:\s*(\S+)\s*\|\s*section:\s*(\S+)\s*-->")

# The Qwen/DashScope embedding API rejects batches larger than 20, so 20 is the
# ceiling — speed comes from running several requests in parallel (EMBED_WORKERS).
EMBED_BATCH = 20
EMBED_WORKERS = 8
MAX_RETRIES = 5          # per-batch retries with exponential backoff (429 / 5xx / connection)
BASE_DELAY = 2.0         # seconds; doubles each retry

CHUNK_SIZE = 1200
CHUNK_OVERLAP = 100

def parse_doc(path: Path, corpus_root: Path) -> Document | None:
    raw = path.read_text(encoding="utf-8", errors="replace")
    meta = {
        "path": path.relative_to(corpus_root).as_posix(),
        "title": path.stem.replace("-", " "),
        "section": path.parent.name,
        "source": "",
    }
    m = FRONT_MATTER.match(raw)
    if m:
        for line in m.group(1).splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                meta[k.strip()] = v.strip()
        raw = raw[m.end():]
    c = SOURCE_COMMENT.search(raw)
    if c:
        meta["source"] = c.group(1)
        meta["section"] = c.group(2)
        raw = SOURCE_COMMENT.sub("", raw, count=1)
    text = raw.strip()
    if len(text) < 80:
        return None
    return Document(page_content=text, metadata=meta)


def load_corpus(corpus_dir: Path) -> list[Document]:
    docs: list[Document] = []
    for path in sorted(corpus_dir.rglob("*.md")):
        if path.name in {"README.md"}:
            continue
        doc = parse_doc(path, corpus_dir)
        if doc:
            docs.append(doc)
    return docs


def _is_retryable(err: Exception) -> bool:
    code = getattr(err, "status_code", None)
    return code is None or code == 429 or code >= 500


def _embed_batch(embeddings, texts: list[str]) -> list[list[float]]:
    """Embed one batch, retrying transient failures (429 / 5xx / connection)."""
    vecs = None
    for attempt in range(MAX_RETRIES):
        try:
            vecs = embeddings.embed_documents(texts)
            break
        except Exception as err:
            if not _is_retryable(err) or attempt == MAX_RETRIES - 1:
                raise
            time.sleep(BASE_DELAY * (2 ** attempt))
    if len(vecs) != len(texts):
        raise RuntimeError(f"embedding returned {len(vecs)} vectors for {len(texts)} texts")
    return vecs


def embed_with_progress(make_embeddings, texts: list[str], batch: int = EMBED_BATCH,
                        workers: int = EMBED_WORKERS) -> list[list[float]]:
    """Embed `texts` in parallel batches, one client per thread, with a progress bar."""
    batches = [texts[i:i + batch] for i in range(0, len(texts), batch)]
    ordered: list = [None] * len(batches)
    local = threading.local()

    def get_client():
        if not hasattr(local, "embeddings"):
            local.embeddings = make_embeddings()
        return local.embeddings

    def _embed(pair):
        idx, chunk = pair
        return idx, _embed_batch(get_client(), chunk)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(_embed, (i, b)): i for i, b in enumerate(batches)}
        with tqdm(total=len(texts), desc="embedding", unit="chunk") as pbar:
            for fut in as_completed(futures):
                idx, vecs = fut.result()
                ordered[idx] = vecs
                pbar.update(len(vecs))
    return [v for batch in ordered for v in batch]


def main() -> None:
    started = time.time()
    corpus = Path(CORPUS_DIR)
    if not corpus.exists():
        raise SystemExit(f"Corpus not found: {corpus}")

    init_db()
    docs = load_corpus(corpus)
    print(f"Loaded {len(docs)} files from {corpus}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n## ", "\n### ", "\n\n", "\n", " "],
    )
    chunks = splitter.split_documents(docs)
    print(f"Split into {len(chunks)} chunks")

    def make_embeddings():
        return OpenAIEmbeddings(
            model=EMBED_MODEL,
            api_key=EMBED_API_KEY,
            base_url=EMBED_BASE_URL,
            tiktoken_enabled=False,
            check_embedding_ctx_length=False,
            chunk_size=EMBED_BATCH,
        )

    vectors = embed_with_progress(make_embeddings, [c.page_content for c in chunks])
    print(f"Embedded {len(vectors)} chunks in {time.time() - started:.1f}s")

    with SessionLocal() as db:
        db.execute(text("TRUNCATE documents"))
        for i, (chunk, vec) in tqdm(enumerate(zip(chunks, vectors)), total=len(chunks),
                                   desc="inserting", unit="chunk"):
            meta = chunk.metadata
            db.add(DocumentRow(
                path=meta.get("path", ""),
                title=meta.get("title", ""),
                section=meta.get("section", ""),
                source=meta.get("source", ""),
                chunk_index=i,
                content=chunk.page_content,
                preview=chunk.page_content[:180].replace("\n", " "),
                embedding=vec,
            ))
            if (i + 1) % 5000 == 0:
                db.flush()
        db.commit()
        count = db.scalar(select(func.count(DocumentRow.id)))
    print(f"Wrote {count} rows in {time.time() - started:.1f}s total")


if __name__ == "__main__":
    main()
