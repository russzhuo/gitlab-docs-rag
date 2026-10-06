# RAG and LangChain, Explained — Using This Project as the Textbook

> For: engineers who are comfortable with full-stack web development and have called an
> LLM API to build a chatbot, but haven't touched RAG / LangChain.
>
> After reading this you'll understand: embeddings, vector search, pgvector, hybrid search,
> re-ranking, the refusal gate, evaluation metrics, and what LangChain actually does.
> All code is from this repo — open the files alongside this document as you read.

---

## 0. First, Orient Yourself: You Already Know Half of It

If you've built a chatbot, you already know how to do this:

```python
messages = [{"role": "user", "content": "Write me a function"}]
reply = openai.chat(messages)   # call the LLM API, get an answer
```

That's **Generation**. RAG doesn't replace it — it adds **two steps in front of it**:

```
Traditional chatbot:   question ──────────────────────────▶ LLM ──▶ answer
RAG:                   question ──▶ ① retrieve (find related docs) ──▶ ② augment (stuff into prompt) ──▶ LLM ──▶ answer
```

**RAG = Retrieval-Augmented Generation.** The three words in the name are exactly those three steps.

The problem it solves is simple: **an LLM is "frozen" at training time** — it doesn't know your
company's internal policies or the contents of your private docs. Ask it "can I expense a
monitor?" and it will confidently make up an answer. That's called **hallucination**. RAG's
solution is not to retrain the model, but to **look the answer up first, stuff the retrieved
text into the prompt, and tell the model to answer only from that text**.

It's like an exam: instead of making the student (the LLM) memorize the whole handbook, you
let it use an open book — with a rule that it may only answer from the pages it found, and
must say "I don't know" if it can't find a page.

---

## 1. The End-to-End Flow of This Project

Two phases.

**Offline (indexing — runs once, or again when the corpus changes)** — `ingest.py`:

```
markdown files
   │  parse_doc(): parse front-matter → title / section / source
   ▼
chunk with RecursiveCharacterTextSplitter: long docs → ~900-char chunks
   │
   ▼
embed_documents(): each chunk → a fixed-dimension vector
   │
   ▼
write into the `documents` table in Postgres (text and vector stored together)
```

**Online (per question)** — `retrieve()` + `answer()` in `app/services/rag_service.py`:

```
question
   │  embed_query(): the question also becomes a vector
   ▼
retrieve: vector similarity + full-text keyword, RRF-fused, dedup, optional re-rank, take top-k
   │
   ▼
gate: is the best hit's score high enough? if not → refuse outright
   │
   ▼
generate: stuff the top-k chunks into the prompt, LLM answers from them + citations
```

Now let's go concept by concept.

---

## 2. Concept by Concept (Each with Code)

### 2.1 Parsing (`parse_doc`) — Document → Text + Metadata

RAG needs more than raw text. It needs to know **where a passage came from, which section
it belongs to, and its original URL** — citations and retrieval depend on all of it.

From [ingest.py](ingest.py):

```python
def parse_doc(path: Path, corpus_root: Path) -> Document | None:
    raw = path.read_text(encoding="utf-8", errors="replace")
    meta = {
        "path": str(path.relative_to(corpus_root)),
        "title": path.stem.replace("-", " "),
        "section": path.parent.name,     # inferred from folder name, e.g. hr/
        "source": "",
    }
    m = FRONT_MATTER.match(raw)            # parse the --- ... --- block at the top
    if m:
        for line in m.group(1).splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                meta[k.strip()] = v.strip()
        raw = raw[m.end():]
    # ... also parses <!-- source: ... | section: ... --> comments
    text = raw.strip()
    if len(text) < 80:                     # drop files that are too short (e.g. empty)
        return None
    return Document(page_content=text, metadata=meta)
```

`Document` is a small LangChain data structure with just two fields: `page_content` (the text)
and `metadata` (a dict). It's "text + labels" — nothing magical.

### 2.2 Chunking — Why You Can't Stuff the Whole Doc In

Two reasons you must split documents:

1. **Context window limits**: a long doc doesn't fit, and there's no need to send all of it.
2. **Retrieval granularity**: you want to find "the paragraph about PTO", not "the whole HR handbook".

The splitter from [ingest.py](ingest.py):

```python
splitter = RecursiveCharacterTextSplitter(
    chunk_size=900,      # each chunk is at most ~900 chars
    chunk_overlap=150,   # adjacent chunks overlap by 150 chars
    separators=["\n## ", "\n### ", "\n\n", "\n", " "],
)
chunks = splitter.split_documents(docs)
```

Two key parameters:

- **`chunk_size`**: bigger chunks → more context but less precise retrieval (irrelevant text leaks in);
  smaller → more precise but may cut a sentence in half. 900 is a middle ground.
- **`chunk_overlap`**: overlapping chunks avoid a key sentence being split at a boundary and left incomplete.
- **`separators`**: tells the splitter to prefer cutting at `## ` headings first, then `### `, then blank
  lines, and so on — so it cuts at section boundaries instead of mid-sentence.

**Interview note**: chunking is one of the single biggest levers on retrieval quality. Get it wrong and
everything downstream suffers.

### 2.3 Embedding — Turning Text Into Numbers

This is the part that feels opaque, but it's a one-layer concept.

**An embedding is a function: text in, a fixed-length array of numbers (a vector) out.**
This project's embedding model and dimension are config (`EMBED_MODEL` / `EMBED_DIM` in `app/config.py`;
the checked-in `.env` uses `qwen3.7-text-embedding` with 1024 dims — the code's default fallback is
`text-embedding-3-small` with 1536):

```python
embeddings = OpenAIEmbeddings(model=EMBED_MODEL, api_key=..., base_url=...)
vectors = embeddings.embed_documents([c.page_content for c in chunks])
# each chunk becomes [0.012, -0.34, 0.88, ...] — EMBED_DIM numbers
```

Note the embedding client is still `OpenAIEmbeddings`, just pointed at a custom `base_url` — the
"OpenAI-compatible" HTTP protocol is a de-facto standard, so many providers expose it.

The magic is in how the model was trained: **texts with similar meaning get vectors that are
"close" in space.** "monitor", "screen", and "display" cluster together, while "expense policy"
sits far away.

So you don't write any rules to understand meaning — you just measure the distance between two
vectors, and that approximates **semantic similarity** between two texts. That's the essence of
"semantic search".

### 2.4 pgvector — Store Vectors and Compute Similarity in Postgres

As a full-stack engineer, this will feel familiar: **you don't need a separate database for vector
search — you add a Postgres extension called `pgvector`.**

The table is defined in [app/models/index.py](app/models/index.py) as a SQLAlchemy model:

```python
class Document(Base):
    __tablename__ = "documents"
    id: Mapped[int] = mapped_column(primary_key=True)
    path: Mapped[str] = mapped_column(Text, nullable=False)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    section: Mapped[str] = mapped_column(Text, nullable=False)   # used to be "department"
    source: Mapped[str] = mapped_column(Text, nullable=False, default="")
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)   # the raw text
    preview: Mapped[str] = mapped_column(Text, nullable=False, default="")
    embedding: Mapped[list] = mapped_column(Vector(EMBED_DIM), nullable=False)  # its vector, same row
    fts: Mapped[str] = mapped_column(TSVECTOR, Computed("to_tsvector('english', content)", persisted=True))
    created_at: ...
```

`Vector(EMBED_DIM)` is a type provided by `pgvector.sqlalchemy`. **Storing the vector in the same
row as the text** is the key design choice: you can JOIN vector similarity with business filters
using plain SQL/ORM, without running two systems. The extension + indexes are created idempotently
in `init_db()` in [app/database.py](app/database.py) (an HNSW index for vectors, a GIN index for `fts`).

pgvector provides distance operators:

| Operator | Meaning |
|----------|---------|
| `<->` | L2 (Euclidean) distance |
| `<=>` | **cosine distance** (smaller = more similar) |
| `<#>` | inner product (negative = more similar) |

This project uses **cosine similarity**, covered next.

### 2.5 Retrieval — Cosine Similarity + Top-k

The core query lives in `vector_search` in [app/repositories/documents.py](app/repositories/documents.py):

```python
def vector_search(db: Session, qvec: list, n: int):
    dist = Document.embedding.cosine_distance(qvec)
    stmt = select(Document, (1 - dist).label("score")).order_by(dist).limit(n)
    return [(row[0], float(row[1])) for row in db.execute(stmt).all()]
```

Breaking it down:

- `Document.embedding.cosine_distance(qvec)`: cosine **distance** between the question vector and each document vector.
- `1 - dist`: since cosine distance = 1 − cosine similarity, this is **cosine similarity**.
  Closer to 1 = more similar. Aliased as `score`.
- `order_by(dist).limit(n)`: sort by similarity descending, take the top `n` chunks — this is
  **top-k retrieval** (`k` defaults to `TOP_K = 4`).

**Retrieval in one sentence**: turn the question into a vector too, then find the chunks whose
vectors point "most in the same direction".

### 2.6 Hybrid Search — Keyword + Semantic, Covering Each Other's Blind Spots

Pure vector search has a known weakness: **it's great at "different words, same meaning", but can
miss exact terms, identifiers, and proper nouns** — like "on-call" vs "oncall", or a specific policy number.

So the standard practice is two legs:

1. **Vector search**: semantic similarity (above).
2. **Full-text search (BM25-style)**: literal keyword matching — the search-engine approach, built
   into Postgres as `tsvector` + `ts_rank`.

The `text_search` in [app/repositories/documents.py](app/repositories/documents.py):

```python
def text_search(db, qvec, question, n):
    query = func.websearch_to_tsquery("english", question)
    dist = Document.embedding.cosine_distance(qvec)
    stmt = (
        select(Document, (1 - dist).label("score"))
        .where(Document.fts.op("@@")(query))
        .order_by(func.ts_rank(Document.fts, query).desc())
        .limit(n)
    )
    return [(row[0], float(row[1])) for row in db.execute(stmt).all()]
```

- `websearch_to_tsquery` turns the user's natural-language question into search terms.
- `fts @@ query` is the "matches" test; `ts_rank` is the relevance score.

Now you have two ranked lists. How do you merge them? **RRF (Reciprocal Rank Fusion)**:

```python
def rrf_fuse(vector_hits, text_hits, k, k_rank=60):
    rrf = {}
    for rank, h in enumerate(vector_hits, start=1):
        rrf[h["id"]] = rrf.get(h["id"], 0.0) + 1.0 / (k_rank + rank)
    for rank, h in enumerate(text_hits, start=1):
        rrf[h["id"]] = rrf.get(h["id"], 0.0) + 1.0 / (k_rank + rank)
    # sort by RRF score descending, take top k
```

RRF's idea is elegant: **ignore the raw scores (the two lists use different scales and can't be
compared), and use ranks instead.** A document ranked #1 on the vector list and #3 on the keyword
list gets `1/(60+1) + 1/(60+3)`. Documents that rank well on *both* lists rise to the top after
fusion.

After fusion, `retrieve()` does two more things: `_dedup_by_path` (keep at most one chunk per
document path, so one doc can't flood the top-k) and — if `RERANK` is enabled — a cross-encoder
re-rank (see 2.7).

**This is the best resume story in the project**: "hybrid search raised recall from X to Y" comes
from exactly this.

### 2.7 Re-ranking (Optional) — A Cross-encoder Re-scores the Candidates

The bi-encoder embedding used for search is fast but coarse: it compares the question and each
chunk *independently*, so it can miss fine-grained relevance. A **cross-encoder** reads the
`(question, chunk)` pair *together* and scores how well the chunk actually answers the question —
slower, but much more precise.

`retrieve()` applies it only to the already-narrowed candidate pool (never the whole index):

```python
def _rerank(question, hits):
    client = _get_reranker_client()   # OpenAI-compatible client, lazily cached
    response = client.post("/reranks", body={
        "model": RERANK_MODEL, "query": question,
        "documents": [h["content"] for h in hits], "top_n": TOP_K,
    }, cast_to=object)
    results = sorted(response["results"], key=lambda r: float(r["relevance_score"]), reverse=True)
    for r in results:
        hits[r["index"]]["rerank"] = round(float(r["relevance_score"]), 4)
    return [hits[r["index"]] for r in results]
```

Config: `RERANK` (default `false`) turns it on, `RERANK_MODEL` picks the model (this repo uses
`qwen3-rerank` via DashScope), `RERANK_CANDIDATES` is the pool size fed to it. It's a classic
**"retrieve coarse, re-rank fine"** two-stage setup.

### 2.8 The Relevance Gate — Refuse Instead of Hallucinating

This is the project's **most valuable design choice**, and the one beginners most often miss.

The catch: vector search **always returns a top-k**, even when nothing relevant exists in the
database — it returns "the closest few", not "the relevant few". Without a guard, the LLM will
take those irrelevant chunks and invent an answer anyway.

The fix is a threshold (`MIN_RELEVANCE`): if the best hit's score is too low, **refuse outright
without calling the LLM at all**:

```python
# answer() in app/services/rag_service.py
hits, best_signal = retrieve(db, question)
grounded = bool(hits) and best_signal >= _grounding_threshold()   # MIN_RELEVANCE or RERANK_MIN_RELEVANCE

if not grounded:
    return {
        "answer": "I cannot find this in the indexed handbook. I will not guess a company policy.",
        "refused": True,
        ...
    }
```

Note a subtle design point: the gate's signal depends on whether re-ranking is enabled. With
`RERANK` off, `best_signal` is the top **cosine similarity** among candidates (the `score` field is
cosine for vector, text and hybrid alike). With `RERANK` on, `best_signal` is the top **rerank
relevance score**, which separates in-scope from out-of-scope questions better than cosine. Because
those two live on different scales, each has its own threshold — `MIN_RELEVANCE` for cosine and
`RERANK_MIN_RELEVANCE` for the rerank score — and `_grounding_threshold()` picks the right one.

There's also a simpler guard up front: `_is_english()` rejects questions containing non-ASCII
letters (the corpus is English-only), refusing before any embedding work happens.

Why does the gate matter so much for policy docs? Because "I'm not sure" is far safer than "here's
a plausible-sounding but wrong policy."

### 2.9 Generation — Stuffing Retrieved Text Into the Prompt

After retrieval, we're back in the chatbot territory you already know — but the prompt is built
more deliberately.

`SYSTEM` and `build_messages` in [app/services/rag_service.py](app/services/rag_service.py):

```python
SYSTEM = """You are an internal company handbook assistant.
Answer ONLY from the provided handbook excerpts.
Rules:
- If the excerpts are not enough, say you cannot find this in the handbook. Do not guess policy.
- Quote or closely paraphrase the relevant rule.
- Format your answer as Markdown (bullet lists, bold, tables) where it helps readability.
"""

def build_messages(question, context):
    return [
        SystemMessage(content=SYSTEM),
        HumanMessage(
            content="Handbook excerpts:\n\n" + "\n\n".join(context) + f"\n\nQuestion: {question}"
        ),
    ]
```

The prompt that actually reaches the model looks like:

```
[system] You may only answer from the handbook excerpts below; say you can't find it otherwise...
[human]
Handbook excerpts:

[1] title=Time Off section=hr url=...
<retrieved chunk #1, verbatim>

[2] title=Expenses section=finance url=...
<chunk #2>

Question: Can I expense a monitor?
```

The key: **the retrieved text (not a summary) is inserted verbatim**, and the model is explicitly
instructed to answer only from it. That's the "Augmented" part.

On `temperature=0`: generation uses 0 to make output more deterministic and less "creative" —
right for looking things up, wrong for creative writing. The chat model is config via `CHAT_MODEL`;
this repo uses DeepSeek through `langchain-deepseek`:

```python
def get_llm():
    return ChatDeepSeek(model=CHAT_MODEL, api_key=DEEPSEEK_API_KEY, temperature=0)
```

> **A deliberate design note on "multi-turn"**: the RAG call itself is **stateless** — each question
> is answered from scratch, and `build_messages` receives *no* conversation history. The project
> *does* persist conversations and messages (see 2.12) so the UI can show history, but that history
> is **not fed back into the LLM** yet. Feeding prior turns back into the prompt is a known future
> extension.

### 2.10 Citations — Traceable Answers

An answer alone isn't enough — the user must be able to click back to the source and verify. Every
retrieved chunk carries a `source` (the original URL), returned as citations:

```python
def _citations(hits):
    return [
        {"n": i, "title": h["title"], "section": h["section"],
         "source": h["source"], "score": round(h["score"], 3)}
        for i, h in enumerate(hits, start=1)
    ]
```

The frontend renders these as clickable source links under the answer, so the user can read the
answer *and* click through to the original handbook page — this is "no hallucination" made concrete.
(The frontend is out of scope for this doc; the important part is that the backend returns a
structured, traceable citation list.)

### 2.11 Streaming — Token-by-token Answers over SSE

The non-streaming `answer()` returns the whole response at once. For a snappier UI, there's a
`stream_answer()` that yields **SSE events** as the LLM generates:

```python
def stream_answer(db, question):
    hits, best_signal = retrieve(db, question)
    grounded = bool(hits) and best_signal >= _grounding_threshold()
    if not grounded:
        yield RefusedEvent(answer="I cannot find this in the indexed handbook. ...")
        return
    yield MetaEvent(citations=_citations(hits))          # citations up front
    for chunk in get_llm().stream(messages):             # LLM streams tokens
        if chunk.content:
            yield TokenEvent(text=chunk.content)         # one event per token
    yield DoneEvent()
```

The event types (`MetaEvent` / `TokenEvent` / `DoneEvent` / `RefusedEvent` / `ConversationEvent` /
`ErrorEvent`) are defined as Pydantic models in [app/schemas/chat.py](app/schemas/chat.py) — a
single source of truth the frontend generates its TypeScript types from. The router
([app/routers/chat.py](app/routers/chat.py)) serializes each event as an SSE `data: {...}` line.

Two things worth noting:

- The **citations are emitted in a `meta` event before the first token** — so the UI can show sources
  even while the answer is still streaming.
- The **gate happens before any generation**: on refusal, a single `refused` event is sent and no
  LLM call is made at all — same safety as the non-streaming path.

### 2.12 Multi-conversation Persistence — History Without Feeding It Back

The app stores conversations so users can come back to them. In [app/models/history.py](app/models/history.py):

```python
class Conversation(Base):
    __tablename__ = "conversations"
    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"))
    title: Mapped[str] = ...
    updated_at: ...

class Message(Base):
    __tablename__ = "messages"
    conversation_id: Mapped[int] = mapped_column(ForeignKey("conversations.id"))
    role: Mapped[str]      # "user" | "assistant"
    content: Mapped[str]
    citations: Mapped[list] = mapped_column(JSONB, ...)   # the structured citations
    refused: Mapped[bool]
```

After each answer, `history_service.persist_exchange(...)` writes the user question + assistant answer
(+ citations, + the `refused` flag) to Postgres. Endpoints (`GET /conversations`,
`GET /conversations/{id}/messages`) expose them to the frontend.

Again: **persistence ≠ context.** These tables give you history *in the UI*; they do **not** get
stuffed back into the prompt. If you wanted true multi-turn RAG, you'd read `Message` rows and pass
them into `build_messages` — that's the obvious next step this project deliberately leaves open.

### 2.13 Evaluation — How to Know If Retrieval Is Any Good

After building it, how do you know it works? Not by feel — by measurement. This is what separates
a demo from something shippable.

[eval/metrics.py](eval/metrics.py) computes a few metrics (pure functions, no database):

- **recall@k**: did the *correct* chunk appear in the top-k? ("did we find it")
- **MRR**: if found, how high did it rank on average? ("how precise")
- **refusal precision / recall**: did we refuse when we should, answer when we should? The
  **"answered when it should have refused"** case is the most dangerous leak.

Core logic:

```python
def evaluate(items, retrieve_fn, min_relevance):
    for q in items:
        hits, best = retrieve_fn(q["question"])
        grounded = bool(hits) and best >= min_relevance   # did the system answer?
        expect = q["expect_in_docs"]                      # should it have answered?

        # confusion matrix over "should answer vs actually answered"
        if expect and grounded:      tp += 1   # should answer, answered ✓
        elif expect and not grounded: fn += 1  # should answer, refused ✗ (miss)
        elif not expect and grounded: fp += 1  # should NOT answer, answered ✗✗ (hallucination!)
        else:                        tn += 1   # should not answer, refused ✓
```

Each item in the gold set [eval/questions.json](eval/questions.json) carries `expect_in_docs`
(should it answer) and `expected` (words the correct answer must contain):

```json
{ "id": 1, "section": "hr",
  "question": "How does flexible PTO work? ...",
  "expect_in_docs": true,
  "expected": ["pto", "time off", "vacation"] }
```

**What you can say in an interview**: you have a gold set and you can compute recall@k / MRR /
refusal-leak rate, and use them to *prove* whether hybrid search actually helped — that's product
thinking, not "I installed another component".

---

## 3. What LangChain Actually Does

This is the biggest source of confusion for beginners, and this project is deliberately clear-eyed
about it.

**LangChain is not a "framework you must use" — it's a grab-bag of utilities.** You can build this
entire project without it. This project uses it in a few places:

| Used for | Role |
|----------|------|
| `RecursiveCharacterTextSplitter` | chunking (2.2) |
| `OpenAIEmbeddings` | calling the (OpenAI-compatible) embedding model (2.3) |
| `ChatDeepSeek` / message types | calling the chat model, building the prompt (2.9) |

The **most important parts — retrieval, storage, the gate, re-ranking, streaming, evaluation — are
hand-written SQL/ORM and plain Python**, not LangChain.

Why is that good? LangChain's higher-level retrieval abstractions (`RetrievalQA`,
`VectorStoreRetriever`, and friends) hide a lot behind a black box, so beginners end up knowing how
to call the wrapper but not how it works. This project does the opposite: **retrieval is a few lines
of ORM you can fully read.** Once you understand that layer, whether you use LangChain or not is
irrelevant — it's a convenience, not knowledge.

Also worth disambiguating the names:

- **LangChain**: a general-purpose LLM utility library (splitting, prompts, agents, retrievers, …).
- **`langchain-openai`** / **`langchain-deepseek`**: LangChain's adapter packages for specific
  providers (this project uses both).
- **`langchain_core.messages`**: just the `SystemMessage` / `HumanMessage` / `AIMessage` structs —
  "role + content" wrappers, which are still `{role, content}` underneath.

---

## 4. Common Misconceptions (Don't Fall for These in Interviews)

1. **"RAG is just installing a vector database."** ❌ Vector search is only one piece. Chunking, the
   gate, re-ranking, citations, and evaluation matter just as much. This project's strength is
   precisely that it doesn't lean on the vector-DB gimmick — it gets the *refusal* right.

2. **"The embedding is part of the model, call it once and you're done."** ❌ The embedding is a
   **separate model** from the chat model. Change `EMBED_MODEL` and the vector dimensionality
   (`EMBED_DIM`) may change — you have to rebuild the index. Don't hardcode the dimension; it must
   match the model.

3. **"Pick the similarity threshold by gut feel."** ❌ A value like `MIN_RELEVANCE=0.35` must be
   tuned with the eval set (this project's docs repeat "tune by eval, not by feel"). And cosine
   similarity for embedding models tends to sit in a narrow band — don't assume.

4. **"If it retrieved something, it'll answer correctly."** ❌ "Similar" ≠ "correct". That's exactly
   why the gate and evaluation exist.

5. **"For multi-turn chat, save the user's questions and you're done."** ❌ A real multi-turn RAG
   also feeds the assistant's previous answers back into the prompt, so the model knows what it said
   earlier. Note that *this* project persists conversations but does **not** yet feed history back —
   it's stateless per question (see 2.9 / 2.12). That's a simplification, not a feature: if you're
   building true multi-turn, feed the history.

---

## 5. Glossary

| Term | One-liner |
|------|-----------|
| RAG | Retrieval-Augmented Generation: look it up first, then have the LLM answer from what you found |
| Embedding | Map text to a numeric vector; similar meaning → close vectors |
| Vector | A list of floats; here, `EMBED_DIM` numbers |
| Cosine similarity | How "same-direction" two vectors are; 1 = most similar |
| pgvector | Postgres extension for storing vectors and computing distances |
| Top-k | Retrieve "the k most similar" |
| Chunking | Splitting a long document into smaller pieces |
| BM25 / full-text search | Keyword matching with relevance scoring (the search-engine approach) |
| RRF | Reciprocal Rank Fusion: merging two ranked lists by rank |
| Re-ranking | A cross-encoder re-scores the candidate pool for precision |
| Gate / refusal | Refuse when similarity is too low, instead of letting the LLM make it up |
| Hallucination | The model inventing plausible-sounding but false content |
| SSE | Server-Sent Events — the streaming transport for token-by-token answers |
| recall@k / MRR | Retrieval quality metrics: did we find it / how high did it rank |
| Gold set | A hand-labeled evaluation set with expected answers |

---

## 6. To Go Deeper, Read in This Order

1. [docs/architecture.md](architecture.md) — the request path and each module's responsibility.
2. [app/services/rag_service.py](app/services/rag_service.py) — retrieve + generate + gate + re-rank + streaming.
3. [app/repositories/documents.py](app/repositories/documents.py) — the vector/full-text search queries.
4. [app/models/index.py](app/models/index.py) — the `documents` table (schema + pgvector column).
5. [app/database.py](app/database.py) — engine, session, and index creation.
6. [eval/metrics.py](eval/metrics.py) — how the metrics are computed.

**If you only remember three things:**

1. RAG = retrieve + stuff into prompt + generate; you weren't missing the generation, you were missing the first two steps.
2. Embeddings turn "semantic similarity" into "numeric distance", and pgvector lets you compute that distance in SQL.
3. Retrieval always returns "the closest few", so you must add a gate that refuses — that's the line between a demo and something you can ship.
