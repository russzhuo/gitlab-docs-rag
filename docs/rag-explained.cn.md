# RAG 与 LangChain 入门 —— 用这个项目当教材

> 写给:熟悉 Web 全栈、调过 LLM API 做过 chatbot,但没碰过 RAG / LangChain 的工程师。
>
> 读完你会理解:embedding、向量检索、pgvector、混合检索、重排序、拒答 gate、评测指标,以及 LangChain 到底在干嘛。
> 所有代码都来自本仓库,你可以在编辑器里跟着点开看。

---

## 0. 先定位:你其实已经会了一半

你做过 chatbot,说明你已经会了这件事:

```python
messages = [{"role": "user", "content": "帮我写个函数"}]
reply = openai.chat(messages)   # 调大模型 API,拿回答
```

这就是 **生成(Generation)**。RAG 没有推翻它,而是**在它前面加了两步**:

```
传统 chatbot:     问题 ──────────────────────────▶ LLM ──▶ 回答
RAG:              问题 ──▶ ①检索(找相关资料) ──▶ ②增强(塞进 prompt) ──▶ LLM ──▶ 回答
```

**RAG = Retrieval-Augmented Generation(检索增强生成)**。名字里的三个词就是这三步。

要解决的问题很朴素:**大模型训练完就"冻结"了,不知道你公司的内部政策、不知道你私有文档里的内容。** 问它「我能不能报销显示器」,它会一本正经地编一个答案——这叫**幻觉(hallucination)**。RAG 的解法不是重新训练模型,而是**先查资料、把查到的原文塞进 prompt、让模型照着原文回答**。

这就像考试:与其让考生(LLM)背下整本手册,不如允许他开卷,而且规定「只能照着翻到的页回答,翻不到就写不知道」。

---

## 1. 这个项目的端到端流程

先看全局,再看细节。这个项目分两个阶段:

**离线阶段(建索引,只跑一次或语料变了才跑)** —— `ingest.py`:

```
markdown 文件
   │  parse_doc(): 解析 front-matter,拿到 title / section / source
   ▼
切块 RecursiveCharacterTextSplitter:长文档切成 ~900 字的块
   │
   ▼
embed_documents():每块变成一个固定维度的向量
   │
   ▼
写入 Postgres 的 documents 表(文本 + 向量放一起)
```

**在线阶段(每次提问)** —— `app/services/rag_service.py` 的 `retrieve()` + `answer()`:

```
问题
   │  embed_query():问题也变成向量
   ▼
检索:向量相似度 + 全文关键词,RRF 融合,去重,可选重排序,取 top-k
   │
   ▼
Gate:最好那一块的分够高吗?不够 → 直接拒答
   │
   ▼
生成:把 top-k 块塞进 prompt,LLM 照原文回答,附引用
```

下面逐个概念拆开讲。

---

## 2. 逐个概念拆解(每个都配代码)

### 2.1 数据解析 parse_doc —— 把文档变成「文本 + 元数据」

RAG 不只是要文本,还要知道**这段话来自哪、属于哪个章节、原文 URL 是什么**,后面引用和检索都要靠这些。

[ingest.py](ingest.py) 里:

```python
def parse_doc(path: Path, corpus_root: Path) -> Document | None:
    raw = path.read_text(encoding="utf-8", errors="replace")
    meta = {
        "path": str(path.relative_to(corpus_root)),
        "title": path.stem.replace("-", " "),
        "section": path.parent.name,   # 从文件夹名推断章节,如 hr/
        "source": "",
    }
    m = FRONT_MATTER.match(raw)            # 解析文件顶部的 --- ... --- 元数据
    if m:
        for line in m.group(1).splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                meta[k.strip()] = v.strip()
        raw = raw[m.end():]
    # ...还会解析 <!-- source: ... | section: ... --> 注释
    text = raw.strip()
    if len(text) < 80:                     # 太短的(如空文件)直接丢弃
        return None
    return Document(page_content=text, metadata=meta)
```

`Document` 是 LangChain 的一个简单数据结构,就两个字段:`page_content`(文本)和 `metadata`(一个 dict)。它不是什么黑科技,就是个「文本 + 标签」的封装。

### 2.2 切块 Chunking —— 为什么不能整篇塞进去

两件事决定了必须切块:

1. **上下文长度有限**:一篇长文档塞不下,也没必要。
2. **检索粒度**:你要找到的是「讲 PTO 的那一段」,不是「整个 HR 手册」。

[ingest.py](ingest.py) 用的切分器:

```python
splitter = RecursiveCharacterTextSplitter(
    chunk_size=900,      # 每块最多 ~900 字符
    chunk_overlap=150,   # 相邻块重叠 150 字符
    separators=["\n## ", "\n### ", "\n\n", "\n", " "],
)
chunks = splitter.split_documents(docs)
```

两个关键参数:

- **`chunk_size`**:块越大,上下文越完整,但检索越不精确(可能混进无关内容);越小越精确,但可能把一句话切两半。900 是折中。
- **`chunk_overlap`**:让相邻块重叠,避免一个关键句子正好被切在边界上、两边都不完整。
- **`separators`**:告诉切分器优先在 `## ` 标题处切、其次 `### `、其次空行…… 这样尽量在「章节边界」切,而不是在句子中间乱切。

**面试常问**:切块是 RAG 检索质量的头号变量之一,做不好后面全白搭。

### 2.3 Embedding —— 把文字变成数字

这是 RAG 里最容易让人「一窍不通」但其实就是一层窗户纸的概念。

**Embedding 是一个函数:输入一段文字,输出一个固定长度的数字数组(向量)。** 这个项目的 embedding 模型和维度都是配置(`app/config.py` 里的 `EMBED_MODEL` / `EMBED_DIM`;仓库里的 `.env` 用的是 `qwen3.7-text-embedding`、1024 维,代码默认兜底是 `text-embedding-3-small`、1536 维):

```python
embeddings = OpenAIEmbeddings(model=EMBED_MODEL, api_key=..., base_url=...)
vectors = embeddings.embed_documents([c.page_content for c in chunks])
# 每个 chunk 变成 [0.012, -0.34, 0.88, ...] 共 EMBED_DIM 个数字
```

注意 embedding 客户端还是 `OpenAIEmbeddings`,只是把 `base_url` 指向了别的服务——「OpenAI 兼容」的 HTTP 协议已经是事实标准,很多厂商都提供这套接口。

神奇的地方在于训练方式:**意思相近的文字,向量在空间里"距离"很近**。比如「显示器」和「屏幕」「monitor」的向量会聚在一起,而和「报销流程」离得远。

所以你不用写任何规则去理解「意思」,只要算两个向量的距离,就近似等于算两段话的**语义相似度**。这就是「语义搜索」的本质。

### 2.4 pgvector —— 在 Postgres 里存向量、算相似度

你是全栈工程师,这块会特别亲切:**向量检索不用换数据库,加一个 Postgres 扩展 `pgvector` 就行。**

表定义在 [app/models/index.py](app/models/index.py),是 SQLAlchemy 模型:

```python
class Document(Base):
    __tablename__ = "documents"
    id: Mapped[int] = mapped_column(primary_key=True)
    path: Mapped[str] = mapped_column(Text, nullable=False)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    section: Mapped[str] = mapped_column(Text, nullable=False)   # 以前叫 department
    source: Mapped[str] = mapped_column(Text, nullable=False, default="")
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)   # 原始文本
    preview: Mapped[str] = mapped_column(Text, nullable=False, default="")
    embedding: Mapped[list] = mapped_column(Vector(EMBED_DIM), nullable=False)  # 向量,和文本同一行
    fts: Mapped[str] = mapped_column(TSVECTOR, Computed("to_tsvector('english', content)", persisted=True))
    created_at: ...
```

`Vector(EMBED_DIM)` 是 `pgvector.sqlalchemy` 提供的类型。**把向量和文本放同一行**是这个设计的精髓:你能用熟悉的 SQL/ORM 把「向量相似度」和业务条件 `JOIN` 在一起,不用两套系统。扩展和索引在 [app/database.py](app/database.py) 的 `init_db()` 里幂等创建(向量用 HNSW 索引、`fts` 用 GIN 索引)。

pgvector 给了几个距离运算符:

| 运算符 | 含义 |
|--------|------|
| `<->` | L2 欧氏距离 |
| `<=>` | **余弦距离**(越小越相似) |
| `<#>` | 内积(负值表示更相似) |

这个项目用**余弦相似度**,下面细说。

### 2.5 检索 Retrieve —— cosine 相似度 + top-k

核心查询在 [app/repositories/documents.py](app/repositories/documents.py) 的 `vector_search`:

```python
def vector_search(db: Session, qvec: list, n: int):
    dist = Document.embedding.cosine_distance(qvec)
    stmt = select(Document, (1 - dist).label("score")).order_by(dist).limit(n)
    return [(row[0], float(row[1])) for row in db.execute(stmt).all()]
```

拆开看:

- `Document.embedding.cosine_distance(qvec)`:问题向量和每个文档向量算**余弦距离**。
- `1 - dist`:余弦距离 = 1 − 余弦相似度,所以 `1 - 距离` = **余弦相似度**。结果越接近 1 越相似。这里把它命名为 `score`。
- `order_by(dist).limit(n)`:按相似度降序,取最像的前 n 块 —— 这就是 **top-k 检索**(k 默认 `TOP_K = 4`)。

**一句话总结检索**:把问题也变成向量,然后在所有文档向量里找「方向最接近」的几块。

### 2.6 混合检索 Hybrid —— 关键词 + 语义,取长补短

纯向量检索有一个已知短板:**它对"字面不同但意思相同"很强,但对"精确词、编号、专有名词"反而可能漏。** 比如「on-call」和「oncall」,或者某个具体的政策编号。

所以业界标准做法是**两条腿走路**:

1. **向量检索**:语义像(上面那个)。
2. **全文检索(BM25)**:关键词字面匹配,就是搜索引擎那套,Postgres 内置 `tsvector` + `ts_rank`。

[app/repositories/documents.py](app/repositories/documents.py) 的 `text_search`:

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

- `websearch_to_tsquery` 把用户问题(自然语言)转成搜索词。
- `fts @@ query` 是「命中」判断,`ts_rank` 是相关性打分(BM25 类算法)。

两个列表都有了,怎么合并?用 **RRF(Reciprocal Rank Fusion,倒数排名融合)**:

```python
def rrf_fuse(vector_hits, text_hits, k, k_rank=60):
    rrf = {}
    for rank, h in enumerate(vector_hits, start=1):
        rrf[h["id"]] = rrf.get(h["id"], 0.0) + 1.0 / (k_rank + rank)
    for rank, h in enumerate(text_hits, start=1):
        rrf[h["id"]] = rrf.get(h["id"], 0.0) + 1.0 / (k_rank + rank)
    # 按 rrf 得分降序取前 k 个
```

RRF 的思路很巧妙:**不看两个榜单各自的分数(因为它们的分数尺度不同,没法直接比),只看排名。** 一个文档在向量榜排第 1、在关键词榜排第 3,它的 RRF 分就是 `1/(60+1) + 1/(60+3)`。在两个榜单都靠前的文档,融合后自然排最前。

融合之后,`retrieve()` 还会做两件事:`_dedup_by_path`(每个文档路径最多留一块,避免同一篇文档霸榜 top-k),以及如果开了 `RERANK`,再做一次交叉编码器重排序(见 2.7)。

**这是这个项目简历上最能讲故事的一处**:「加了混合检索后 recall 从 X 提到 Y」,就是靠这套。

### 2.7 重排序 Re-rank(可选)—— 交叉编码器给候选重新打分

检索用的 bi-encoder embedding 快但糙:它把问题和每块**各自独立**编码再比,所以会漏掉细粒度的相关性。**交叉编码器(cross-encoder)** 是把 `(问题, 块)` 当成一个**整体**读进去,直接打分「这块到底答没答这个问题」——慢一些,但准很多。

`retrieve()` 只对已经收窄的候选池做重排(绝不对全库做):

```python
def _rerank(question, hits):
    client = _get_reranker_client()   # OpenAI 兼容客户端,懒加载并缓存
    response = client.post("/reranks", body={
        "model": RERANK_MODEL, "query": question,
        "documents": [h["content"] for h in hits], "top_n": TOP_K,
    }, cast_to=object)
    results = sorted(response["results"], key=lambda r: float(r["relevance_score"]), reverse=True)
    for r in results:
        hits[r["index"]]["rerank"] = round(float(r["relevance_score"]), 4)
    return [hits[r["index"]] for r in results]
```

配置:`RERANK`(默认 `false`)开启、`RERANK_MODEL` 选模型(本项目用 DashScope 的 `qwen3-rerank`)、`RERANK_CANDIDATES` 是喂给它的候选数。这就是经典的 **「粗检索、精排序」两段式**。

### 2.8 相关性 Gate —— 拒答,别让 LLM 编

这是整个项目**最有价值的设计**,也是新手最容易忽略的。

问题:向量检索**总会返回 top-k**,哪怕数据库里根本没有相关内容——它只是返回「最接近的几块」,而不是「相关的几块」。如果不加判断,LLM 会拿着几块无关内容硬编一个答案。

解法:设一个阈值 `MIN_RELEVANCE`,最好那块的分不够就**直接拒答,根本不调 LLM**:

```python
# app/services/rag_service.py 的 answer()
hits, best_signal = retrieve(db, question)
grounded = bool(hits) and best_signal >= _grounding_threshold()   # MIN_RELEVANCE 或 RERANK_MIN_RELEVANCE

if not grounded:
    return {
        "answer": "I cannot find this in the indexed handbook. I will not guess a company policy.",
        "refused": True,
        ...
    }
```

注意一个设计细节:gate 用的信号取决于有没有开重排序。`RERANK` 关时,`best_signal` 是候选里的**最高余弦相似度**(vector / text / hybrid 的 `score` 字段都是 cosine);`RERANK` 开时,`best_signal` 是**最高 rerank 相关性分**,它比 cosine 更能区分「该答」和「不该答」。两者刻度不同,所以各配一个阈值——cosine 用 `MIN_RELEVANCE`,rerank 分用 `RERANK_MIN_RELEVANCE`,由 `_grounding_threshold()` 按 `RERANK` 选。

更前面还有一道更简单的闸门:`_is_english()` 会拒绝带非 ASCII 字母的问题(语料是纯英文),在做任何 embedding 之前就直接拒了。

为什么对政策类场景这么重要?因为「我不确定」比「我编一个错误政策」安全得多。

### 2.9 生成 —— 把检索结果塞进 prompt

检索完了,回到你熟悉的 chatbot 部分。只是 prompt 构造得更讲究:

[app/services/rag_service.py](app/services/rag_service.py) 的 `SYSTEM` 和 `build_messages`:

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

最终发给模型的 prompt 长这样:

```
[system] 你只准根据下面的手册节选回答,不知道就说不知道……
[human]
Handbook excerpts:

[1] title=Time Off section=hr url=...
<检索到的第 1 块原文>

[2] title=Expenses section=finance url=...
<第 2 块原文>

Question: 我能报销显示器吗?
```

关键点:**检索到的原文(而不是摘要)被原样塞进 prompt**,模型被明确指令「只准照这些回答」。这就是「增强(Augmented)」的含义。

关于 `temperature=0`:生成时设为 0,让模型输出更确定、少发挥,适合"查资料"而不是"创作"。chat 模型由 `CHAT_MODEL` 配置;这个仓库用 DeepSeek(通过 `langchain-deepseek`):

```python
def get_llm():
    return ChatDeepSeek(model=CHAT_MODEL, api_key=DEEPSEEK_API_KEY, temperature=0)
```

> **一个刻意的设计说明,关于"多轮"**:RAG 调用本身是**无状态**的——每个问题都从零回答,`build_messages` 不接收任何历史。项目**会**持久化会话和消息(见 2.12)供 UI 展示,但那历史**目前不会喂回给 LLM**。把历史喂回 prompt 是一个已知的未来扩展。

### 2.10 引用 Citations —— 答案可追溯

光有答案不够,还得让用户能点回原文核实。检索命中的每一块都带 `source`(原文 URL),返回时把它整理成引用:

```python
def _citations(hits):
    return [
        {"n": i, "title": h["title"], "section": h["section"],
         "source": h["source"], "score": round(h["score"], 3)}
        for i, h in enumerate(hits, start=1)
    ]
```

前端把这些渲染成答案下方可点击的来源链接,用户既能看答案,也能点链接去 handbook 原文验证——这正是「不幻觉」的落地。(前端不在本文范围,重点是后端返回了结构化、可追溯的引用列表。)

### 2.11 流式 Streaming —— 通过 SSE 逐 token 吐答案

非流式的 `answer()` 一次性返回完整回答。想要更跟手的体验,有 `stream_answer()`,在 LLM 生成过程中**逐个 yield SSE 事件**:

```python
def stream_answer(db, question):
    hits, best_signal = retrieve(db, question)
    grounded = bool(hits) and best_signal >= _grounding_threshold()
    if not grounded:
        yield RefusedEvent(answer="I cannot find this in the indexed handbook. ...")
        return
    yield MetaEvent(citations=_citations(hits))          # 引用先发
    for chunk in get_llm().stream(messages):             # LLM 流式吐 token
        if chunk.content:
            yield TokenEvent(text=chunk.content)         # 每个 token 一个事件
    yield DoneEvent()
```

事件类型(`MetaEvent` / `TokenEvent` / `DoneEvent` / `RefusedEvent` / `ConversationEvent` / `ErrorEvent`)定义成 [app/schemas/chat.py](app/schemas/chat.py) 里的 Pydantic 模型——单一真相源,前端从它生成 TypeScript 类型。路由 [app/routers/chat.py](app/routers/chat.py) 把每个事件序列化成一行 SSE `data: {...}`。

两个值得注意的点:

- **引用在第一个 token 之前就通过 `meta` 事件发出去**——UI 可以在答案还在流式的时候就把来源显示出来。
- **gate 发生在任何生成之前**:拒答时只发一个 `refused` 事件,完全不调 LLM——和非流式路径同样安全。

### 2.12 多会话持久化 —— 存历史,但不喂回历史

应用会存会话,让用户能回来接着看。在 [app/models/history.py](app/models/history.py):

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
    citations: Mapped[list] = mapped_column(JSONB, ...)   # 结构化引用
    refused: Mapped[bool]
```

每次回答后,`history_service.persist_exchange(...)` 把用户问题 + assistant 回答(+ 引用、+ `refused` 标志)写进 Postgres。接口(`GET /conversations`、`GET /conversations/{id}/messages`)把它们暴露给前端。

再强调一次:**持久化 ≠ 上下文。** 这些表给你的是 UI 里的「历史」,它们**不会**被塞回 prompt。如果你想做真正的多轮 RAG,就去读 `Message` 行、把它们传进 `build_messages`——这是这个项目刻意留出的下一步。

### 2.13 评测 Eval —— 怎么知道检索好不好

做完功能,你怎么知道它行不行?不能靠感觉。这就是评测存在的意义,也是区分「demo」和「能上线」的地方。

[eval/metrics.py](eval/metrics.py) 里算几个指标(纯函数,不碰数据库):

- **recall@k**:正确答案(预期的那一块)有没有出现在 top-k 里?「找没找到」。
- **MRR**:找到的话,平均排第几?「找得多准」。
- **拒答 precision / recall**:该拒的拒了没?该答的答了没?其中**「该拒答却回答了」是最危险的漏报**。

核心逻辑:

```python
def evaluate(items, retrieve_fn, min_relevance):
    for q in items:
        hits, best = retrieve_fn(q["question"])
        grounded = bool(hits) and best >= min_relevance   # 系统答了没
        expect = q["expect_in_docs"]                      # 应该答吗

        # 混淆矩阵:「应该答 vs 实际答」
        if expect and grounded:      tp += 1   # 该答,答了 ✓
        elif expect and not grounded: fn += 1  # 该答,拒了 ✗(漏)
        elif not expect and grounded: fp += 1  # 不该答,答了 ✗✗(幻觉!)
        else:                        tn += 1   # 不该答,拒了 ✓
```

金标集 [eval/questions.json](eval/questions.json) 里每道题带 `expect_in_docs`(该不该答)和 `expected`(正确答案该含哪些词):

```json
{ "id": 1, "section": "hr",
  "question": "How does flexible PTO work? ...",
  "expect_in_docs": true,
  "expected": ["pto", "time off", "vacation"] }
```

**面试能讲的点**:你有金标集,能算出 recall@k / MRR / 拒答漏报率,并且能用它验证「加混合检索到底有没有用」——这才是"产品思维",而不是"我又装了个组件"。

---

## 3. LangChain 到底扮演什么角色

这是很多新手最大的困惑,也是这个项目设计上很清醒的一点。

**LangChain 不是一个"必须用的框架",而是一堆工具的集合。** 你完全可以不用它。这个项目只在几处用了它:

| 用到的 | 作用 |
|--------|------|
| `RecursiveCharacterTextSplitter` | 切块(2.2) |
| `OpenAIEmbeddings` | 调(OpenAI 兼容的)embedding 模型(2.3) |
| `ChatDeepSeek` / 消息类型 | 调 chat 模型、组织 prompt(2.9) |

而**最核心的检索、存储、gate、重排序、流式、评测,全是手写 SQL/ORM 和纯 Python**,没碰 LangChain。

这为什么是好事?因为 LangChain 的检索抽象层(`RetrievalQA`、`VectorStoreRetriever` 那套)把很多东西藏进黑盒,新手容易「会调包但不懂原理」。这个项目反其道而行:**检索就是几行你完全看得懂的 ORM**。理解了这层,你再用不用 LangChain 都无所谓——它只是个"方便胶水",不是"知识"。

顺带说清几个容易混的名字:

- **LangChain**:Python/JS 的通用 LLM 工具库(切分、prompt、agents、retrievers……)。
- **`langchain-openai`** / **`langchain-deepseek`**:LangChain 里对接具体厂商的适配包(这个项目两个都用了)。
- **`langchain_core.messages`**:就是 `SystemMessage` / `HumanMessage` / `AIMessage` 这几个"角色 + 内容"的结构体,底层还是一个 `{role, content}`。

---

## 4. 常见误区(面试别踩)

1. **「RAG 就是装个向量数据库」** ❌。向量检索只是 RAG 的一环,切块、gate、重排序、引用、评测同等重要。这个项目最大的亮点恰恰是不靠向量数据库的噱头,而是把「拒答」做扎实。

2. **「embedding 是模型的一部分,调一次就完事」** ❌。embedding 是一个**独立模型**,和生成用的 chat 模型是两个东西。换 `EMBED_MODEL`,向量维度(`EMBED_DIM`)可能变,要重建索引。维度别写死,必须和模型匹配。

3. **「相似度阈值拍脑袋定」** ❌。`MIN_RELEVANCE=0.35` 这种值必须用评测集来调(这个项目文档里反复强调 "tune by eval, not by feel")。而且 cosine 相似度对 embedding 模型而言往往挤在一个窄区间里,不能想当然。

4. **「检索到了就一定能答对」** ❌。检索到「相似的」≠「正确的」,这就是 gate 和评测存在的理由。

5. **「多轮对话只要把用户问题存起来就行」** ❌。真正的多轮 RAG 还要把 assistant 之前的回答也喂回 prompt,否则模型不知道"自己刚才说过啥"。注意 *这个* 项目持久化了会话、但**还没**把历史喂回——它是每问无状态的(见 2.9 / 2.12)。这是简化、不是特性:要做真正的多轮,就把历史喂进去。

---

## 5. 术语表

| 术语 | 一句话解释 |
|------|-----------|
| RAG | 检索增强生成:先查资料再让 LLM 照资料回答 |
| Embedding | 把文字映射成数字向量,语义相近则向量相近 |
| 向量 / vector | 一组浮点数,这里就是 `EMBED_DIM` 个数字 |
| 余弦相似度 cosine similarity | 两个向量"方向多接近",1 = 最像 |
| pgvector | Postgres 扩展,让你能存向量、算距离 |
| top-k | 检索取"最像的前 k 个" |
| Chunking / 切块 | 把长文档切成小块 |
| BM25 / 全文检索 | 关键词匹配打分,搜索引擎那套 |
| RRF | 倒数排名融合:把两个榜单按排名合并 |
| 重排序 re-rank | 交叉编码器给候选重新打分,提精度 |
| Gate / 拒答 | 相似度不够就直接拒绝回答,不让 LLM 编 |
| 幻觉 hallucination | 模型编造看似合理实则错误的内容 |
| SSE | Server-Sent Events,逐 token 流式回答的传输方式 |
| recall@k / MRR | 检索质量指标:找到没 / 排多前 |
| 金标集 gold set | 人工标注好标准答案的评测集 |

---

## 6. 想继续深入,读这些顺序

1. [docs/architecture.md](architecture.md) —— 请求路径 + 各模块职责。
2. [app/services/rag_service.py](app/services/rag_service.py) —— 检索 + 生成 + gate + 重排序 + 流式。
3. [app/repositories/documents.py](app/repositories/documents.py) —— 向量/全文检索查询。
4. [app/models/index.py](app/models/index.py) —— `documents` 表(schema + pgvector 列)。
5. [app/database.py](app/database.py) —— engine、session、索引创建。
6. [eval/metrics.py](eval/metrics.py) —— 评测指标怎么算。

**如果只记三句话:**

1. RAG = 检索 + 塞进 prompt + 生成,你缺的不是生成,是前两步。
2. Embedding 让"语义相似"变成"数字距离",pgvector 让这距离能用 SQL 算。
3. 检索永远会返回"最接近的几块",所以必须加 gate 拒答——这是从 demo 到能上线的分水岭。
