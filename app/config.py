import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parents[1]


DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
EMBED_DIM = int(os.getenv("EMBED_DIM", "1536"))
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://rag:rag@127.0.0.1:5432/handbook",
)
CORPUS_DIR = os.getenv("CORPUS_DIR", str(ROOT / "data"))
TOP_K = int(os.getenv("TOP_K", "4"))
MIN_RELEVANCE = float(os.getenv("MIN_RELEVANCE", "0.35"))
RETRIEVAL_MODE = os.getenv("RETRIEVAL_MODE", "hybrid")  # vector | text | hybrid
SEARCH_K = int(os.getenv("SEARCH_K", "20"))
RERANK = os.getenv("RERANK", "false").lower() in {"1", "true", "yes", "on"}
RERANK_MODEL = os.getenv("RERANK_MODEL", "qwen3-rerank")
RERANK_CANDIDATES = int(os.getenv("RERANK_CANDIDATES", "20"))
# Grounding threshold for the rerank score when RERANK is on. Rerank relevance
# scores are on a different scale than cosine, so they need their own threshold
# (0.74 = best-accuracy operating point from eval --compare-grounding).
RERANK_MIN_RELEVANCE = float(os.getenv("RERANK_MIN_RELEVANCE", "0.74"))

EMBED_BASE_URL = os.getenv("EMBED_BASE_URL", "")
EMBED_API_KEY = os.getenv("DASHSCOPE_API_KEY", "")

DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "")
DASHSCOPE_WORKSPACE_ID = os.getenv("DASHSCOPE_WORKSPACE_ID", "")

FRONTEND_ORIGIN = os.getenv("FRONTEND_ORIGIN", "http://localhost:3000")