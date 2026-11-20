# config.py (Erweiterung)
from dataclasses import dataclass
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

@dataclass(frozen=True)
class Settings:
    openrouter_api_key: str = os.environ.get("OPENROUTER_API_KEY", "")
    openrouter_base_url: str = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    pdf_dir: str = os.environ.get("RAG_PDF_DIR", "")
    qdrant_host: str = os.environ.get("QDRANT_HOST", "localhost")
    qdrant_grpc_port: int = int(os.environ.get("QDRANT_GRPC_PORT", "6334"))
    collection: str = os.environ.get("RAG_COLLECTION_NAME", "CollectionWithData")
    embedding_model: str = os.environ.get("EMBEDDING_MODEL", "openai/text-embedding-3-large")
    vector_size: int = 3072
    chunk_tokens: int = int(os.environ.get("RAG_CHUNK_TOKENS", "500"))
    chunk_overlap: int = int(os.environ.get("RAG_CHUNK_OVERLAP", "50"))
    # Neu für den Chat:
    chat_model: str = os.environ.get("CHAT_MODEL", "openai/gpt-6-luna")
    top_k: int = int(os.environ.get("RAG_TOP_K", "5"))
    score_threshold: float = float(os.environ.get("RAG_SCORE_THRESHOLD", "0.25"))
    max_context_tokens: int = int(os.environ.get("RAG_MAX_CONTEXT_TOKENS", "2000"))
    max_answer_tokens: int = int(os.environ.get("RAG_MAX_ANSWER_TOKENS", "400"))
    # unten in Settings ergänzen:
    candidate_k: int = int(os.environ.get("RAG_CANDIDATE_K", "20"))
    mmr_lambda: float = float(os.environ.get("RAG_MMR_LAMBDA", "0.5"))
    doc_filter: str = os.environ.get("RAG_DOC_FILTER", "").strip()
    stream: bool = os.environ.get("RAG_STREAM", "false").lower() in {"1","true","yes"}

    def require_api_key(self) -> str:
        if not self.openrouter_api_key.strip():
            raise SystemExit("Bitte OPENROUTER_API_KEY in der .env im Repository-Hauptverzeichnis eintragen.")
        return self.openrouter_api_key

