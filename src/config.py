"""Small environment-based operational configuration."""

import os
from dataclasses import dataclass


def _positive_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    value = int(raw)
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _boolean(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean value")


@dataclass(frozen=True)
class AppConfig:
    ollama_url: str = os.getenv("OLLAMA_URL", "http://localhost:11434")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
    ollama_num_predict: int = _positive_int("GRBL_OLLAMA_NUM_PREDICT", 200)
    embedding_model: str = os.getenv(
        "GRBL_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    )
    reranker_model: str = os.getenv(
        "GRBL_RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"
    )
    vector_top_k: int = _positive_int("GRBL_VECTOR_TOP_K", 20)
    bm25_top_k: int = _positive_int("GRBL_BM25_TOP_K", 20)
    hybrid_top_k: int = _positive_int("GRBL_HYBRID_TOP_K", 20)
    rerank_top_k: int = _positive_int("GRBL_RERANK_TOP_K", 10)
    answer_context_k: int = _positive_int("GRBL_ANSWER_CONTEXT_K", 5)
    force_rerank_all: bool = _boolean("GRBL_FORCE_RERANK_ALL", False)


CONFIG = AppConfig()
