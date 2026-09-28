import json
from pathlib import Path
from time import perf_counter

from config import CONFIG

# =============================================================================
# CONFIGURATION
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CORPUS_PATH = PROJECT_ROOT / "data" / "processed" / "grbl_corpus.json"

MODEL_NAME = CONFIG.reranker_model

TOP_K_CANDIDATES = 20
TOP_K_FINAL = 5


# =============================================================================
# LAZY RERANKER
# =============================================================================

_reranker = None
_initialization_seconds = None


def get_reranker():
    """Load and cache the cross-encoder only when reranking is requested."""

    global _reranker, _initialization_seconds

    if _reranker is None:
        from sentence_transformers import CrossEncoder

        started_at = perf_counter()
        _reranker = CrossEncoder(MODEL_NAME)
        _initialization_seconds = perf_counter() - started_at

    return _reranker


def get_initialization_seconds():
    return _initialization_seconds


# =============================================================================
# RERANK FUNCTION
# =============================================================================

def rerank(query, candidates, top_k=TOP_K_FINAL):
    """
    Rerank retrieved GRBL documents using a Cross-Encoder.

    Parameters
    ----------
    query : str
        User's question.

    candidates : list
        Candidate document dictionaries.

    top_k : int
        Number of final documents to return.

    Returns
    -------
    list
        Reranked documents.
    """

    if not candidates:
        return []

    pairs = []

    for candidate in candidates:
        text = candidate.get("text", "")

        # Include section information because GRBL commands/settings
        # are often identified directly by their section title.
        section = candidate.get("section", "")

        combined_text = f"{section}\n{text}"

        pairs.append((query, combined_text))

    scores = get_reranker().predict(
        pairs,
        show_progress_bar=False
    )

    results = []

    for candidate, score in zip(candidates, scores):
        item = dict(candidate)

        item["reranker_score"] = float(score)

        results.append(item)

    results.sort(
        key=lambda x: x["reranker_score"],
        reverse=True
    )

    return results[:top_k]


# =============================================================================
# HELPER: GET CHUNK
# =============================================================================

def get_chunk(chunk_id, corpus=None):
    """
    Find a corpus chunk by ID.
    """

    if corpus is None:
        with open(CORPUS_PATH, "r", encoding="utf-8") as file:
            corpus = json.load(file)

    for chunk in corpus:
        if chunk.get("chunk_id") == chunk_id:
            return chunk

    return None


# =============================================================================
def main():
    """Run the original standalone reranker demonstration."""

    with open(CORPUS_PATH, "r", encoding="utf-8") as file:
        corpus = json.load(file)

    test_chunk_ids = [
        "settings_0029", "grbl_gcode_0012", "grbl_gcode_0023",
        "grbl_gcode_0024", "grbl_gcode_0025", "grbl_gcode_0026",
        "grbl_gcode_0032", "commands_0013", "settings_0030",
        "settings_0031", "settings_0032",
    ]
    test_candidates = [
        chunk for chunk_id in test_chunk_ids
        if (chunk := get_chunk(chunk_id, corpus)) is not None
    ]
    test_queries = [
        "What does $100 control?", "What does G90 do?",
        "What does G38.2 do?", "What does M3 do?", "What is Alarm 1?",
        "My X axis moves the wrong distance. What should I check?",
    ]

    for query in test_queries:
        print("\n" + "=" * 80)
        print(f"QUERY: {query}")
        for rank, result in enumerate(
            rerank(query, test_candidates, top_k=5), start=1
        ):
            text = result.get("text", "").replace("\n", " ")
            print(
                f"{rank}. {result.get('chunk_id')} "
                f"{result['reranker_score']:.6f} {text[:350]}"
            )


if __name__ == "__main__":
    main()
