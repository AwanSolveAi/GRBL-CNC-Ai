from pathlib import Path
import hashlib
import json

from bm25_retriever import BM25Retriever
from metadata_utils import normalize_metadata


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

VECTOR_DIR = (
    BASE_DIR
    / "data"
    / "vector_store"
)

INDEX_PATH = (
    VECTOR_DIR
    / "grbl_faiss.index"
)

METADATA_PATH = (
    VECTOR_DIR
    / "metadata.json"
)

CORPUS_PATH = BASE_DIR / "data" / "processed" / "grbl_corpus.json"
MANIFEST_PATH = VECTOR_DIR / "index_manifest.json"


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


# ============================================================
# HYBRID RETRIEVER
# ============================================================

class HybridRetriever:

    def __init__(
        self,
        vector_top_k=20,
        bm25_top_k=20
    ):

        print("=" * 80)
        print("GRBL CNC AI - HYBRID RETRIEVER")
        print("=" * 80)
        print()

        self.vector_top_k = vector_top_k
        self.bm25_top_k = bm25_top_k
        self.degraded_reasons = []

        # BM25 remains available even when optional ML dependencies fail.
        self.bm25 = BM25Retriever()

        # ----------------------------------------------------
        # FAISS
        # ----------------------------------------------------

        self.index = None
        self.metadata = []
        self.embedding_model = None
        self._np = None

        try:
            import faiss
            import numpy as np
            from embedding_model import EmbeddingModel, MODEL_NAME

            print("Loading FAISS index...")
            self.index = faiss.read_index(str(INDEX_PATH))

            print("Loading metadata...")
            with METADATA_PATH.open("r", encoding="utf-8") as file:
                self.metadata = json.load(file)

            if self.index.ntotal != len(self.metadata):
                raise RuntimeError(
                    "FAISS vector count does not match metadata records."
                )

            if MANIFEST_PATH.exists():
                with MANIFEST_PATH.open("r", encoding="utf-8") as file:
                    manifest = json.load(file)
                expected_model = manifest.get("embedding_model")
                if expected_model and expected_model != MODEL_NAME:
                    raise RuntimeError(
                        "Configured embedding model does not match index manifest: "
                        f"{expected_model}"
                    )
                checks = {
                    "embedding dimension": (self.index.d, manifest.get("embedding_dimension")),
                    "FAISS vector count": (self.index.ntotal, manifest.get("faiss_vector_count")),
                    "metadata record count": (len(self.metadata), manifest.get("metadata_record_count")),
                    "corpus checksum": (sha256_file(CORPUS_PATH), manifest.get("corpus_sha256")),
                }
                for label, (actual, expected) in checks.items():
                    if expected is not None and actual != expected:
                        raise RuntimeError(
                            f"Index compatibility error: {label} is {actual}, "
                            f"expected {expected}."
                        )

            self.embedding_model = EmbeddingModel()
            if self.index.d != len(self.embedding_model.encode_query("dimension check")):
                raise RuntimeError(
                    "Embedding output dimension is incompatible with the FAISS index."
                )
            self._np = np
            print(f"FAISS vectors: {self.index.ntotal}")

        except Exception as exc:
            self.index = None
            self.metadata = []
            self.embedding_model = None
            self.degraded_reasons.append(
                f"Vector retrieval unavailable: {exc}"
            )
            print("WARNING: Vector retrieval unavailable; using BM25 only.")
            print(f"Reason: {exc}")

        print()
        print(
            "Hybrid retriever ready."
        )
        print()

    # ========================================================
    # VECTOR SEARCH
    # ========================================================

    def vector_search(
        self,
        query,
        top_k=None
    ):

        if self.index is None or self.embedding_model is None:
            return []

        if top_k is None:

            top_k = self.vector_top_k

        query_embedding = (
            self.embedding_model
            .encode_query(query)
        )

        query_embedding = self._np.asarray(
            [query_embedding],
            dtype="float32"
        )

        scores, indices = (
            self.index.search(
                query_embedding,
                top_k
            )
        )

        results = []

        for rank, (score, index) in enumerate(
            zip(
                scores[0],
                indices[0]
            ),
            start=1
        ):

            if index < 0:
                continue

            metadata = normalize_metadata(self.metadata[index])

            results.append(
                {
                    "rank": rank,

                    "score": float(score),

                    "chunk_id":
                        metadata[
                            "chunk_id"
                        ],

                    "source":
                        metadata[
                            "source"
                        ],

                    "document":
                        metadata.get(
                            "document"
                        ),

                    "category":
                        metadata[
                            "category"
                        ],

                    "section":
                        metadata[
                            "section"
                        ],

                    "section_path":
                        metadata[
                            "section_path"
                        ],

                    "text":
                        metadata[
                            "text"
                        ],

                    "metadata":
                        metadata,
                }
            )

        return results

    # ========================================================
    # RRF
    # ========================================================

    @staticmethod
    def reciprocal_rank_fusion(
        vector_results,
        bm25_results,
        k=60,
        vector_weight=1.0,
        bm25_weight=1.0,
    ):

        fused = {}

        # ----------------------------------------------------
        # Vector rankings
        # ----------------------------------------------------

        for result in vector_results:

            chunk_id = result[
                "chunk_id"
            ]

            if chunk_id not in fused:

                fused[chunk_id] = {
                    "chunk_id":
                        chunk_id,

                    "rrf_score":
                        0.0,

                    "vector_rank":
                        None,

                    "bm25_rank":
                        None,

                    "vector_score":
                        None,

                    "bm25_score":
                        None,

                    "result":
                        result,
                }

            fused[
                chunk_id
            ][
                "rrf_score"
            ] += (
                vector_weight
                /
                (
                    k
                    +
                    result["rank"]
                )
            )

            fused[
                chunk_id
            ][
                "vector_rank"
            ] = result["rank"]

            fused[
                chunk_id
            ][
                "vector_score"
            ] = result["score"]

        # ----------------------------------------------------
        # BM25 rankings
        # ----------------------------------------------------

        for result in bm25_results:

            chunk_id = result[
                "chunk_id"
            ]

            if chunk_id not in fused:

                fused[chunk_id] = {
                    "chunk_id":
                        chunk_id,

                    "rrf_score":
                        0.0,

                    "vector_rank":
                        None,

                    "bm25_rank":
                        None,

                    "vector_score":
                        None,

                    "bm25_score":
                        None,

                    "result":
                        result,
                }

            fused[
                chunk_id
            ][
                "rrf_score"
            ] += (
                bm25_weight
                /
                (
                    k
                    +
                    result["rank"]
                )
            )

            fused[
                chunk_id
            ][
                "bm25_rank"
            ] = result["rank"]

            fused[
                chunk_id
            ][
                "bm25_score"
            ] = result["score"]

        ranked = sorted(
            fused.values(),
            key=lambda x: x[
                "rrf_score"
            ],
            reverse=True
        )

        final_results = []

        for rank, item in enumerate(
            ranked,
            start=1
        ):

            result = item[
                "result"
            ].copy()

            result[
                "hybrid_rank"
            ] = rank

            result[
                "rrf_score"
            ] = item[
                "rrf_score"
            ]

            result[
                "vector_rank"
            ] = item[
                "vector_rank"
            ]

            result[
                "bm25_rank"
            ] = item[
                "bm25_rank"
            ]

            result[
                "vector_score"
            ] = item[
                "vector_score"
            ]

            result[
                "bm25_score"
            ] = item[
                "bm25_score"
            ]

            final_results.append(
                result
            )

        return final_results

    # ========================================================
    # HYBRID SEARCH
    # ========================================================

    def search(
        self,
        query,
        top_k=10,
        query_info=None,
    ):

        query_type = (query_info or {}).get("query_type", "general")
        retrieval_query = (query_info or {}).get("retrieval_query", query)
        exact_types = {
            "exact_setting", "gcode", "mcode", "alarm", "system_command"
        }
        exact_query = query_type in exact_types

        vector_results = (
            self.vector_search(
                retrieval_query,
                min(self.vector_top_k, 10) if exact_query else self.vector_top_k
            )
        )

        bm25_results = (
            self.bm25.search(
                retrieval_query,
                self.bm25_top_k
            )
        )

        hybrid_results = (
            self.reciprocal_rank_fusion(
                vector_results,
                bm25_results,
                vector_weight=1.0,
                bm25_weight=2.0 if exact_query else 1.0,
            )
        )

        strategy = "exact_bm25_weighted" if exact_query else "hybrid"
        if not vector_results:
            strategy += "_degraded"

        for result in hybrid_results:
            result["retrieval_strategy"] = strategy
            result["query_type"] = query_type

        return hybrid_results[
            :top_k
        ]


# ============================================================
# TEST
# ============================================================

def main():

    retriever = HybridRetriever()

    test_queries = [

        "What does $100 control?",

        "What does G90 do?",

        "What does G38.2 do?",

        "What does M3 do?",

        "What is Alarm 1?",

        "My X axis moves the wrong distance. "
        "What should I check?",

    ]

    for query in test_queries:

        print()
        print("=" * 80)
        print(
            f"QUERY: {query}"
        )
        print("=" * 80)

        results = retriever.search(
            query,
            top_k=5
        )

        for result in results:

            print()
            print(
                f"Hybrid Rank: "
                f"{result['hybrid_rank']}"
            )

            print(
                f"RRF Score: "
                f"{result['rrf_score']:.6f}"
            )

            print(
                f"Chunk: "
                f"{result['chunk_id']}"
            )

            print(
                f"Category: "
                f"{result['category']}"
            )

            print(
                f"Vector Rank: "
                f"{result['vector_rank']}"
            )

            print(
                f"BM25 Rank: "
                f"{result['bm25_rank']}"
            )

            print(
                f"Section: "
                f"{result['section']}"
            )

            preview = (
                result["text"]
                .replace("\n", " ")
            )

            print(
                f"Text: "
                f"{preview[:300]}..."
            )


if __name__ == "__main__":
    main()
