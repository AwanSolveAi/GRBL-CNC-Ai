from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json

import faiss
import numpy as np

from embedding_model import EmbeddingModel
from metadata_utils import normalize_metadata


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

CORPUS_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "grbl_corpus.json"
)

VECTOR_DIR = (
    BASE_DIR
    / "data"
    / "vector_store"
)

INDEX_PATH = (
    VECTOR_DIR
    / "grbl_faiss.index"
)

EMBEDDINGS_PATH = (
    VECTOR_DIR
    / "grbl_embeddings.npy"
)

METADATA_PATH = (
    VECTOR_DIR
    / "metadata.json"
)

MANIFEST_PATH = VECTOR_DIR / "index_manifest.json"


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


# ============================================================
# RETRIEVAL TEXT
# ============================================================

def build_retrieval_text(chunk):

    parts = [
        f"Source: {chunk.get('source', '')}",
        f"Controller: {chunk.get('controller', '')}",
        f"Version: {chunk.get('version', '')}",
        f"Category: {chunk.get('category', '')}",
        f"Section: {chunk.get('section', '')}",
        f"Section path: {chunk.get('section_path', '')}",
    ]

    parameters = chunk.get(
        "grbl_parameters",
        []
    )

    if parameters:

        parts.append(
            "GRBL parameters: "
            + ", ".join(parameters)
        )

    gcode_commands = chunk.get(
        "gcode_commands",
        []
    )

    if gcode_commands:

        parts.append(
            "G-code commands: "
            + ", ".join(gcode_commands)
        )

    alarms = chunk.get(
        "alarms",
        []
    )

    if alarms:

        parts.append(
            "Alarms: "
            + ", ".join(alarms)
        )

    parts.append(
        f"Content: {chunk.get('text', '')}"
    )

    return "\n".join(parts)


# ============================================================
# MAIN
# ============================================================

def main():

    VECTOR_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print()
    print("=" * 80)
    print("GRBL CNC AI - BUILD VECTOR INDEX")
    print("=" * 80)
    print()

    # --------------------------------------------------------
    # Load corpus
    # --------------------------------------------------------

    print(
        f"Loading corpus:"
    )

    print(
        CORPUS_PATH
    )

    with CORPUS_PATH.open(
        "r",
        encoding="utf-8"
    ) as file:

        chunks = json.load(
            file
        )

    print(
        f"Chunks loaded: "
        f"{len(chunks)}"
    )

    print()

    if not chunks:

        raise RuntimeError(
            "Corpus is empty."
        )

    # --------------------------------------------------------
    # Build retrieval documents
    # --------------------------------------------------------

    print(
        "Building retrieval documents..."
    )

    retrieval_texts = [
        build_retrieval_text(chunk)
        for chunk in chunks
    ]

    print(
        f"Retrieval documents: "
        f"{len(retrieval_texts)}"
    )

    print()

    # --------------------------------------------------------
    # Embeddings
    # --------------------------------------------------------

    embedding_model = EmbeddingModel()

    print(
        "Generating embeddings..."
    )

    embeddings = embedding_model.encode(
        retrieval_texts,
        batch_size=32,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    embeddings = np.asarray(
        embeddings,
        dtype="float32"
    )

    print()

    print(
        f"Embedding shape: "
        f"{embeddings.shape}"
    )

    # --------------------------------------------------------
    # FAISS
    # --------------------------------------------------------

    dimension = embeddings.shape[1]

    print()

    print(
        f"FAISS dimension: "
        f"{dimension}"
    )

    print(
        "Creating FAISS index..."
    )

    # Because embeddings are normalized,
    # inner product is equivalent to cosine similarity.
    index = faiss.IndexFlatIP(
        dimension
    )

    index.add(
        embeddings
    )

    print(
        f"FAISS vectors: "
        f"{index.ntotal}"
    )

    # --------------------------------------------------------
    # Save FAISS
    # --------------------------------------------------------

    faiss.write_index(
        index,
        str(INDEX_PATH)
    )

    print()
    print(
        f"Saved FAISS index:"
    )

    print(
        INDEX_PATH
    )

    # --------------------------------------------------------
    # Save embeddings
    # --------------------------------------------------------

    np.save(
        EMBEDDINGS_PATH,
        embeddings
    )

    print()
    print(
        "Saved embeddings:"
    )

    print(
        EMBEDDINGS_PATH
    )

    # --------------------------------------------------------
    # Save metadata
    # --------------------------------------------------------

    metadata = []

    for chunk in chunks:

        chunk = normalize_metadata(chunk)

        metadata.append(
            {
                "chunk_id":
                    chunk.get(
                        "chunk_id"
                    ),

                "source":
                    chunk.get(
                        "source"
                    ),

                "source_url": chunk.get("source_url"),

                "source_document": chunk.get("source_document"),

                "source_section": chunk.get("source_section"),

                "source_type": chunk.get("source_type"),

                "retrieved_at": chunk.get("retrieved_at"),

                "upstream_revision": chunk.get("upstream_revision"),

                "checksum": chunk.get("checksum"),

                "document":
                    chunk.get(
                        "document"
                    ),

                "document_type":
                    chunk.get(
                        "document_type"
                    ),

                "controller":
                    chunk.get(
                        "controller"
                    ),

                "version":
                    chunk.get(
                        "version"
                    ),

                "authority":
                    chunk.get(
                        "authority"
                    ),

                "category":
                    chunk.get(
                        "category"
                    ),

                "section":
                    chunk.get(
                        "section"
                    ),

                "section_path":
                    chunk.get(
                        "section_path"
                    ),

                "chunk_index":
                    chunk.get(
                        "chunk_index"
                    ),

                "grbl_parameters":
                    chunk.get(
                        "grbl_parameters",
                        []
                    ),

                "gcode_commands":
                    chunk.get(
                        "gcode_commands",
                        []
                    ),

                "alarms":
                    chunk.get(
                        "alarms",
                        []
                    ),

                "text":
                    chunk.get(
                        "text"
                    ),
            }
        )

    with METADATA_PATH.open(
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            metadata,
            file,
            indent=2,
            ensure_ascii=False
        )

    manifest = {
        "embedding_model": embedding_model.model_name,
        "embedding_dimension": int(dimension),
        "faiss_vector_count": int(index.ntotal),
        "metadata_record_count": len(metadata),
        "corpus_sha256": sha256_file(CORPUS_PATH),
        "index_sha256": sha256_file(INDEX_PATH),
        "metadata_sha256": sha256_file(METADATA_PATH),
        "build_identity": sha256_file(CORPUS_PATH),
        "built_at": datetime.now(timezone.utc).isoformat(),
    }
    with MANIFEST_PATH.open("w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2)

    print()
    print(
        "Saved metadata:"
    )

    print(
        METADATA_PATH
    )

    # --------------------------------------------------------
    # Final verification
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("VECTOR INDEX BUILD COMPLETE")
    print("=" * 80)

    print(
        f"Documents:       {len(chunks)}"
    )

    print(
        f"Embedding shape: {embeddings.shape}"
    )

    print(
        f"FAISS vectors:   {index.ntotal}"
    )

    print()
    print("Files:")

    print(
        f"  {INDEX_PATH}"
    )

    print(
        f"  {EMBEDDINGS_PATH}"
    )

    print(
        f"  {METADATA_PATH}"
    )

    print("=" * 80)


if __name__ == "__main__":
    main()
