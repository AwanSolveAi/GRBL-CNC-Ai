from pathlib import Path
import json


BASE_DIR = Path(__file__).resolve().parent.parent

CORPUS_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "grbl_corpus.json"
)


def main():
    if not CORPUS_PATH.exists():
        print(f"ERROR: Corpus not found:")
        print(CORPUS_PATH)
        return

    with CORPUS_PATH.open(
        "r",
        encoding="utf-8"
    ) as file:
        documents = json.load(file)

    print("=" * 80)
    print("GRBL CNC AI - CORPUS INSPECTION")
    print("=" * 80)

    print()
    print(f"Corpus file: {CORPUS_PATH}")
    print(f"Total chunks: {len(documents)}")

    print()
    print("=" * 80)
    print("FIRST 20 CHUNKS")
    print("=" * 80)

    for document in documents[:20]:

        print(
            f"{document['chunk_id']} | "
            f"{document['category']} | "
            f"{document['section_path']}"
        )

    print()
    print("=" * 80)
    print("FIRST CHUNK - FULL METADATA")
    print("=" * 80)

    if documents:
        print(
            json.dumps(
                documents[0],
                indent=2,
                ensure_ascii=False
            )
        )

    print()
    print("=" * 80)


if __name__ == "__main__":
    main()