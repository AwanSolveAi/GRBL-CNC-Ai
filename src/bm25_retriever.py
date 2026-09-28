from pathlib import Path
import json
import re
from time import perf_counter

from metadata_utils import normalize_metadata


BASE_DIR = Path(__file__).resolve().parent.parent

CORPUS_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "grbl_corpus.json"
)


def tokenize(text):
    """
    Technical tokenizer designed for CNC/GRBL terminology.

    Keeps tokens such as:
        $100
        $101
        G38.2
        G90
        M3
        Alarm
    """

    text = text.lower()

    # Keep decimal G-code commands together.
    text = re.sub(
        r"([gm]\d+(?:\.\d+)?)",
        r" \1 ",
        text
    )

    # Keep GRBL settings together.
    text = re.sub(
        r"(\$\d+)",
        r" \1 ",
        text
    )

    # Split remaining punctuation.
    tokens = re.findall(
        r"\$?\w+(?:\.\w+)?",
        text
    )

    return tokens


class BM25Retriever:

    def __init__(self, corpus_path=CORPUS_PATH):

        started_at = perf_counter()

        from rank_bm25 import BM25Okapi

        print("=" * 80)
        print("GRBL CNC AI - BM25 RETRIEVER")
        print("=" * 80)
        print()

        self.corpus_path = Path(
            corpus_path
        )

        with self.corpus_path.open(
            "r",
            encoding="utf-8"
        ) as file:

            self.chunks = json.load(file)

        print(
            f"Corpus chunks: {len(self.chunks)}"
        )

        self.documents = [
            self.build_document(chunk)
            for chunk in self.chunks
        ]

        self.tokenized_documents = [
            tokenize(document)
            for document in self.documents
        ]

        self.bm25 = BM25Okapi(
            self.tokenized_documents
        )

        print(
            "BM25 index created."
        )

        print()
        self.initialization_seconds = perf_counter() - started_at

    @staticmethod
    def build_document(chunk):

        parts = [
            chunk.get("source", ""),
            chunk.get("controller", ""),
            chunk.get("version", ""),
            chunk.get("category", ""),
            chunk.get("section", ""),
            chunk.get("section_path", ""),
            chunk.get("text", ""),
        ]

        parameters = chunk.get(
            "grbl_parameters",
            []
        )

        if parameters:

            parts.append(
                " ".join(parameters)
            )

        gcode_commands = chunk.get(
            "gcode_commands",
            []
        )

        if gcode_commands:

            parts.append(
                " ".join(gcode_commands)
            )

        alarms = chunk.get(
            "alarms",
            []
        )

        if alarms:

            parts.append(
                " ".join(alarms)
            )

        return " ".join(
            str(part)
            for part in parts
            if part
        )

    def search(
        self,
        query,
        top_k=10
    ):

        query_tokens = tokenize(
            query
        )

        scores = self.bm25.get_scores(
            query_tokens
        )

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )

        results = []

        for rank, index in enumerate(
            ranked_indices[:top_k],
            start=1
        ):

            chunk = normalize_metadata(self.chunks[index])

            results.append(
                {
                    "rank": rank,

                    "score": float(
                        scores[index]
                    ),

                    "chunk_id":
                        chunk.get(
                            "chunk_id"
                        ),

                    "source":
                        chunk.get(
                            "source"
                        ),

                    "document":
                        chunk.get(
                            "document"
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

                    "text":
                        chunk.get(
                            "text"
                        ),

                    "metadata":
                        chunk,
                }
            )

        return results


def main():

    retriever = BM25Retriever()

    test_queries = [
        "What does $100 control?",
        "What does G90 do?",
        "What does G38.2 do?",
        "What does M3 do?",
        "What is Alarm 1?",
    ]

    for query in test_queries:

        print()
        print("=" * 80)
        print(f"QUERY: {query}")
        print("=" * 80)

        results = retriever.search(
            query,
            top_k=5
        )

        for result in results:

            print()
            print(
                f"Rank:  {result['rank']}"
            )

            print(
                f"Score: {result['score']:.4f}"
            )

            print(
                f"ID:    {result['chunk_id']}"
            )

            print(
                f"Category: "
                f"{result['category']}"
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
                f"{preview[:250]}..."
            )


if __name__ == "__main__":
    main()
