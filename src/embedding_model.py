from sentence_transformers import SentenceTransformer
from time import perf_counter

from config import CONFIG


MODEL_NAME = CONFIG.embedding_model


class EmbeddingModel:
    """
    Wrapper around the Sentence Transformers embedding model.
    """

    def __init__(self, model_name=MODEL_NAME):

        started_at = perf_counter()

        print("=" * 80)
        print("GRBL CNC AI - EMBEDDING MODEL")
        print("=" * 80)

        print()
        print(f"Model: {model_name}")
        print()

        self.model_name = model_name

        self.model = SentenceTransformer(
            model_name
        )

        print("Embedding model loaded.")
        print()
        self.initialization_seconds = perf_counter() - started_at

    def encode(
        self,
        texts,
        batch_size=32,
        normalize_embeddings=True,
        show_progress_bar=True
    ):

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=normalize_embeddings,
            show_progress_bar=show_progress_bar,
            convert_to_numpy=True
        )

        return embeddings

    def encode_query(self, query):

        return self.encode(
            [query],
            show_progress_bar=False
        )[0]
