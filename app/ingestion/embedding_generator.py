class EmbeddingGenerator:
    """Generates vector embeddings for document chunks."""

    def embed_text(self, text: str) -> list[float]:
        raise NotImplementedError

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError
