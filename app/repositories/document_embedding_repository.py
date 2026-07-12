import locale
from typing import Any, List

from llama_index.core import Settings as LlamaSettings
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.schema import BaseNode, ImageNode, TextNode
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.vector_stores.postgres import PGVectorStore

from app.core.config import settings
from app.core.logging import get_logger
from app.schemas.ingestion import ContentType, SourceItem

logger = get_logger(__name__)


def _runtime_safe_text(value: str) -> str:
    """Prevent Windows code-page crashes while preserving text when UTF-8 is active."""
    encoding = locale.getpreferredencoding(False) or "utf-8"
    return value.encode(encoding, errors="replace").decode(encoding, errors="replace")


def _runtime_safe_value(value: Any) -> Any:
    if isinstance(value, str):
        return _runtime_safe_text(value)
    if isinstance(value, list):
        return [_runtime_safe_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _runtime_safe_value(item) for key, item in value.items()}
    return value


class DocumentEmbeddingRepository:
    """Stores unified text/table/image-caption items into one pgvector index."""

    def __init__(self) -> None:
        settings.validate_for_indexing()

    def save_batch(self, items: List[SourceItem]) -> bool:
        if not settings.enable_vector_indexing:
            logger.info("Vector indexing disabled by ENABLE_VECTOR_INDEXING=false.")
            return False

        LlamaSettings.llm = AzureOpenAI(
            model=settings.azure_openai_llm_deployment,
            deployment_name=settings.azure_openai_llm_deployment,
            api_key=settings.azure_openai_api_key,
            azure_endpoint=settings.azure_openai_endpoint,
            api_version=settings.azure_openai_api_version,
        )
        LlamaSettings.embed_model = AzureOpenAIEmbedding(
            model=settings.azure_openai_embedding_deployment,
            deployment_name=settings.azure_openai_embedding_deployment,
            api_key=settings.azure_openai_api_key,
            azure_endpoint=settings.azure_openai_endpoint,
            api_version=settings.azure_openai_api_version,
        )
        vector_store = PGVectorStore.from_params(
            connection_string=settings.pgvector_connection_string,
            async_connection_string=settings.pgvector_async_connection_string,
            table_name=settings.db_table_name,
            embed_dim=settings.embedding_dims,
        )
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        nodes = [self._to_llama_node(item) for item in items]
        VectorStoreIndex(nodes, storage_context=storage_context)
        return True

    def _to_llama_node(self, item: SourceItem) -> BaseNode:
        metadata = _runtime_safe_value(item.metadata.model_dump(mode="json"))
        searchable_text = _runtime_safe_text(item.searchable_text)
        if item.metadata.content_type == ContentType.IMAGE_CAPTION and item.metadata.image_path:
            return ImageNode(
                id_=item.item_id,
                text=searchable_text,
                image_path=item.metadata.image_path,
                metadata=metadata,
            )
        return TextNode(
            id_=item.item_id,
            text=searchable_text,
            metadata=metadata,
        )
