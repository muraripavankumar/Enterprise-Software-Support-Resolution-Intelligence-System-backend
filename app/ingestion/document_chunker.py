from typing import Any, Dict, List

from llama_index.core import Document
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding

from app.core.config import settings
from app.ingestion.text_normalizer import normalize_extracted_text, remove_markdown_tables


class DocumentChunker:
    """Chunks parsed text using LlamaIndex SemanticSplitterNodeParser."""

    def __init__(self) -> None:
        embed_model = AzureOpenAIEmbedding(
            model=settings.azure_openai_embedding_deployment,
            deployment_name=settings.azure_openai_embedding_deployment,
            api_key=settings.azure_openai_api_key,
            azure_endpoint=settings.azure_openai_endpoint,
            api_version=settings.azure_openai_api_version,
        )
        self.parser = SemanticSplitterNodeParser(
            buffer_size=1,
            breakpoint_percentile_threshold=settings.semantic_breakpoint_percentile,
            embed_model=embed_model,
        )

    def chunk_pages(self, pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        documents = [
            Document(
                text=remove_markdown_tables(normalize_extracted_text(str(page.get("text", "")))),
                metadata={
                    **dict(page.get("metadata", {}) or {}),
                    "page_number": page.get("page_number"),
                },
            )
            for page in pages
            if remove_markdown_tables(normalize_extracted_text(str(page.get("text", "")))).strip()
        ]
        nodes = self.parser.get_nodes_from_documents(documents)
        return [
            {
                "text": node.get_content(),
                "page_number": dict(node.metadata or {}).get("page_number"),
                "chunk_index": index,
                "metadata": {
                    **dict(node.metadata or {}),
                    "chunking_strategy": "llamaindex_semantic_splitter",
                    "llamaindex_node_id": node.node_id,
                },
            }
            for index, node in enumerate(nodes)
        ]


