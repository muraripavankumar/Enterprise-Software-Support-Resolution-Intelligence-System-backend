import logging
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.core.config import settings
from app.ingestion.document_chunker import DocumentChunker
from app.ingestion.document_parser import DocumentParser
from app.ingestion.image_captioner import ImageCaptioner
from app.ingestion.image_extractor import ImageExtractor
from app.ingestion.table_extractor import TableExtractor
from app.ingestion.table_summarizer import TableSummarizer
from app.ingestion.text_normalizer import normalize_extracted_text
from app.repositories.document_embedding_repository import DocumentEmbeddingRepository
from app.schemas.ingestion import ContentType, FileMetadata, SourceItem, SourceItemMetadata, UploadResponse

logger = logging.getLogger(__name__)


class DocumentIngestionService:
    """Coordinates parsing, chunking, multimodal normalization, metadata, and optional indexing."""

    def __init__(
        self,
        parser: Optional[DocumentParser] = None,
        chunker: Optional[DocumentChunker] = None,
        table_extractor: Optional[TableExtractor] = None,
        table_summarizer: Optional[TableSummarizer] = None,
        image_extractor: Optional[ImageExtractor] = None,
        image_captioner: Optional[ImageCaptioner] = None,
        repository: Optional[DocumentEmbeddingRepository] = None,
    ) -> None:
        self.parser = parser or DocumentParser()
        self.chunker = chunker or DocumentChunker()
        self.table_extractor = table_extractor or TableExtractor()
        self.table_summarizer = table_summarizer or TableSummarizer()
        self.image_extractor = image_extractor or ImageExtractor()
        self.image_captioner = image_captioner or ImageCaptioner()
        self.repository = repository or DocumentEmbeddingRepository()

    def ingest_pdf(self, document_id: str, pdf_path: Path, file_metadata: FileMetadata) -> UploadResponse:
        logger.info("parsing started for document_id=%s", document_id)
        parsed_pages = self.parser.parse_pdf(pdf_path)
        logger.info("parsing completed for document_id=%s pages=%s", document_id, len(parsed_pages))

        text_chunks = self.chunker.chunk_pages(parsed_pages)
        logger.info("text chunks created for document_id=%s count=%s", document_id, len(text_chunks))

        tables = self.table_extractor.extract_tables(parsed_pages)
        logger.info("tables found for document_id=%s count=%s", document_id, len(tables))

        images = self.image_extractor.extract_images(pdf_path, document_id)
        logger.info("images extracted for document_id=%s count=%s", document_id, len(images))

        items: List[SourceItem] = []
        for chunk in text_chunks:
            items.append(
                self._create_item(
                    document_id=document_id,
                    searchable_text=str(chunk["text"]),
                    content_type=ContentType.TEXT,
                    file_metadata=file_metadata,
                    page_number=chunk.get("page_number"),
                    chunk_index=chunk.get("chunk_index"),
                    extra=chunk.get("metadata", {}),
                )
            )

        for table in tables:
            summary = self.table_summarizer.summarize(table)
            items.append(
                self._create_item(
                    document_id=document_id,
                    searchable_text=summary,
                    content_type=ContentType.TABLE_SUMMARY,
                    file_metadata=file_metadata,
                    page_number=table.get("page_number"),
                    table_index=table.get("table_index"),
                    original_table_text=table.get("text"),
                    extra=table.get("metadata", {}),
                )
            )
        logger.info("tables summarized for document_id=%s count=%s", document_id, len(tables))

        for image in images:
            caption = self.image_captioner.caption_image(image)
            stored_path = self._copy_image_to_permanent_store(image, document_id)
            items.append(
                self._create_item(
                    document_id=document_id,
                    searchable_text=caption,
                    content_type=ContentType.IMAGE_CAPTION,
                    file_metadata=file_metadata,
                    page_number=image.get("page_number"),
                    image_index=image.get("image_index"),
                    image_path=str(stored_path) if stored_path else image.get("image_path"),
                    extra=image.get("metadata", {}),
                )
            )
        logger.info("images captioned for document_id=%s count=%s", document_id, len(images))

        indexed = self.repository.save_batch(items)
        logger.info("index/store completed for document_id=%s indexed=%s", document_id, indexed)

        return UploadResponse(
            success=True,
            message="PDF ingested into unified searchable items.",
            document_id=document_id,
            original_filename=file_metadata.original_filename,
            file_metadata=file_metadata,
            items_created=len(items),
            text_nodes=sum(1 for item in items if item.content_type == ContentType.TEXT),
            table_nodes=sum(1 for item in items if item.content_type == ContentType.TABLE_SUMMARY),
            image_nodes=sum(1 for item in items if item.content_type == ContentType.IMAGE_CAPTION),
            indexed=indexed,
            items=items,
            created_at=datetime.utcnow(),
        )

    def _create_item(
        self,
        document_id: str,
        searchable_text: str,
        content_type: ContentType,
        file_metadata: FileMetadata,
        page_number: Optional[int] = None,
        chunk_index: Optional[int] = None,
        table_index: Optional[int] = None,
        image_index: Optional[int] = None,
        original_table_text: Optional[str] = None,
        image_path: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> SourceItem:
        metadata = SourceItemMetadata(
            content_type=content_type,
            source=file_metadata.source_name,
            source_file=self._resolve_source_file(file_metadata),
            original_filename=file_metadata.original_filename,
            page_number=page_number,
            chunk_index=chunk_index,
            table_index=table_index,
            image_index=image_index,
            original_table_text=original_table_text,
            image_path=image_path,
            parser=settings.document_parser_provider,
            extra=extra or {},
        )
        return SourceItem(
            item_id=str(uuid.uuid4()),
            document_id=document_id,
            searchable_text=normalize_extracted_text(searchable_text),
            content_type=content_type,
            source=file_metadata.source_name,
            text_preview=normalize_extracted_text(searchable_text)[:300],
            metadata=metadata,
        )

    def _resolve_source_file(self, file_metadata: FileMetadata) -> str:
        project_root = Path(__file__).resolve().parents[3]
        documents_path = project_root / "Documents" / file_metadata.original_filename
        if documents_path.exists():
            return str(documents_path)
        return file_metadata.source_name

    def _copy_image_to_permanent_store(self, image: Dict[str, Any], document_id: str) -> Optional[Path]:
        image_path = image.get("image_path")
        if not image_path:
            return None
        source = Path(str(image_path))
        if not source.exists():
            return None
        destination_dir = settings.stored_image_dir / document_id
        destination_dir.mkdir(parents=True, exist_ok=True)
        destination = destination_dir / source.name
        shutil.copy2(source, destination)
        return destination




