from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ContentType(str, Enum):
    TEXT = "text"
    TABLE_SUMMARY = "table_summary"
    IMAGE_CAPTION = "image_caption"


class FileMetadata(BaseModel):
    original_filename: str
    file_extension: str
    file_size_bytes: int
    temporary_path: str
    source_name: str


class SourceItemMetadata(BaseModel):
    content_type: ContentType
    source: str
    source_file: str
    original_filename: str
    page_number: Optional[int] = None
    chunk_index: Optional[int] = None
    table_index: Optional[int] = None
    image_index: Optional[int] = None
    original_table_text: Optional[str] = None
    image_path: Optional[str] = None
    parser: Optional[str] = None
    extra: Dict[str, Any] = Field(default_factory=dict)


class SourceItem(BaseModel):
    item_id: str
    document_id: str
    searchable_text: str
    content_type: ContentType
    source: str
    text_preview: str
    metadata: SourceItemMetadata


class UploadResponse(BaseModel):
    success: bool
    message: str
    document_id: str
    original_filename: str
    file_metadata: FileMetadata
    items_created: int
    text_nodes: int
    table_nodes: int
    image_nodes: int
    indexed: bool
    items: List[SourceItem]
    created_at: datetime


class ErrorResponse(BaseModel):
    error: str
    detail: str


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=3)
    top_k: int = Field(default=5, ge=1, le=20)
    filters: Dict[str, Any] = Field(default_factory=dict)


class QuerySource(BaseModel):
    item_id: str
    document_id: str
    score: Optional[float] = None
    text_preview: str
    metadata: SourceItemMetadata


class QueryResponse(BaseModel):
    query: str
    answer: Optional[str] = None
    sources: List[QuerySource]
    total_sources: int
