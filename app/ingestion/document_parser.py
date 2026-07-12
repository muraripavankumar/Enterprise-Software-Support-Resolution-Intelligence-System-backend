from pathlib import Path
from typing import Any, Dict, List

import fitz
from llama_parse import LlamaParse

from app.core.config import settings
from app.ingestion.text_normalizer import normalize_extracted_text


class DocumentParser:
    """Parses PDFs into markdown/text using LlamaParse, with PyMuPDF fallback only for text extraction failures."""

    def parse_pdf(self, pdf_path: Path) -> List[Dict[str, Any]]:
        if settings.document_parser_provider.lower() == "llamaparse":
            parser = LlamaParse(api_key=settings.llama_cloud_api_key, result_type="markdown")
            documents = parser.load_data(str(pdf_path))
            pages: List[Dict[str, Any]] = []
            for index, document in enumerate(documents, start=1):
                text = normalize_extracted_text(getattr(document, "text", "")).strip()
                if not text:
                    continue
                metadata = dict(getattr(document, "metadata", {}) or {})
                pages.append(
                    {
                        "text": text,
                        "page_number": metadata.get("page_number") or metadata.get("page_label") or index,
                        "metadata": {**metadata, "parser": "llamaparse"},
                    }
                )
            return pages
        return self._parse_with_pymupdf(pdf_path)

    def _parse_with_pymupdf(self, pdf_path: Path) -> List[Dict[str, Any]]:
        pages: List[Dict[str, Any]] = []
        with fitz.open(pdf_path) as document:
            for page_index, page in enumerate(document, start=1):
                text = normalize_extracted_text(page.get_text("text")).strip()
                if text:
                    pages.append(
                        {
                            "text": text,
                            "page_number": page_index,
                            "metadata": {"parser": "pymupdf", "page_count": document.page_count},
                        }
                    )
        return pages


