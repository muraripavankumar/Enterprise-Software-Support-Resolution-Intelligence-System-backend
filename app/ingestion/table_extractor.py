import re
from typing import Any, Dict, List

from app.ingestion.text_normalizer import normalize_extracted_text


class TableExtractor:
    """Extracts markdown tables from parsed PDF markdown/text."""

    markdown_table_pattern = re.compile(r"(^\s*\|.+\|\s*$\n?)+", re.MULTILINE)

    def extract_tables(self, pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        tables: List[Dict[str, Any]] = []
        table_index = 0
        for page in pages:
            text = str(page.get("text", ""))
            for match in self.markdown_table_pattern.finditer(text):
                table_text = normalize_extracted_text(match.group(0)).strip()
                if not table_text:
                    continue
                tables.append(
                    {
                        "text": table_text,
                        "page_number": page.get("page_number"),
                        "table_index": table_index,
                        "metadata": {"extraction_method": "markdown_regex"},
                    }
                )
                table_index += 1
        return tables



