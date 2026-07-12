from typing import Any, Dict

from openai import AzureOpenAI, OpenAIError

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class TableSummarizer:
    """Summarizes markdown tables into concise natural-language text."""

    def __init__(self) -> None:
        self.client = AzureOpenAI(
            api_key=settings.azure_openai_api_key,
            azure_endpoint=settings.azure_openai_endpoint,
            api_version=settings.azure_openai_api_version,
        )

    def summarize(self, table: Dict[str, Any]) -> str:
        table_text = str(table.get("text", "")).strip()
        page_number = table.get("page_number")
        try:
            response = self.client.chat.completions.create(
                model=settings.azure_openai_llm_deployment,
                messages=[
                    {
                        "role": "system",
                        "content": "Summarize support-documentation tables for search retrieval. Do not invent facts.",
                    },
                    {
                        "role": "user",
                        "content": f"Page: {page_number}\nMarkdown table:\n{table_text}",
                    },
                ],
                temperature=0,
                max_tokens=180,
            )
            return response.choices[0].message.content or self._fallback(table_text, page_number)
        except OpenAIError as exc:
            logger.warning("Table summarization failed; using fallback summary: %s", exc)
            return self._fallback(table_text, page_number)

    def _fallback(self, table_text: str, page_number: object) -> str:
        preview = table_text.replace("\n", " | ")[:1200]
        return f"Table summary from page {page_number}: {preview}"
