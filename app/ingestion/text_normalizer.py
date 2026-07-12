import html
import re

MARKDOWN_TABLE_PATTERN = re.compile(r"(^\s*\|.+\|\s*$\n?)+", re.MULTILINE)

MOJIBAKE_REPLACEMENTS = {
    "âœ“": "✓",
    "âœ”": "✓",
    "âœ…": "✓",
    "â– ": "■",
    "â€¢": "•",
    "â€“": "-",
    "â€”": "-",
    "â€˜": "'",
    "â€™": "'",
    "â€œ": '"',
    "â€": '"',
    "â€¦": "...",
    "Â ": " ",
    "Â": "",
}


def normalize_extracted_text(text: str) -> str:
    """Clean parser output while preserving meaningful Unicode content."""
    cleaned = html.unescape(str(text))
    for bad_value, good_value in MOJIBAKE_REPLACEMENTS.items():
        cleaned = cleaned.replace(bad_value, good_value)
    return cleaned


def remove_markdown_tables(text: str) -> str:
    """Remove Markdown tables so table content is indexed through table summaries only."""
    without_tables = MARKDOWN_TABLE_PATTERN.sub("\n", str(text))
    return re.sub(r"\n{3,}", "\n\n", without_tables).strip()
