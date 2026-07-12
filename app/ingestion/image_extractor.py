from pathlib import Path
from typing import Any, Dict, List

import fitz

from app.core.config import settings


class ImageExtractor:
    """Extracts embedded images from PDFs using PyMuPDF."""

    def extract_images(self, pdf_path: Path, document_id: str) -> List[Dict[str, Any]]:
        image_dir = settings.temp_image_dir / document_id
        image_dir.mkdir(parents=True, exist_ok=True)
        images: List[Dict[str, Any]] = []
        with fitz.open(pdf_path) as document:
            for page_index, page in enumerate(document, start=1):
                for image_index, image_info in enumerate(page.get_images(full=True), start=1):
                    xref = image_info[0]
                    image_data = document.extract_image(xref)
                    extension = image_data.get("ext", "png")
                    image_path = image_dir / f"page_{page_index}_image_{image_index}.{extension}"
                    image_path.write_bytes(image_data["image"])
                    images.append(
                        {
                            "image_path": str(image_path),
                            "page_number": page_index,
                            "image_index": image_index,
                            "metadata": {"xref": xref, "extension": extension},
                        }
                    )
        return images
