import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Dict, Tuple

from fastapi import UploadFile

from app.core.config import settings


class DocumentLoader:
    """Validates PDF uploads and saves them to a configured temporary directory."""

    allowed_content_types = {"application/pdf", "application/octet-stream"}

    def validate_pdf(self, file: UploadFile) -> None:
        if not file.filename:
            raise ValueError("Uploaded file must include a filename.")
        if Path(file.filename).suffix.lower() != ".pdf":
            raise ValueError("Only PDF files are supported.")
        if file.content_type and file.content_type not in self.allowed_content_types:
            raise ValueError(f"Unsupported content type: {file.content_type}")

        file.file.seek(0, 2)
        size = file.file.tell()
        file.file.seek(0)

        if size <= 0:
            raise ValueError("Uploaded PDF is empty.")
        if size > settings.max_upload_bytes:
            raise ValueError(f"Uploaded PDF exceeds configured MAX_UPLOAD_MB={settings.max_upload_mb}.")

    def save_temp_pdf(self, file: UploadFile, document_id: str | None = None) -> Tuple[Path, Dict[str, object]]:
        self.validate_pdf(file)
        settings.temp_upload_dir.mkdir(parents=True, exist_ok=True)
        original_filename = Path(file.filename or "uploaded.pdf").name
        extension = Path(original_filename).suffix.lower()
        file.file.seek(0, 2)
        size = file.file.tell()
        file.file.seek(0)

        temp_name = f"{document_id or uuid.uuid4()}_{original_filename}"
        temp_handle = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=extension,
            prefix=temp_name + "_",
            dir=settings.temp_upload_dir,
        )
        temp_path = Path(temp_handle.name)
        try:
            with temp_handle:
                shutil.copyfileobj(file.file, temp_handle)
        finally:
            file.file.seek(0)

        metadata: Dict[str, object] = {
            "original_filename": original_filename,
            "file_extension": extension,
            "file_size_bytes": size,
            "temporary_path": str(temp_path),
            "source_name": original_filename,
        }
        return temp_path, metadata
