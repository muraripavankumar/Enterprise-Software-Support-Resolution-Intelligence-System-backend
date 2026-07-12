import shutil
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status

from app.core.config import ConfigurationError, settings
from app.core.auth import AuthenticatedUser, require_permissions
from app.core.logging import get_logger
from app.ingestion.document_loader import DocumentLoader
from app.schemas.ingestion import ErrorResponse, FileMetadata, UploadResponse
from app.services.document_ingestion_service import DocumentIngestionService

router = APIRouter(prefix="/ingestion", tags=["ingestion"])
logger = get_logger(__name__)


@router.post(
    "/pdf",
    summary="Ingest support PDF",
    description="Upload a PDF and index its text, tables, and images for RAG retrieval.",
    response_model=UploadResponse,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
async def ingest_pdf(
    file: UploadFile = File(...),
    _current_user: AuthenticatedUser = Depends(require_permissions("view:evaluation")),
) -> UploadResponse:
    """Ingest an uploaded support PDF into the retrieval index."""

    document_id = str(uuid.uuid4())
    loader = DocumentLoader()
    temp_pdf_path: Path | None = None
    temp_image_dir: Path | None = None

    try:
        logger.info("upload received for document_id=%s", document_id)
        loader.validate_pdf(file)
        temp_pdf_path, raw_metadata = loader.save_temp_pdf(file, document_id)
        logger.info("file saved temporarily for document_id=%s", document_id)
        file_metadata = FileMetadata(**raw_metadata)
        temp_image_dir = settings.temp_image_dir / document_id
        return DocumentIngestionService().ingest_pdf(document_id, temp_pdf_path, file_metadata)
    except UnicodeError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=ErrorResponse(
                error="ingestion_encoding_error",
                detail="Document encoding could not be processed.",
            ).model_dump(),
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=ErrorResponse(error="invalid_upload", detail="Uploaded file is invalid.").model_dump(),
        ) from exc
    except ConfigurationError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=ErrorResponse(
                error="missing_configuration",
                detail="Server ingestion configuration is incomplete.",
            ).model_dump(),
        ) from exc
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=ErrorResponse(
                error="ingestion_runtime_error",
                detail="Document ingestion failed due to a runtime error.",
            ).model_dump(),
        ) from exc
    except Exception as exc:
        logger.exception("ingestion failed for document_id=%s", document_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=ErrorResponse(
                error="ingestion_runtime_error",
                detail="Unexpected server error while ingesting document.",
            ).model_dump(),
        ) from exc
    finally:
        if temp_pdf_path and temp_pdf_path.exists():
            temp_pdf_path.unlink()
        if temp_image_dir and temp_image_dir.exists():
            shutil.rmtree(temp_image_dir, ignore_errors=True)
        try:
            file.file.close()
        except Exception:
            pass
        logger.info("cleanup completed for document_id=%s", document_id)
