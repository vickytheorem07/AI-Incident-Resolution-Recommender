import csv
from io import BytesIO, StringIO
from pathlib import Path
from time import perf_counter

from fastapi import HTTPException, UploadFile, status

from app.core.config import settings
from app.core.logger import setup_logger
from app.services.loaders import DocumentLoader, REQUIRED_TICKET_COLUMNS
from app.services.embedding import Hugging_face_embedding
from app.services.vector_db import ChromaVectorDB


async def load_docs(req_id: str, file: UploadFile):
    logger = setup_logger(__name__, req_id)
    request_started = perf_counter()
    response_status = status.HTTP_500_INTERNAL_SERVER_ERROR
    document_count = 0
    log_filename = Path(file.filename or "unnamed").name.replace("\r", "").replace("\n", "")
    logger.info("/load_doc request started: filename=%s", log_filename)

    try:
        # Validate file extension
        if not file.filename or not file.filename.lower().endswith(".csv"):
            logger.warning("CSV upload rejected: reason=invalid_extension filename=%s", log_filename)
            raise HTTPException(status_code=400, detail="Please upload a CSV file.")

        # Read no more than the configured maximum plus one byte.
        max_upload_bytes = settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024
        contents = await file.read(max_upload_bytes + 1)
        if len(contents) > max_upload_bytes:
            logger.warning(
                "Document upload rejected: size exceeds limit, filename=%s limit_mb=%d",
                log_filename,
                settings.MAX_UPLOAD_SIZE_MB,
            )
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"Document exceeds the {settings.MAX_UPLOAD_SIZE_MB} MB upload limit",
            )
        logger.info("CSV upload received: filename=%s size_bytes=%d", log_filename, len(contents))
        
        try:
            header = next(csv.reader(StringIO(contents.decode("utf-8-sig"))), [])
        except (UnicodeDecodeError, csv.Error) as exc:
            logger.warning("CSV upload rejected: reason=invalid_csv_encoding filename=%s", log_filename)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="CSV must be a valid UTF-8 CSV file.",
            ) from exc

        missing_columns = [
            column for column in REQUIRED_TICKET_COLUMNS if column not in header
        ]
        if missing_columns:
            logger.warning(
                "CSV upload rejected: reason=missing_required_columns filename=%s missing_columns=%s",
                log_filename,
                ",".join(missing_columns),
            )
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"CSV is missing required columns: {', '.join(missing_columns)}",
            )

        # Each CSV data row becomes one LangChain Document and one vector.
        loader = DocumentLoader(req_id=req_id)
        try:
            documents = loader.load_csv_rows(BytesIO(contents), filename=log_filename)
        except (ValueError, csv.Error, UnicodeDecodeError) as exc:
            logger.warning(
                "CSV upload rejected during row parsing: filename=%s error_type=%s",
                log_filename,
                type(exc).__name__,
            )
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(exc),
            ) from exc
        if not documents:
            logger.warning("CSV upload rejected: reason=no_ticket_rows filename=%s", log_filename)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="CSV contains no ticket rows.",
            )
        document_count = len(documents)
        logger.info(
            "CSV validation completed: filename=%s document_count=%d",
            log_filename,
            document_count,
        )

        logger.info(
            "Embedding model initialization started: filename=%s model=%s",
            log_filename,
            settings.huggingface_embedding_model_name,
        )
        embedding = Hugging_face_embedding(
            settings.huggingface_embedding_model_name,
            settings.huggingface_api,
            req_id=req_id,
        )
        vector_db = ChromaVectorDB(
            embedding.embedding,
            settings.collection_name,
            settings.persist_dir,
            req_id=req_id,
        )
        logger.info(
            "Vector insertion started: filename=%s document_count=%d collection=%s",
            log_filename,
            document_count,
            settings.collection_name,
        )
        inserted_ids = vector_db.add_documents(documents)
        logger.info(
            "Vector insertion completed: filename=%s inserted_count=%d collection=%s",
            log_filename,
            len(inserted_ids),
            settings.collection_name,
        )

        response_status = status.HTTP_201_CREATED
        return {
            "status": "success",
            "filename": file.filename,
            "message": "CSV tickets loaded and stored in vector db successfully.",
            "chunks": len(inserted_ids),
        }

    except HTTPException as exc:
        response_status = exc.status_code
        logger.warning(
            "CSV request finished with HTTP error: filename=%s status_code=%d",
            log_filename,
            response_status,
        )
        raise
    except Exception:
        logger.exception("csv processing failed: filename=%s", log_filename)
        response_status = status.HTTP_500_INTERNAL_SERVER_ERROR
        raise HTTPException(
            status_code=500,
            detail="Error processing document"
        )

    finally:
        try:
            await file.close()
        finally:
            logger.info(
                "/load_doc request ended: status_code=%d document_count=%d total_time_seconds=%.2f",
                response_status,
                document_count,
                perf_counter() - request_started,
            )
