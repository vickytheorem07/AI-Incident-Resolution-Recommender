import time
from abc import ABC, abstractmethod
from threading import Lock
from typing import Any

from langchain_core.documents import Document
from langchain_chroma import Chroma

from app.core.logger import setup_logger


class VectorDB(ABC):
    @abstractmethod
    def add_documents(self, docs: list[Document]) -> list[str]:
        ...

    @abstractmethod
    def similarity_search(
        self,
        query: str,
        k: int = 3,
        category: str | None = None,
    ) -> list[Document]:
        ...

    @abstractmethod
    def update_document(self, doc_id: str, doc: Document) -> None:
        ...

    @abstractmethod
    def delete_document(self, doc_id: str) -> None:
        ...

    @abstractmethod
    def get_all(self) -> dict[str, Any]:
        ...


class ChromaVectorDB(VectorDB):
    _insert_lock = Lock()
    _insert_batch_size = 1000

    def __init__(self, embedding_model, collection_name: str, persist_dir: str, req_id: str = "N/A"):
        self.logger = setup_logger(__name__, req_id)
        self.collection_name = collection_name
        self.logger.info("Opening Chroma collection: collection=%s persist_dir=%s", collection_name, persist_dir)
        try:
            self.db = Chroma(
                collection_name=collection_name,
                persist_directory=persist_dir,
                embedding_function=embedding_model,
            )
        except Exception:
            self.logger.exception("Could not open Chroma collection: collection=%s", collection_name)
            raise

    def add_documents(self, docs: list[Document]) -> list[str]:
        started = time.perf_counter()
        self.logger.info("Vector db data insertion started: collection=%s new_count=%d", self.collection_name, len(docs))
        ids = [doc.id for doc in docs]
        if any(not ticket_id for ticket_id in ids):
            raise ValueError("Every vector document must have a ticket_id document id")

        inserted_ids: list[str] = []
        batch_count = (len(docs) + self._insert_batch_size - 1) // self._insert_batch_size
        try:
            for batch_index, start in enumerate(
                range(0, len(docs), self._insert_batch_size),
                start=1,
            ):
                batch_docs = docs[start : start + self._insert_batch_size]
                batch_ids = ids[start : start + self._insert_batch_size]
                batch_inserted_ids = self.db.add_documents(
                    documents=batch_docs,
                    ids=batch_ids,
                )
                inserted_ids.extend(batch_inserted_ids)
                self.logger.info(
                    "Vector insertion batch completed: collection=%s batch=%d/%d inserted=%d total_inserted=%d",
                    self.collection_name,
                    batch_index,
                    batch_count,
                    len(batch_inserted_ids),
                    len(inserted_ids),
                )
        except Exception:
            self.logger.exception(
                "Vector db data insertion failed: collection=%s inserted_before_failure=%d batch_size=%d",
                self.collection_name,
                len(inserted_ids),
                self._insert_batch_size,
            )
            raise
        self.logger.info("Vector db data insertion completed: collection=%s count=%d elapsed_ms=%.1f", self.collection_name, len(inserted_ids), (time.perf_counter() - started) * 1000)
        return inserted_ids

    def similarity_search(
        self,
        query: str,
        k: int = 3,
        category: str | None = None,
    ) -> list[Document]:
        started = time.perf_counter()
        metadata_filter = {"category": category} if category is not None else None
        self.logger.info(
            "Vector search started: collection=%s k=%d category_filter=%s",
            self.collection_name,
            k,
            category,
        )
        try:
            if metadata_filter is None:
                documents = self.db.similarity_search(query, k=k)
            else:
                documents = self.db.similarity_search(query, k=k, filter=metadata_filter)
        except Exception:
            self.logger.exception(
                "Vector search failed: collection=%s k=%d category_filter=%s",
                self.collection_name,
                k,
                category,
            )
            raise
        self.logger.info(
            "Vector search completed: collection=%s result_count=%d elapsed_ms=%.1f",
            self.collection_name,
            len(documents),
            (time.perf_counter() - started) * 1000,
        )
        return documents

    def update_document(self, doc_id: str, doc: Document) -> None:
        self.logger.info("Vector document update started: collection=%s", self.collection_name)
        try:
            self.db.update_documents(ids=[doc_id], documents=[doc])
        except Exception:
            self.logger.exception("Vector document update failed: collection=%s", self.collection_name)
            raise
        self.logger.info("Vector document update completed: collection=%s", self.collection_name)

    def delete_document(self, doc_id: str) -> None:
        self.logger.info("Vector document deletion started: collection=%s", self.collection_name)
        try:
            self.db.delete(ids=[doc_id])
        except Exception:
            self.logger.exception("Vector document deletion failed: collection=%s", self.collection_name)
            raise
        self.logger.info("Vector document deletion completed: collection=%s", self.collection_name)

    def get_all(self) -> dict[str, Any]:
        try:
            result = self.db.get()
        except Exception:
            self.logger.exception("Could not list Chroma collection: collection=%s", self.collection_name)
            raise
        self.logger.info("Chroma collection inspected: collection=%s document_count=%d", self.collection_name, len(result.get("ids", [])))
        return result
