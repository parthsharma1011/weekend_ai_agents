from __future__ import annotations

import argparse
import sys
from pathlib import Path

from langchain_community.document_loaders import (
    DirectoryLoader,
    PyPDFLoader,
    TextLoader,
)
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import Providers, Settings


class DocumentIngestor:
    """Loads documents, splits them, embeds them, and saves a FAISS index."""

    def __init__(self, settings: Settings | None = None, providers: Providers | None = None):
        self.settings = settings or Settings()
        self.providers = providers or Providers(self.settings)

    # -- step 1: load ----------------------------------------------------------
    def _load(self, folder: str) -> list:
        docs = []
        # PDFs: PyPDFLoader splits by page and keeps the page number in metadata.
        docs.extend(
            DirectoryLoader(
                folder, glob="**/*.pdf", loader_cls=PyPDFLoader, show_progress=True
            ).load()
        )
        # Text files: one document each, filename kept in metadata.
        docs.extend(
            DirectoryLoader(
                folder,
                glob="**/*.txt",
                loader_cls=TextLoader,
                loader_kwargs={"encoding": "utf-8"},
                show_progress=True,
            ).load()
        )
        return docs

    # -- step 2: split ---------------------------------------------------------
    def _split(self, docs: list) -> list:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        return splitter.split_documents(docs)

    # -- orchestration ---------------------------------------------------------
    def ingest(self, folder: str) -> None:
        if not Path(folder).is_dir():
            print(f"[ingest] Error: '{folder}' is not a valid directory.")
            sys.exit(1)

        print(f"[ingest] Loading documents from {folder} ...")
        docs = self._load(folder)
        if not docs:
            print("[ingest] No .pdf or .txt files found. Add some to /docs.")
            sys.exit(1)

        chunks = self._split(docs)
        print(f"[ingest] Split {len(docs)} document(s) into {len(chunks)} chunks.")

        print("[ingest] Embedding chunks and building the FAISS index ...")
        store = FAISS.from_documents(chunks, self.providers.embeddings)
        store.save_local(self.settings.faiss_index_path)
        print(f"[ingest] Done. Index saved to {self.settings.faiss_index_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest documents into FAISS.")
    parser.add_argument(
        "folder",
        nargs="?",
        default="./docs",
        help="Folder containing .pdf/.txt files (default: ./docs)",
    )
    args = parser.parse_args()
    DocumentIngestor().ingest(args.folder)


if __name__ == "__main__":
    main()
