from pydantic import BaseModel
from typing import List
from llama_index.schema import Document
from llama_index.loaders import load_document_from_path, load_document_from_url
from pathlib import Path
from llama_index.embeddings import OpenAIEmbedding
from llama_index.text_splitter import SentenceSplitter
from llama_index.extractors import TitleExtractor
from llama_index.ingestion import IngestionPipeline, IngestionCache
from llama_index.exceptions import LoaderError

class Library(BaseModel):
    documents: List[Document]

    def add(self, source: Union[Path, str]) -> None:
        try:
            if isinstance(source, Path):
                document = load_document_from_path(source)
            else:
                document = load_document_from_url(source)
            self.documents.append(document)
        except LoaderError as e:
            print(f"Error loading document: {e}")

    def _extract_nodes(self, documents: List[Document]) -> List:
        pipeline = IngestionPipeline(
            transformations=[
                SentenceSplitter(chunk_size=25, chunk_overlap=0),
                TitleExtractor(),
                OpenAIEmbedding(),
            ]
        )
        nodes = pipeline.run(documents=documents)
        return nodes
