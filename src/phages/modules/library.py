from pydantic import validator, ValidationError, BaseModel
from typing import List, Union, Any, Callable, Dict, Generator, List, Optional, Type, Sequence
from llama_index.schema import Document, BaseNode, TextNode
from llama_index import ServiceContext, SimpleDirectoryReader
from pathlib import Path
from llama_index.embeddings import OpenAIEmbedding
from llama_index.text_splitter import SentenceSplitter
from llama_index import SimpleDirectoryReader
from llama_index.extractors import (
    SummaryExtractor,
    QuestionsAnsweredExtractor,
    TitleExtractor,
    KeywordExtractor,
    EntityExtractor
)
from llama_index.ingestion import IngestionPipeline
import toml
import requests
from urllib.parse import urlparse
import os
import logging
import mimetypes
import os
from datetime import datetime
from pathlib import Path

from llama_index.readers.base import BaseReader
from llama_index.readers.file.docs_reader import PDFReader
from llama_index.vector_stores import ChromaVectorStore
from llama_index.indices.vector_store import VectorStoreIndex
from llama_index.vector_stores import WeaviateVectorStore
import weaviate
from llama_index.storage.storage_context import StorageContext

from phages.modules.extractors import get_citation

class FullPDFReader(PDFReader):
    """Full PDF parser with default full document return."""

    def __init__(self) -> None:
        """
        Initialize FullPDFReader with return_full_document set to True by default.
        """
        super().__init__(return_full_document=True)

# for SimpleDirectoryReader file_extractor param
CUSTOM_FILE_READER_CLS: Dict[str, Type[BaseReader]] = {
    ".pdf": FullPDFReader(),
}


class SourceType(BaseModel):
    source: Union[Path, str]

    @validator('source', pre=True, always=True)
    def validate_source(cls, value):
        if isinstance(value, Path) or (isinstance(value, str) and os.path.exists(value)):
            return str(value)
        elif isinstance(value, str) and urlparse(value).scheme in ('http', 'https'):
            return value
        raise ValueError("The source must be a valid URL or an existing local file path.")

class Library():
    documents: List[Document] = []
    nodes: List[BaseNode] = []
    docs_index: VectorStoreIndex | None
    nodes_index: VectorStoreIndex | None

    # class Config:
    #     arbitrary_types_allowed = True

    def __init__(self):
        self._initialize_storage()

    def _initialize_storage(self):
        config = toml.load("../config.toml")
        WEAVIATE_PERSISTENCE_DATA_PATH = config["paths"]["WEAVIATE_PERSISTENCE_DATA_PATH"]
        WEAVIATE_BINARY_PATH = config["paths"]["WEAVIATE_BINARY_PATH"]
        os.makedirs(WEAVIATE_PERSISTENCE_DATA_PATH, exist_ok=True)
        os.makedirs(WEAVIATE_BINARY_PATH, exist_ok=True)

        # connect to your weaviate instance
        client = weaviate.Client(
            embedded_options=weaviate.embedded.EmbeddedOptions(
                persistence_data_path=WEAVIATE_PERSISTENCE_DATA_PATH,
                binary_path=WEAVIATE_BINARY_PATH
            ), 
            additional_headers={ 'X-OpenAI-Api-Key': os.environ["OPENAI_API_KEY"]})


        # self.chroma_client = chromadb.EphemeralClient()
        vector_store = WeaviateVectorStore(weaviate_client = client, index_name="Docs", text_key="text")
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        self.docs_index = VectorStoreIndex(
            nodes=[TextNode(text='hello world')],
            storage_context = storage_context,
            service_context=ServiceContext.from_defaults(embed_model=OpenAIEmbedding())
            )
        
        vector_store = WeaviateVectorStore(weaviate_client = client, index_name="Nodes", text_key="text")
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        self.nodes_index = VectorStoreIndex(
            nodes=[TextNode(text='hello world')],
            storage_context = storage_context,
            service_context=ServiceContext.from_defaults(embed_model=OpenAIEmbedding())
            )


    def add(self, source: Union[Path, str]) -> bool:
        # Load configuration
        config = toml.load("../config.toml")
        data_dir = config["paths"]["DATA"]
        os.makedirs(data_dir, exist_ok=True)

        file_path= self._get_file(source, data_dir)

        # Read the document using SimpleDirectoryReader
        reader = SimpleDirectoryReader(input_files=[file_path], file_extractor=CUSTOM_FILE_READER_CLS)
        new_documents = reader.load_data()
        print('New documents count:', len(new_documents))

        if (self._document_exists_in_library(file_path)):
            return False

        # Update document level metadata
        to_add_to_docs_index : List[BaseNode] = []
        for doc in new_documents:
            print('calling get_citation')
            citation = get_citation(doc)
            doc.metadata['citation']=citation
            to_add_to_docs_index.append(TextNode(text=citation, extra_info=doc.metadata))
            # setup docname?
        self.documents.extend(new_documents)
        self.docs_index.insert_nodes(to_add_to_docs_index)

        # Extract nodes : chunks and add node level metadata
        nodes = self._extract_nodes(new_documents)
        self.nodes_index.insert_nodes(nodes)
        # self.nodes.extend(nodes)

        return True
    
    def _document_exists_in_library(self, file_path):
        file_name = Path(file_path).name
        existing_file_names = [doc.metadata["file_name"] for doc in self.documents]
        if file_name in existing_file_names:
            return True
        return False

    def _get_file(self, source: Union[Path, str], data_dir: str) -> str:
        validated_source = SourceType(source=source).source  # Validate source using SourceType
        if urlparse(validated_source).scheme in ('http', 'https'):
            # Source is a URL, download the file and save it locally
            local_file_name = os.path.join(data_dir, os.path.basename(urlparse(validated_source).path))
            with requests.get(validated_source, stream=True) as r:
                r.raise_for_status()
                with open(local_file_name, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
            return local_file_name
        else:
            # Source is a local file, use it directly
            return str(validated_source)

    def _extract_nodes(self, documents: List[Document]) -> Sequence[BaseNode]:
        pipeline = IngestionPipeline(
            transformations=[
                SentenceSplitter(chunk_size=1024, chunk_overlap=256), #SentenceSplitter(chunk_size=512, chunk_overlap=128),
                # TitleExtractor(nodes=5),
                # QuestionsAnsweredExtractor(questions=3),
                # SummaryExtractor(summaries=["self"]),
                # KeywordExtractor(keywords=10),
            ]
        )
        nodes = pipeline.run(documents=documents)
        return nodes
