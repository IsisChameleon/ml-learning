from pydantic import BaseModel
from pydantic import validator, ValidationError
from typing import List, Union, Any, Callable, Dict, Generator, List, Optional, Type
from typing import List
from llama_index.schema import Document
from llama_index import SimpleDirectoryReader
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

class Library(BaseModel):
    documents: List[Document] = []

    def add(self, source: Union[Path, str]) -> None:
        # Load configuration
        config = toml.load("../config.toml")
        data_dir = config["paths"]["DATA"]
        storage_dir = config["paths"]["PERSIST"]
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(storage_dir, exist_ok=True)

        file_name = self._get_file(source, data_dir)

        # Read the document using SimpleDirectoryReader
        reader = SimpleDirectoryReader(input_files=[file_name], file_extractor=CUSTOM_FILE_READER_CLS)
        new_documents = reader.load_data()
        print('New documents count:', len(new_documents))
        for doc in new_documents:
            print('calling get_citation')
            citation = get_citation(doc)
            doc.metadata['citation']=citation
        self.documents.extend(new_documents)

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

    def _extract_nodes(self, documents: List[Document]) -> List:
        pipeline = IngestionPipeline(
            transformations=[
                SentenceSplitter(chunk_size=256, chunk_overlap=64), #SentenceSplitter(chunk_size=512, chunk_overlap=128),
                # TitleExtractor(nodes=5),
                # QuestionsAnsweredExtractor(questions=3),
                SummaryExtractor(summaries=["self"]),
                KeywordExtractor(keywords=10),
            ]
        )
        nodes = pipeline.run(documents=documents)
        return nodes
