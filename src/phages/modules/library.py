from pydantic import BaseModel
from typing import List, Union, Any
from pydantic import FilePath, HttpUrl
from typing import List
from llama_index.schema import Document
from llama_index import SimpleDirectoryReader
from pathlib import Path
from llama_index.embeddings import OpenAIEmbedding
from llama_index.text_splitter import SentenceSplitter
from llama_index.extractors import TitleExtractor
from llama_index.ingestion import IngestionPipeline, IngestionCache
from llama_index import SimpleDirectoryReader
import toml
import os
import requests
from shutil import copy2
from urllib.parse import urlparse
import os

class SourceType(str, Enum):
    LOCAL_FILE = "local_file"
    URL = "url"

    @classmethod
    def get_source_type(cls, value: str) -> 'SourceType':
        if os.path.exists(value):
            return cls.LOCAL_FILE
        elif urlparse(value).scheme in ('http', 'https'):
            return cls.URL
        else:
            raise ValueError(f"Invalid source type for value: {value}")
from llama_index import SimpleDirectoryReader

class Library(BaseModel):
    documents: List[Document]

    def add(self, source: Union[Path, str]) -> None:
        # Load configuration
        config = toml.load("config.toml")  # This assumes that 'config.toml' is in the current working directory
        data_dir = config["paths"]["DATA"]
        storage_dir = config["paths"]["PERSIST"]
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(storage_dir, exist_ok=True)

        try:
            # Determine if the source is a URL or a local file
            source_type = SourceType.get_source_type(str(source))
            if source_type == SourceType.URL:
                # Source is a URL, download the file and save it locally
                local_file_name = os.path.join(storage_dir, os.path.basename(urlparse(source).path))
                with requests.get(source, stream=True) as r:
                    r.raise_for_status()
                    with open(local_file_name, 'wb') as f:
                        for chunk in r.iter_content(chunk_size=8192):
                            f.write(chunk)
                file_name = local_file_name
            else:
            elif source_type == SourceType.LOCAL_FILE:
                file_name = str(source)  # Use the local file path directly

            # Read the document using SimpleDirectoryReader
            reader = SimpleDirectoryReader(input_files=[file_name])
            new_documents = reader.load_data()
            self.documents.extend(new_documents)
        except (ValueError, Exception) as e:
            print(f"Error handling the document source: {e}")

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
