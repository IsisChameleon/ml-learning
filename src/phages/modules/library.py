from pydantic import BaseModel
from typing import List, Union
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
from llama_index import SimpleDirectoryReader

class Library(BaseModel):
    documents: List[Document]

    def add(self, source: Union[Path, str]) -> None:
        # Load configuration
        config = toml.load("config.toml")
        data_dir = config["paths"]["DATA"]
        storage_dir = config["paths"]["PERSIST"]
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(storage_dir, exist_ok=True)

        try:
            file_name = None
            if isinstance(source, str):
                # Assume source is a URL
                file_name = os.path.join(data_dir, os.path.basename(urlparse(source).path))
            else:
                # Download the file and save it locally
                with requests.get(source, stream=True) as r:
                    r.raise_for_status()
                    with open(file_name, 'wb') as f:
                        for chunk in r.iter_content(chunk_size=8192):
                            f.write(chunk)
            elif isinstance(source, Path):
                # If source is a relative Path, copy to the data directory
                file_name = os.path.join(data_dir, source.name)
                if not source.is_absolute():
                    copy2(source, file_name)
                else:
                    # If source is an absolute Path, use it as is
                    file_name = str(source)

            # Read the document using SimpleDirectoryReader
            reader = SimpleDirectoryReader(input_files=[file_name])
            new_documents = reader.load_data()
            self.documents.extend(new_documents)
        except Exception as e:
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
