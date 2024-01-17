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
from llama_index import SimpleDirectoryReader
import toml
import os
import requests

class Library(BaseModel):
    documents: List[Document]

    def add(self, source: Union[Path, str]) -> None:
        # Load configuration
        config = toml.load("config.toml")
        data_dir = config["paths"]["DATA"]
        os.makedirs(data_dir, exist_ok=True)

        try:
            if isinstance(source, Path):
                document = load_document_from_path(source)
            else:
                # Download the file and save it locally
                response = requests.get(source)
                response.raise_for_status()
                file_name = os.path.join(data_dir, os.path.basename(source))
                with open(file_name, 'wb') as f:
                    f.write(response.content)
                document = load_document_from_path(Path(file_name))

                # Read documents using SimpleDirectoryReader
                documents = SimpleDirectoryReader(data_dir).load_data()
                self.documents.extend(documents)

            self.documents.append(document)
        except LoaderError as e:
            print(f"Error loading document: {e}")
        except requests.RequestException as e:
            print(f"Error downloading document: {e}")

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
