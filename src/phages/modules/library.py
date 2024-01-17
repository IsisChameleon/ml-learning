from pydantic import BaseModel
from typing import List
from llama_index.schema import Document

class Library(BaseModel):
    documents: List[Document]
