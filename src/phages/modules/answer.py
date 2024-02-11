
from pydantic import validator, ValidationError, BaseModel
from typing import List, Union, Any, Callable, Dict, Generator, List, Optional, Type, Sequence
from llama_index.schema import NodeWithScore
from phages.modules.schema import CustomNodeWithScore


class Answer(BaseModel):
    """A class to hold the answer to a question."""

    query: str
    answer: str = ""
    context: str = ""
    contexts: List[CustomNodeWithScore] = []
    selected_documents: List[NodeWithScore] = []
    summary_length: str = "about 250 words"
    answer_length: str = "about 250 words"
    memory: Optional[str] = None
