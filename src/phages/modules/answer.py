
from pydantic import validator, ValidationError, BaseModel
from typing import List, Union, Any, Callable, Dict, Generator, List, Optional, Type, Sequence
from llama_index.schema import NodeWithScore


class Answer(BaseModel):
    """A class to hold the answer to a question."""

    query: str
    answer: str = ""
    context: str = ""
    contexts: List[NodeWithScore] = []
    selected_documents: List[NodeWithScore] = []
    summary_length: str = "about 100 words"
    answer_length: str = "about 100 words"
    memory: Optional[str] = None

class CustomNodeWithScore(NodeWithScore):
    """A class to hold the context of a question."""

    extra_info: Dict[str, Any] = {}
