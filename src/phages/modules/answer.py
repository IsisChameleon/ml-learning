
from pydantic import validator, ValidationError, BaseModel
from typing import List, Union, Any, Callable, Dict, Generator, List, Optional, Type, Sequence
from llama_index.schema import NodeWithScore


class Answer(BaseModel):
    """A class to hold the answer to a question."""

    question: str
    answer: str = ""
    context: str = ""
    contexts: List[Context] = []
    references: str = ""
    formatted_answer: str = ""
    dockey_filter: Optional[Set[DocKey]] = None
    summary_length: str = "about 100 words"
    answer_length: str = "about 100 words"
    memory: Optional[str] = None

class CustomNodeWithScore(NodeWithScore):
    """A class to hold the context of a question."""

    raise NotImplementedError
