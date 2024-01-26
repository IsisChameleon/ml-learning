
from pydantic import validator, ValidationError, BaseModel
from typing import List, Union, Any, Callable, Dict, Generator, List, Optional, Type, Sequence
from llama_index.schema import NodeWithScore

class CustomNodeWithScore(NodeWithScore):
    """A node with score that holds the semantic search score but also the LLM score."""

    llm_score: Optional[float] = None
    
    def get_score(self, raise_error: bool = False) -> float:
        """Get score."""
        if self.llm_score is None:
            if raise_error:
                raise ValueError("Score not set.")
            else:
                return 0.0
        else:
            return self.llm_score