import mimetypes
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Generator, List, Optional, Type
from llama_index import Document
from llama_index.llms import OpenAI
from llama_index.schema import MetadataMode

from phages.modules.prompts import citation_prompt

def get_citation(doc: Document, chunk_size=3000)->str:

    current_year = datetime.now().year
    cit = citation_prompt.partial_format(year=current_year)
    print('citation prompt:', cit)
    llm = OpenAI(model="gpt-3.5-turbo", temperature=0)

    text = doc.get_content(metadata_mode=MetadataMode.NONE)[:3000]
    print('text:', text)

    citation = llm.predict(cit, text = text)
    print('after llm call citation=', citation)
    return citation

# class KeywordExtractor(BaseExtractor):
#     """Citation extractor. Document-level extractor. Extract citation from first page of document.

#     Args:
#         llm (Optional[LLM]): LLM
#     """

#     llm: LLMPredictorType = Field(description="The LLM to use for generation.")

#     def __init__(
#         self,
#         llm: Optional[LLM] = None,
#         num_workers: int = DEFAULT_NUM_WORKERS,
#         **kwargs: Any,
#     ) -> None:
#         """Init params."""

#         super().__init__(
#             llm=llm or resolve_llm("default"),
#             num_workers=num_workers,
#             **kwargs,
#         )

#     @classmethod
#     def class_name(cls) -> str:
#         return "DocumentCitationExtractor"

#     async def _aextract_keywords_from_node(self, node: BaseNode) -> Dict[str, str]:
#         """Extract keywords from a node and return it's metadata dict."""
#         if self.is_text_node_only and not isinstance(node, TextNode):
#             return {}

#         current_year = datetime.now().year
#         citation = await self.llm.apredict(
#             citation_prompt.partial_format(year=current_year),
#             text=cast(TextNode, node).text,
#         )

#         return {"citation": citation.strip()}

#     async def aextract(self, nodes: Sequence[BaseNode]) -> List[Dict]:
#         citation_jobs = []
#         for node in nodes:
#             citation_jobs.append(self._aextract_citation_from_node(node))

#         metadata_list: List[Dict] = await run_jobs(
#             citation_jobs, show_progress=self.show_progress, workers=self.num_workers
#         )

#         return metadata_list
    
#     async def aextract_from_documents(self, document: Document) -> Dict:
