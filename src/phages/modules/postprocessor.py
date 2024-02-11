"""Node postprocessor."""

import json
import logging
from typing import Dict, List, Optional
from pydantic import BaseModel

from llama_index.bridge.pydantic import Field, validator
from llama_index.postprocessor.types import BaseNodePostprocessor
from llama_index.prompts.base import PromptTemplate
from llama_index.response_synthesizers import ResponseMode, get_response_synthesizer
from llama_index.response_synthesizers.refine import DefaultRefineProgram
from llama_index.schema import NodeWithScore, QueryBundle, MetadataMode
from llama_index.service_context import ServiceContext

from phages.modules.prompts import summary_prompt
from phages.modules.schema import CustomNodeWithScore

logger = logging.getLogger(__name__)


class SummaryAndScoreOutput(BaseModel):
    """Output of the summary and score node postprocessor.
    """
    summary: str
    llm_score: float = Field(default=0.0)
  
class SummaryAndScoreNodePostProcessor(BaseNodePostprocessor):
    """LLM relevancy score of each node
    """
    service_context: ServiceContext
    summary_and_score_prompt = summary_prompt
    summary_length: str = "about 250 words"

    @classmethod
    def class_name(cls) -> str:
        return "SummaryAndScoreNodePostProcessor"

    def _postprocess_nodes(
        self,
        nodes: List[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None,
    ) -> List[CustomNodeWithScore]:
        """Postprocess nodes."""
        if query_bundle is None:
            raise ValueError("SummaryAndScoreNodePostProcessor : query is empty.")

        new_nodes=[]
        for node in nodes:
            new_node = CustomNodeWithScore(node)
            logger.debug(f"\nPostprocessing node {node.node.id_}:{node.node.text} into new_node {new_node.node.id_}:{new_node.node.text}")
            citation = node.node.metadata.get('citation', node.node.metadata.get('file_path', 'No citation available'))
            prompt_tpl = self.summary_and_score_prompt \
                .partial_format(summary_length=self.summary_length, text=node.node.get_content(), citation=citation, question=str(query_bundle))
            logger.debug(f"\nPrompt {prompt_tpl}")

            answer = self.service_context.llm.structured_predict(output_cls=SummaryAndScoreOutput, prompt=prompt_tpl)

            new_node.llm_score = answer.llm_score
            new_node.node.metadata['original_text'] = node.node.text
            new_node.node.text = answer.summary
            new_nodes.append(new_node)
            logger.debug(f"\n\nNode {new_node.node.metadata['original_text']}, {new_node.node.get_content(metadata_mode=MetadataMode.NONE)}, {new_node.llm_score}\n")

        sorted_nodes = sorted(new_nodes, key=lambda x: x.llm_score, reverse=True)

        return list(sorted_nodes)
