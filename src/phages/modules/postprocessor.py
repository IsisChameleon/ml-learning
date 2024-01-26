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
from llama_index.schema import NodeWithScore, QueryBundle
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
    verbose: bool = Field(default=False)

    # class Config:
    #     """Configuration for this pydantic object."""

    #     arbitrary_types_allowed = True

    @classmethod
    def class_name(cls) -> str:
        return "SummaryAndScoreNodePostProcessor"

    # def _parse_prediction(self, raw_pred: str) -> str:
    #     """Parse prediction."""
    #     pred = raw_pred.strip().lower()
    #     if "previous" in pred:
    #         return "previous"
    #     elif "next" in pred:
    #         return "next"
    #     elif "none" in pred:
    #         return "none"
    #     raise ValueError(f"Invalid prediction: {raw_pred}")

    def _postprocess_nodes(
        self,
        nodes: List[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None,
    ) -> List[CustomNodeWithScore]:
        """Postprocess nodes."""
        if query_bundle is None:
            raise ValueError("Missing query bundle.")
        
        

        new_nodes=[]
        for node in nodes:
            new_node = CustomNodeWithScore(node)
            citation = node.node.metadata.get('citation', node.node.metadata.get('file_path', 'No citation available'))
            prompt = self.summary_and_score_prompt \
                .partial_format(summary_length=self.summary_length, text=node.node.get_content(), citation=citation, question=str(query_bundle))
            print('prompt:', prompt)
            chain = DefaultRefineProgram(prompt, self.service_context.llm, SummaryAndScoreOutput)
            structured_refine_answer = json.loads(chain())
            print('answer:',structured_refine_answer)
            new_nodes.append(new_node)

        return new_nodes

        # all_nodes: Dict[str, NodeWithScore] = {}
        # for node in nodes:
        #     all_nodes[node.node.node_id] = node
        #     # use response builder instead of llm directly
        #     # to be more robust to handling long context
        #     response_builder = get_response_synthesizer(
        #         service_context=self.service_context,
        #         text_qa_template=infer_prev_next_prompt,
        #         refine_template=refine_infer_prev_next_prompt,
        #         response_mode=ResponseMode.GENERATION,
        #     )
        #     raw_pred = response_builder.get_response(
        #         text_chunks=[node.node.get_content()],
        #         query_str=query_bundle.query_str,
        #     )
        #     raw_pred = cast(str, raw_pred)
        #     mode = self._parse_prediction(raw_pred)

        #     logger.debug(f"> Postprocessor Predicted mode: {mode}")
        #     if self.verbose:
        #         print(f"> Postprocessor Predicted mode: {mode}")

        #     if mode == "next":
        #         all_nodes.update(get_forward_nodes(node, self.num_nodes, self.docstore))
        #     elif mode == "previous":
        #         all_nodes.update(
        #             get_backward_nodes(node, self.num_nodes, self.docstore)
        #         )
        #     elif mode == "none":
        #         pass
        #     else:
        #         raise ValueError(f"Invalid mode: {mode}")

        # sorted_nodes = sorted(all_nodes.values(), key=lambda x: x.node.node_id)
        # return list(sorted_nodes)
