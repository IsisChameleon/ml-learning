import unittest
import os
from dotenv import load_dotenv
from llama_index.service_context import ServiceContext
from phages.modules.postprocessor import SummaryAndScoreNodePostProcessor
from llama_index.schema import NodeWithScore, QueryBundle
from llama_index.bridge.pydantic import Document

class TestSummaryAndScoreNodePostProcessor(unittest.TestCase):
    def setUp(self):
        load_dotenv("/workspaces/ml-learning/.env")
        self.service_context = ServiceContext.from_defaults()
        self.postprocessor = SummaryAndScoreNodePostProcessor(service_context=self.service_context)
        self.query_bundle = QueryBundle(query_str="What is the meaning of life?")
        self.node = NodeWithScore(node=Document(node_id="1", content="The meaning of life is 42."), score=1.0)
        self.nodes = [self.node]

    def test_class_name(self):
        self.assertEqual(self.postprocessor.class_name(), "SummaryAndScoreNodePostProcessor")

    def test_postprocess_nodes(self):
        result = self.postprocessor._postprocess_nodes(self.nodes, self.query_bundle)
        self.assertIsInstance(result, list)
        self.assertIsInstance(result[0], CustomNodeWithScore)
        self.assertEqual(result[0].node.node_id, "1")
        self.assertEqual(result[0].score, 1.0)

if __name__ == '__main__':
    unittest.main()
