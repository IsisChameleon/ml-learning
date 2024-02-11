import unittest
from dotenv import load_dotenv
from llama_index.service_context import ServiceContext
from phages.modules.postprocessor import SummaryAndScoreNodePostProcessor
from llama_index.schema import NodeWithScore, QueryBundle, TextNode

from phages.modules.schema import CustomNodeWithScore

import warnings
warnings.filterwarnings("ignore") #, category=DeprecationWarning)

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('phages.modules.postprocessor')
logger.setLevel(logging.DEBUG)

class TestSummaryAndScoreNodePostProcessor(unittest.TestCase):
    def setUp(self):
        load_dotenv("/workspaces/ml-learning/.env")
        self.service_context = ServiceContext.from_defaults()
        self.postprocessor = SummaryAndScoreNodePostProcessor(service_context=self.service_context)


    def test_class_name(self):
        self.assertEqual(self.postprocessor.class_name(), "SummaryAndScoreNodePostProcessor")

    def test_postprocess_nodes(self):
        query_bundle = QueryBundle(query_str="What is the meaning of life?")
        node = NodeWithScore(node=TextNode(id_="1", text="The meaning of life is 42."), score=1.0)
        nodes = [node]
        result = self.postprocessor._postprocess_nodes(nodes, query_bundle)
        self.assertIsInstance(result, list)
        self.assertIsInstance(result[0], CustomNodeWithScore)
        self.assertEqual(result[0].node.metadata['original_text'], self.node.node.text)  
        self.assertEqual(result[0].node.text, 'Not applicable')
        self.assertGreaterEqual(result[0].llm_score, 0.0)
        self.assertLessEqual(result[0].llm_score, 1.0)

    def test_postprocess_nodes(self):
        query_bundle = QueryBundle(query_str="What is the colour of the sky?")
        node1 = NodeWithScore(node=TextNode(id_="1", text="The meaning of life is 42."), score=0.3)
        node2 = NodeWithScore(node=TextNode(id_="2", text="""The color of the sky varies depending on the time of day and atmospheric conditions. 
                                            During a clear day, the sky appears blue due to the scattering of sunlight by the atmosphere. 
                                            This phenomenon, known as Rayleigh scattering, causes shorter blue wavelengths to be scattered more than the longer wavelengths, like red. 
                                            During sunrise and sunset, the sky can appear in shades of red and orange due to the longer path of sunlight through the atmosphere,
                                             which scatters the shorter wavelengths and allows the longer red wavelengths to dominate. 
                                            At night, the sky appears dark as there is no sunlight to scatter. 
                                            Cloudy or stormy conditions can lead to a gray or even nearly black sky, as clouds can block and scatter light in different ways."""), score=0.9)
        
        nodes = [node1, node2]
        result = self.postprocessor._postprocess_nodes(nodes, query_bundle)
        self.assertIsInstance(result, list)
        self.assertIsInstance(result[0], CustomNodeWithScore)
        self.assertIsInstance(result[1], CustomNodeWithScore)
        self.assertEqual(result[1].node.metadata['original_text'], node1.node.text)  
        self.assertEqual(result[1].node.text, 'Not applicable')
        self.assertGreaterEqual(result[1].llm_score, 0.0)
        self.assertLessEqual(result[1].llm_score, 1.0)
        self.assertEqual(result[0].node.metadata['original_text'], node2.node.text)  
        self.assertEqual(result[0].node.text, 'The color of the sky varies depending on the time of day and atmospheric conditions. During a clear day, the sky appears blue due to the scattering of sunlight by the atmosphere. This phenomenon, known as Rayleigh scattering, causes shorter blue wavelengths to be scattered more than the longer wavelengths, like red. During sunrise and sunset, the sky can appear in shades of red and orange due to the longer path of sunlight through the atmosphere, which scatters the shorter wavelengths and allows the longer red wavelengths to dominate. At night, the sky appears dark as there is no sunlight to scatter. Cloudy or stormy conditions can lead to a gray or even nearly black sky, as clouds can block and scatter light in different ways.')
        self.assertEqual(result[0].llm_score, 1.0)

if __name__ == '__main__':
    unittest.main()
