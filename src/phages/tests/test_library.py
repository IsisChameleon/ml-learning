import os
from dotenv import load_dotenv
load_dotenv("/workspaces/ml-learning/.env", override=True)

import unittest
from unittest.mock import patch, MagicMock, AsyncMock
from llama_index.schema import NodeWithScore, TextNode
from phages.modules.library import Library
from phages.modules.prompts import ask_llm_prompt

class TestLibrary(unittest.TestCase):

    def setUp(self):
        # Setup code here (if needed)
        
        self.library = Library()

    # @patch('os.path.exists')
    # def test_add_local_file(self, mock_exists):
    #     # Mocking os.path.exists to always return True for local files
    #     mock_exists.return_value = True

    #     # Assuming 'local_file_path' is a valid file path
    #     local_file_path = '/path/to/local/file.txt'
    #     self.library.add(local_file_path)

    #     # Check if the document was added
    #     self.assertEqual(len(self.library.documents), 1)  # Adjust according to your implementation

    # @patch('os.makedirs')
    # @patch('requests.get')
    # @patch('os.path.exists')
    # def test_add_url(self, mock_exists, mock_requests_get, mock_makedirs):
    #     # Mocking URL validation and download process
    #     mock_exists.return_value = False
    #     mock_response = MagicMock()
    #     mock_response.iter_content.return_value = [b'file content']
    #     mock_response.raise_for_status = MagicMock()
    #     mock_requests_get.return_value = mock_response

    #     # Test with a mock URL
    #     test_url = 'http://example.com/document.pdf'
    #     self.library.add(test_url)

    #     # Check if the document was added
    #     self.assertEqual(len(self.library.documents), 1)  # Adjust according to your implementation

    # def test_invalid_source(self):
    #     with self.assertRaises(ValueError):
    #         # Test with an invalid source
    #         self.library.add('invalid_source')

    def test_get_context_str(self):
        # Create a list of NodeWithScore objects with dummy data
        node1 = NodeWithScore(node=TextNode(id_=1, text='text1', extra_info={'citation': 'citation1'}), score=0.9)
        node2 = NodeWithScore(node=TextNode(id_=2, text='text2', extra_info={'citation': 'citation2'}), score=0.8)
        nodes = [node1, node2]

        # Call the _get_context_str method
        context_str = Library._get_context_str(nodes)
        # Check the returned string
        expected_str = """1: text1

Based on citation1

2: text2

Based on citation2

Valid keys: 1, 2"""
        self.assertEqual(context_str, expected_str)

    @patch('llama_index.llms.LLM.predict')
    def test_ask_llm(self, mock_predict):
        # Mock the predict method of the LLM object
        mock_predict.return_value = 'Test answer'

        # Call the _ask_llm method with a test query
        query = 'Test query'
        result = self.library._ask_llm(query)

        # Check if the predict method was called with the correct prompt
        mock_predict.assert_called_once_with(prompt=ask_llm_prompt.partial_format(query=query))

        # Check if the _ask_llm method returned the correct result
        self.assertEqual(result, 'Test answer')


    def test_ask_llm_2(self):
        # Call the _ask_llm method with a test query
        query = 'What are the bacteriophages(s) that show promising results in aquaculture and food industries and what is the pathogen that each can control?'
        result = self.library._ask_llm(query)
        print(f'test_ask_llm_2: \n{query} \n{result}')
        self.assertEqual(1, 1)


if __name__ == '__main__':
    unittest.main()
