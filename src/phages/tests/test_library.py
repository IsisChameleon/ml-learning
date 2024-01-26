import unittest
from unittest.mock import patch, MagicMock
from phages.modules.library import Library
from dotenv import load_dotenv

class TestLibrary(unittest.TestCase):

    def setUp(self):
        # Setup code here (if needed)
        load_dotenv("/workspaces/ml-learning/.env")
        self.library = Library()

    @patch('os.path.exists')
    def test_add_local_file(self, mock_exists):
        # Mocking os.path.exists to always return True for local files
        mock_exists.return_value = True

        # Assuming 'local_file_path' is a valid file path
        local_file_path = '/path/to/local/file.txt'
        self.library.add(local_file_path)

        # Check if the document was added
        self.assertEqual(len(self.library.documents), 1)  # Adjust according to your implementation

    @patch('os.makedirs')
    @patch('requests.get')
    @patch('os.path.exists')
    def test_add_url(self, mock_exists, mock_requests_get, mock_makedirs):
        # Mocking URL validation and download process
        mock_exists.return_value = False
        mock_response = MagicMock()
        mock_response.iter_content.return_value = [b'file content']
        mock_response.raise_for_status = MagicMock()
        mock_requests_get.return_value = mock_response

        # Test with a mock URL
        test_url = 'http://example.com/document.pdf'
        self.library.add(test_url)

        # Check if the document was added
        self.assertEqual(len(self.library.documents), 1)  # Adjust according to your implementation

    def test_invalid_source(self):
        with self.assertRaises(ValueError):
            # Test with an invalid source
            self.library.add('invalid_source')

    # Additional tests can be added for other methods and scenarios

if __name__ == '__main__':
    unittest.main()
