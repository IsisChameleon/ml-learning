import unittest
from llama_index.service_context import ServiceContext
from phages.modules.postprocessor import SummaryAndScoreNodePostProcessor

class TestSummaryAndScoreNodePostProcessor(unittest.TestCase):
    def setUp(self):
        self.service_context = ServiceContext.from_defaults()
        self.postprocessor = SummaryAndScoreNodePostProcessor(service_context=self.service_context)

    def test_class_name(self):
        self.assertEqual(self.postprocessor.class_name(), "SummaryAndScoreNodePostProcessor")

    # Add more tests as needed

if __name__ == '__main__':
    unittest.main()
