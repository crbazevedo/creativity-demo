import unittest
from creative_combinatorics.knowledge_graph import calculate_embeddings_for_text

class TestKnowledgeGraph(unittest.TestCase):
    def test_calculate_embeddings_for_text(self):
        text = "Hello, world!"
        expected_embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
        
        # Mock the openai.Embedding.create method
        class MockEmbedding:
            @staticmethod
            def create(input, model):
                return {'data': [{'embedding': expected_embedding}]}
        
        # Replace the original openai.Embedding with the mock
        original_openai_embedding = openai.Embedding
        openai.Embedding = MockEmbedding
        
        # Call the function and assert the result
        result = calculate_embeddings_for_text(text)
        self.assertEqual(result, expected_embedding)
        
        # Restore the original openai.Embedding
        openai.Embedding = original_openai_embedding

if __name__ == '__main__':
    unittest.main()