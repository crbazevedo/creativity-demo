"""
The `CreativeTextGenerator` class orchestrates generating creative text using GPT-4 integrated with a knowledge graph. 
It includes methods for query parsing, semantic analysis using embeddings, generating concept combinations, 
ranking these combinations, and crafting prompts for GPT-4. This workflow aims to leverage semantic relationships and 
combinatorics to inject creativity into text generation. Usage involves initializing the class, passing a query to parse, 
and calling a method to generate creative text, which integrates knowledge graph insights into GPT-4 prompts for enriched, 
context-aware outputs.
"""

from typing import List, Dict
import creative_combinatorics.api_client as api_client
import random, requests
import numpy as np
from creative_combinatorics.knowledge_graph import KnowledgeGraph
from creative_combinatorics.api_client import get_embeddings, generate_text

class CreativeTextGenerator:
    def __init__(self, knowledge_graph: KnowledgeGraph):
        self.knowledge_graph = knowledge_graph

    def parse_query(text):
        api_key = api_client.configure_api_key()
        endpoint = 'https://api.openai.com/v1/completions'
        
        headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json',
        }
        
        # Update the prompt to instruct GPT-4 to analyze the query for concepts, relationships, and tasks
        prompt = f"Given the query: '{text}', identify the concepts, relationships, and tasks. Return the analysis in JSON format with lists of values."
        
        data = {
            'model': 'gpt-4-turbo', 
            'prompt': prompt,
            'max_tokens': 4000,
            'temperature': 0.1,
            'top_p': 1.0,
            'frequency_penalty': 0.0,
            'presence_penalty': 0.0,
        }
        
        response = requests.post(endpoint, headers=headers, json=data)
        
        if response.status_code == 200:
            response_data = response.json()
            # Process the JSON response to extract concepts, relationships, and tasks
            # This step will depend on how GPT-4 structures its response based on your prompt
            extracted_info = response_data  # Placeholder for actual processing logic
            return extracted_info
        else:
            return f"Error: {response.text}"


    def calculate_similarity_score(self, embeddings1: List[float], embeddings2: List[float]) -> float:
        """
        Calculates the similarity score between two sets of embeddings.

        Parameters:
        - embeddings1 (list): The first set of embeddings.
        - embeddings2 (list): The second set of embeddings.

        Returns:
        - float: The similarity score between the two sets of embeddings.
        """
        # Example similarity calculation
        similarity_score = np.dot(embeddings1, embeddings2) / (np.linalg.norm(embeddings1) * np.linalg.norm(embeddings2))
        return similarity_score

    def generate_concept_combinations(self, parsed_query: Dict) -> List[List[str]]:
        """
        Generates combinations of concepts based on the parsed query.

        Parameters:
        - parsed_query (dict): The parsed query containing concepts and relationships.

        Returns:
        - list: A list of concept combinations.
        """
       
        # Find extracted concepts and relationships from the parsed query
        concepts = parsed_query.get('concepts', [])
        relationships = parsed_query.get('relationships', [])

        # Generate concept combinations based on relationships
        concept_combinations = []

        for concept1 in concepts:
            for concept2 in concepts:
                if concept1 != concept2:
                    concept_combinations.append([concept1, concept2])

        return concept_combinations

    def rank_concept_combinations(self, concept_combinations: List[List[str]]) -> List[List[str]]:
        """
        Ranks concept combinations based on semantic similarity and knowledge graph insights.

        Parameters:
        - concept_combinations (list): A list of concept combinations.

        Returns:
        - list: A list of ranked concept combinations.
        """
        # Example ranking logic
        ranked_combinations = sorted(concept_combinations, key=lambda x: random.random(), reverse=True)
        return ranked_combinations

    def craft_gpt_prompt(self, concept_combination: List[str]) -> str:
        """
        Crafts a GPT-4 prompt based on a concept combination and knowledge graph insights.

        Parameters:
        - concept_combination (list): A combination of concepts.

        Returns:
        str: A crafted prompt for GPT-4 incorporating the selected concept combination and insights from the knowledge graph.
        """

        prompt = "Write a story about " + " and ".join(concept_combination) + ", exploring their unique relationship."
        return prompt
    
    def generate_creative_text(self, query: str) -> str:
        """
        Generates creative text based on the input query by leveraging a knowledge graph and GPT-4.

        Parameters:
        - query (str): The input query string.

        Returns:
        - str: The generated creative text.
        """
        parsed_query = self.parse_query(query)
        concept_combinations = self.generate_concept_combinations(parsed_query)
        ranked_combinations = self.rank_concept_combinations(concept_combinations)
        for combination in ranked_combinations:
            prompt = self.craft_gpt_prompt(combination)
            creative_text = generate_text(prompt)
            return creative_text
        return "No creative text generated."