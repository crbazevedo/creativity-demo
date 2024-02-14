import openai
import networkx as nx
import numpy as np
from typing import Callable, Dict, Any, Set, List

class KnowledgeGraph:
    def __init__(self):
        self.graph = nx.Graph()

    def add_concept(self, concept: str, attributes: Dict[str, Any]):
        if 'embeddings' not in attributes:
            raise ValueError("Missing 'embeddings' attribute in concept attributes.")
        self.graph.add_node(concept, **attributes)

    def remove_concept(self, concept: str):
        self.graph.remove_node(concept)

    def add_relationship(self, concept1: str, concept2: str, relationships: Set[Callable]):
        self.graph.add_edge(concept1, concept2, relationships=relationships)

    def remove_relationship(self, concept1: str, concept2: str):
        self.graph.remove_edge(concept1, concept2)

    def deep_parse(self, concept_attributes: Dict[str, Any]) -> str:
        """
        Deeply parses the concept attributes and relationships to generate a string representation.
        """
        # Iterate through the dictionary and its nested dictionaries to generate a string representation
        # of the concept attributes and relationships
        def parse_dict(d: Dict[str, Any]) -> str:
            parsed = ""
            for k, v in d.items():
                if isinstance(v, dict):
                    parsed += parse_dict(v)
                else:
                    parsed += f"{k}: {v}, "
            return parsed

        return parse_dict(concept_attributes)

    def calculate_embeddings_for_text(self, text: str) -> np.ndarray:
        """
        Calculates embeddings for the given text using OpenAI's API with a specified model.
        """
        # Example: Calculate embeddings using OpenAI's API
        response = openai.Embedding.create(
            input=text,
            model="text-embedding-3-large"  # Updated to use the latest large embedding model
        )
        return response['data'][0]['embedding']
        
    
    def calculate_embeddings_for_dicts(self, concept_attributes: Dict[str, Any]):
        """
        Ensures that the embedding process incorporates information from the edges and relationships between nodes, 
        so that the embeddings reflect not just the properties of individual nodes but also their connections and roles within the graph.
        Parameters:
        - concept_attributes (dict): A dictionary containing the attributes and relationships for a concept.

        Returns:
        - np.ndarray: The calculated embeddings for the concept.
        """
        # Generate a string representing a deep parsing of the concept attributes
        concept_string = self.deep_parse(concept_attributes)

        # Calculate embeddings for the concept string
        concept_embeddings = calculate_embeddings_for_text(concept_string)

        return concept_embeddings

    def calculate_concept_embeddings(self, concept: str, depth: int = 1):
        """
         Ensures that the embedding process incorporates information from the edges and relationships between nodes, 
         so that the embeddings reflect not just the properties of individual nodes but also their connections and roles within the graph.
         Parameters:
            - concept (str): The concept for which to calculate embeddings.
            - depth (int): The maximum depth to consider for relationships when calculating embeddings.

        Returns:
        - np.ndarray: The calculated embeddings for the concept.
        """

        # Retrieve the concept's attributes, relationships, and neighbor nodes up to the specified depth
        concept_attributes = self.graph.nodes[concept]
        neighbor_nodes = list(nx.single_source_shortest_path_length(self.graph, concept, cutoff=depth).keys())
        neighbor_attributes = {node: self.graph.nodes[node] for node in neighbor_nodes}
        concept_relationships = self.get_concept_relationships(concept)


        # Combine the concept's attributes, relationships and neighbor attributes into a single dictionary
        combined_attributes = {concept: concept_attributes, "attributes": **neighbor_attributes, "relationships": concept_relationships}
    
        # Calculate embeddings for the combined attributes
        concept_embeddings = calculate_embeddings_for_dicts(combined_attributes)


    def calculate_similarity_score(self, embeddings1, embeddings2):
        # We are using the cosine similarity measure to calculate the similarity score.
        # Numerically, this is the dot product of the embeddings divided by the product of their magnitudes.
        # The result is a value between -1 and 1, where 1 indicates identical embeddings and -1 indicates opposite embeddings.
        similarity_score = np.dot(embeddings1, embeddings2) / (np.linalg.norm(embeddings1) * np.linalg.norm(embeddings2))
        return similarity_score
    

    def get_most_similar_concepts(self, concepts: List[str], n: int):
        """
        Retrieves the top n most similar concepts to the input concepts based on their embeddings.
        Parameters:
        - concepts (list): A list of concepts for which to find similar concepts.
        - n (int): The number of most similar concepts to retrieve.

        Returns:
        - dict: A dictionary containing the top n most similar KG concepts for the resulting embedded comma-separated list of input concepts and their similarity scores.
        """
        # Retrieve 1-hop neighbors of the input concepts
        neighbors = []

        for concept in concepts:
            neighbors.extend(list(self.graph.neighbors(concept)))
        

        # Create a dic
        input_concepts = ','.join(concepts)

        # Embedd the input_concepts
        input_concepts_embeddings = self.get_concept_embeddings(input_concepts)

        # Calculate similarity scores between the input_concepts and all other concepts in the knowledge graph




            


        
        # Sort concepts by similarity score and return the top n most similar (the closer to 1, the more similar).
        # The reverse=True argument sorts the dictionary in descending order.
        # Returns also pointers to the top n nearest concepts in the knowledge graph.
        return 

    def get_concept_relationships(self, concept):
        return self.graph.edges(concept, data=True)    

    def get_concept_embeddings(self, concept):
        # Assuming embedding data is stored as node attribute 'embedding'
        return self.graph.nodes[concept].get('embedding')

    def set_concept_embeddings(self, concept, embedding):
        self.graph.nodes[concept]['embedding'] = embedding

    def query_concepts(self, criteria, relationship=lambda a, b: a == b):
        """
        Queries the knowledge graph for concepts that match the given criteria based on a specified relationship.
        Parameters:
        - criteria (dict): A dictionary with node attributes and their desired values for comparison.
        - relationship (callable): A function that takes two arguments (attribute value, criterion value)
                                and returns True if the relationship holds.

        Returns:
        - list: A list of concepts (nodes) that match the criteria based on the relationship.
        """
        filtered_nodes = []
        for node, attributes in self.graph.nodes(data=True):
            if all(relationship(attributes.get(key), value) for key, value in criteria.items()):
                filtered_nodes.append(node)
        return filtered_nodes
    
    def is_transitive(self, relationship: Callable[[Any, Any], bool]):
        for a in self.graph.nodes():
            for b in self.graph.successors(a):
                for c in self.graph.successors(b):
                    if relationship(a, b) and relationship(b, c) and not relationship(a, c):
                        return False
        return True
    
    def get_graph(self):
        return self.graph
    
    def set_graph(self, graph):
        self.graph = graph
        
