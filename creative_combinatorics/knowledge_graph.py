import os
import faiss
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

import networkx as nx
import numpy as np
from typing import Callable, Dict, Any, Set, List
from creative_combinatorics.api_client import get_embeddings

class KnowledgeGraph:
    def __init__(self):
        self.graph = nx.Graph()

    def get_all_concepts(self):
        """
        Retrieves all concepts (nodes) in the graph along with their attributes.

        Returns:
        - Generator of tuples: Each tuple is (concept, attributes) where `concept` is a string
          and `attributes` is a dictionary of the concept's attributes.
        """
        return self.graph.nodes(data=True)

    def get_all_relations(self):
        """
        Retrieves all relationships (edges) in the graph along with their attributes.

        Returns:
        - Generator of tuples: Each tuple is (source, target, attributes) where `source` and `target`
          are strings representing concepts and `attributes` is a dictionary of the relationship's attributes.
        """
        return self.graph.edges(data=True)
    
    def encode_nodes(self):
        for node in self.graph.nodes():
            print(f"node: {node}, content: {self.graph.nodes[node]}")
            self.graph.nodes[node]['embedding'] = self.calculate_concept_embeddings(node)
    
    def add_concept(self, concept: str, attributes: Dict[str, Any]):
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

    @staticmethod
    def calculate_embeddings_for_text(text: str) -> List[float]:
            """
            Calculates embeddings for the given text using OpenAI's API with a specified model.

            Args:
                text (str): The input text for which embeddings need to be calculated.

            Returns:
                List[float]: The calculated embeddings for the given text.
            """
            print ("Calculating embeddings for text: ", text)
            # Calculate embeddings using OpenAI's API
            response = client.embeddings.create(input=text,
            model="text-embedding-3-large")
            #print ("Response: ", response)
            return response.data[0].embedding
        
    
    def calculate_embeddings_for_dicts(self, concept_attributes: Dict[str, Any]) -> List[float]:
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
        concept_embeddings = KnowledgeGraph.calculate_embeddings_for_text(concept_string)

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
        # The neighbor nodes will be used to incorporate information from the edges and relationships between nodes.
        concept_attributes = self.graph.nodes[concept]
        #print(f"concept: {concept}, attributes: {concept_attributes}")
        neighbor_nodes = list(nx.single_source_shortest_path_length(self.graph, concept, cutoff=depth).keys())
        #print(f"neighbor_nodes: {neighbor_nodes}")
        neighbor_attributes = {node: self.graph.nodes[node] for node in neighbor_nodes}
        #print(f"neighbor_attributes: {neighbor_attributes}")
        concept_relationships = self.get_concept_relationships(concept)
        #print(f"concept_relationships: {concept_relationships}")

        # Add concept_relationships to the concept_attributes
        concept_attributes['relationships'] = concept_relationships
        #print(f"concept_attributes: {concept_attributes}")

        # Exclude the 'embedding' attribute from the concept and neighbor attributes
        concept_attributes.pop('embedding', None)
        for neighbor in neighbor_attributes:
            neighbor_attributes[neighbor].pop('embedding', None)

        # Add neighbor attributes to the concept attributes
        combined_attributes = {**concept_attributes, **neighbor_attributes}
        print(f"combined_attributes: {combined_attributes}")

        # Calculate embeddings for the combined attributes
        concept_embeddings = self.calculate_embeddings_for_dicts(combined_attributes)

        return concept_embeddings


    def calculate_similarity_score(self, embeddings1, embeddings2):
        # We are using the cosine similarity measure to calculate the similarity score.
        # Numerically, this is the dot product of the embeddings divided by the product of their magnitudes.
        # The result is a value between -1 and 1, where 1 indicates identical embeddings and -1 indicates opposite embeddings.
        similarity_score = np.dot(embeddings1, embeddings2) / (np.linalg.norm(embeddings1) * np.linalg.norm(embeddings2))
        return similarity_score
    
    # This function will return a subset of subgraphs from the graph that are most similar to the input concepts.
    def get_most_similar_subgraphs(self, combined_concepts: np.ndarray, n: int):
        """
        Retrieves the top n most similar concepts to the input concepts based on their embeddings.
        Parameters:
        - combined_concepts (np.ndarray): The combined embeddings of the input concepts.
        - n (int): The number of most similar concepts to retrieve.
        
        Returns:
        - list: A list of the top n most similar concepts to the input concepts.
        """
        # Computes n-nearest neighbors for the combined_concepts
        # The result is a dictionary where keys are the nearest concepts and values are the similarity scores.

        # Construct a flat index and add the combined_concepts to it
        
        print(f"combined_concepts: {combined_concepts}")
        print(f"nodes: {self.graph.nodes()}")
        print(f"graph: {self.graph}")

        nodes_subgraphs = [{'subgraph': self.get_concept_subgraph(node), 'embedding': get_embeddings(self.get_concept_subgraph(node))} for node in self.graph.nodes()]
        graph_node_embeddings = np.vstack([node['embedding'] for node in nodes_subgraphs])

        print(f"nodes_subgraphs: {nodes_subgraphs}")
        print(f"graph_node_embeddings.shape: {graph_node_embeddings.shape}")
        print(f"graph_node_embeddings: {graph_node_embeddings}")
        index = faiss.IndexFlatL2(combined_concepts.shape[0])
        index.add(graph_node_embeddings)
        # We now search for the nearest neighbors of combined_concepts in the index
        # Assuming combined_concepts is initially a 1D array
        if combined_concepts.ndim == 1:
            combined_concepts = np.array([combined_concepts])  # Reshape to 2D for FAISS

        # Proceed with the FAISS search
        S = index.search(combined_concepts, n)
        D, I = S
        print(f"I: {I}, D: {D}")
        nn_index = I[0][0]
        print(f"nn_index: {nn_index}")
        print(f"combined_concepts.shape: {combined_concepts.shape}")
        nn_vector = graph_node_embeddings[nn_index]
        nn_subgraph = nodes_subgraphs[nn_index]['subgraph']
        print(f"nn_vector: {nn_vector}")
        print(f"nn_subgraph: {nn_subgraph}")

        # Here, D is the distance and I is the index of the nearest neighbor.
        # We can use I to retrieve the nearest concepts from the knowledge graph.

        # Retrieve the nearest concepts from the knowledge graph.
        # Here, I[0] contains the indices of the nearest concepts.
        print(f"nn_vector: {nn_vector}")
        nearest_subgraphs = [nodes_subgraphs[concept] for concept in I[0]]

        # Here `nearest_concepts` is of type list, and it contains the nearest concepts to the input combined_concepts.
        # Each element of the list is a dictionary containing the attributes of the nearest concept. That's why we 
        # specify the return type of this function as List[Dict].
        return D[0][0], nearest_subgraphs


    def get_most_similar_concepts(self, concepts: List[str], n: int):
        """
        Retrieves the top n most similar concepts to the input concepts based on their embeddings.
        Parameters:
        - concepts (list): A list of concepts for which to find similar concepts.
        - n (int): The number of most similar concepts to retrieve.

        Returns:
        - dict: A dictionary containing the top n most similar KG concepts for the resulting embedded comma-separated list of input concepts and their similarity scores.
        """
        # Convert the list of concepts to a comma-separated string
        input_concepts = ','.join(concepts)

        # Embedd the input_concepts
        input_concepts_embeddings = self.get_concept_embeddings(input_concepts)

        # Computes n-nearest neighbors for the input_concepts_embeddings
        # The result is a dictionary where keys are the nearest concepts and values are the similarity scores.

        # Construct a flat index and add the input_concepts_embeddings to it
        index = faiss.IndexFlatL2(input_concepts_embeddings.shape[1])
        index.add(input_concepts_embeddings)
        D, I = index.search(input_concepts_embeddings, n)

        # Here, D is the distance and I is the index of the nearest neighbor.
        # We can use I to retrieve the nearest concepts from the knowledge graph.

        # Retrieve the nearest concepts from the knowledge graph.
        # Here, I[0] contains the indices of the nearest concepts.
        nearest_concepts = [self.graph.nodes[concept] for concept in I[0]]

        # Returns the rearest concepts and their distance
        return nearest_concepts, D[0] 

    # Return the stored nodes which are of type 'embedding' in the graph
    def get_concept_embeddings(self) -> List[np.ndarray]:
        """
        Retrieves the embeddings of all concepts in the graph.

        Returns:
        - list: A list of numpy arrays, each representing the embeddings of a concept.
        """
        return [self.graph.nodes[concept]['embedding'] for concept in self.graph.nodes() if 'embedding' in self.graph.nodes[concept]]


    def get_concept_relationships(self, concept):
        return self.graph.edges(concept, data=True)    

    def get_concept_embeddings(self, concept) -> np.ndarray:
        # Computes embeddings for the given concept (may be a list of concepts comma-separated).
        # The result is a numpy array of embeddings.
        return get_embeddings(concept)

    # Return Graph node neighbors
    def get_concept_neighbors(self, concept):
        return list(self.graph.neighbors(concept))
    
    # Return the subgraph corresponding to a node's neighborhood
    def get_concept_subgraph(self, concept, depth=1):
        subgraph = nx.single_source_shortest_path_length(self.graph, concept, cutoff=depth)
        print(f"subgraph: {subgraph}")
        return subgraph
        

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
        
