import networkx as nx
from typing import Callable, Dict, Any, Set

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
        
