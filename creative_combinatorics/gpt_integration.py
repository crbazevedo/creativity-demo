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
from rdflib import Graph, URIRef, Literal, Namespace, BNode
from rdflib.namespace import RDF, XSD
from urllib.parse import urlparse
from creative_combinatorics.knowledge_graph import KnowledgeGraph
from creative_combinatorics.api_client import get_embeddings, generate_text

class CreativeTextGenerator:
    def __init__(self, knowledge_graph: KnowledgeGraph):
        self.knowledge_graph = knowledge_graph

    def parse_query(self, text):
        api_key = api_client.configure_api_key()
        endpoint = 'https://api.openai.com/v1/completions'
        
        headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json',
        }
        
        # Update the prompt to instruct GPT-4 to analyze the query for concepts, relationships, and tasks
        prompt = f"Given the query: '{text}', identify the 'concepts', 'relationships', and 'task'. Assume the relationships are formatted as a triple string `(concept1, relation, concept2)`. "+\
                  "So, for each pair of concepts, assume either no relationship or generate at least one triple. Return the analysis in JSON format with lists of triples for each attribute."
        
        data = {
            'model': 'gpt-4-turbo', 
            'prompt': prompt,
            'max_tokens': 50000,
            'temperature': 0.1,
            'top_p': 1.0,
            'frequency_penalty': 0.0,
            'presence_penalty': 0.0,
        }
        
        response = requests.post(endpoint, headers=headers, json=data)
        
        if response.status_code == 200:
            response_data = response.json()
            # Process the JSON response to extract triples of concepts and relationships.
            # Extracted information should be in the format: {'triples': ['(concept1,relation,concept2)',...]}
            # Try-catch block that checks that it containts all exepcted keys
            try:
                # response_data contains the parsed query in JSON format, so we can extract the concepts, relationships, and task
                extracted_info = {
                    # We get a list of triples in the format `(concept1, relation, concept2)`. So, we need to split each triple into its components.
                    # The concepts correspond to the first and third elements of each triple, and the relationships correspond to the second element.
                    # First, we get the list of triples from the response data.
                    'triples': [triple['text'] for triple in response_data['choices'][0]['logprobs']['top_level']]
                }
            except KeyError:
                return f"Error: {response.text}"
            return extracted_info
        else:
            return f"Error: {response.text}"

    def build_knowledge_graph_from_triples(triples: List[str]) -> KnowledgeGraph:
        # Initialize the knowledge graph
        kg = KnowledgeGraph()

        # Unique concepts set to avoid duplicate calculations
        unique_concepts = set()

        # Extract concepts from triples and add them to the unique concepts set
        concepts = [concept.strip() for triple in triples for concept in (triple.split(',')[0], triple.split(',')[2])]
        unique_concepts.update(concepts)

        # Calculate embeddings and add nodes to the knowledge graph
        for concept in unique_concepts:
            embedding = KnowledgeGraph.calculate_embeddings_for_text(concept)
            kg.add_node(concept, embedding)

        # Add edges based on relationships
        for triple in triples:
            concept1, relation, concept2 = [element.strip() for element in triple.split(',')]
            kg.add_edge(concept1, relation, concept2)

        return kg

    def save_knowledge_graph_as_turtle(kg: KnowledgeGraph, file_path: str):
        g = Graph()
        
        # Define your namespaces
        MY_NS = Namespace("http://example.org/my_knowledge_graph#")
        
        for concept, attrs in kg.get_all_concepts():
            concept_uri = URIRef(MY_NS[concept])
            # Here, add the concept node to the RDF graph with a generic type or more specific based on your class's structure
            g.add((concept_uri, RDF.type, MY_NS.Concept))
            
            # Serialize embeddings as a single attribute, if present
        if 'embedding' in attrs:
            # Convert the embedding list to a string representation
            embedding_str = ','.join(map(str, attrs['embedding']))
            g.add((concept_uri, MY_NS.embedding, Literal(embedding_str, datatype=XSD.string)))
                
        for source, relation_type, target, _ in kg.get_all_relations():
            source_uri = URIRef(MY_NS[source])
            target_uri = URIRef(MY_NS[target])
            relation_uri = MY_NS[relation_type]  # This could be a more specific URI based on your application's ontology
            g.add((source_uri, relation_uri, target_uri))

        # Serialize the graph to Turtle format and save to file
        g.serialize(destination=file_path, format='turtle')

    def load_knowledge_graph_from_turtle(file_path: str) -> KnowledgeGraph:
        kg = KnowledgeGraph()
        
        g = Graph()
        g.parse(file_path, format="turtle")
        
        # Iterate over all RDF triples in the graph
        for subject, predicate, obj in g:
            # Simplify the URIRefs and Literals to strings
            subject_str = str(subject)
            predicate_str = str(predicate)
            obj_str = str(obj) if isinstance(obj, Literal) else str(obj)
            
            # Extract the local name of the subject and predicate for use as identifiers in the KnowledgeGraph
            subject_id = urlparse(subject_str).path.split('/')[-1]
            predicate_id = urlparse(predicate_str).path.split('/')[-1]
            
            # Check if the predicate represents a relationship, an attribute, or an embedding
            if predicate_id == "type":
                # Type assertion; handle accordingly by setting a type attribute
                if not kg.graph.has_node(subject_id):
                    kg.add_concept(subject_id, {'type': obj_str})
                else:
                    kg.graph.nodes[subject_id]['type'] = obj_str
            elif predicate_id == "embedding":
                # This triple represents an embedding; parse and add/update the node with this embedding
                embedding_list = np.array([float(x) for x in obj_str.split(',')])  # Convert string back to list of floats
                if not kg.graph.has_node(subject_id):
                    kg.add_concept(subject_id, {'embedding': embedding_list})
                else:
                    kg.graph.nodes[subject_id]['embedding'] = embedding_list
            elif isinstance(obj, Literal):
                # Attribute of the subject
                if not kg.graph.has_node(subject_id):
                    kg.add_concept(subject_id, {predicate_id: obj_str})
                else:
                    kg.graph.nodes[subject_id][predicate_id] = obj_str
            else:
                # Relationship
                target_id = urlparse(str(obj)).path.split('/')[-1]
                if not kg.graph.has_node(subject_id):
                    kg.add_concept(subject_id, {})
                if not kg.graph.has_node(target_id):
                    kg.add_concept(target_id, {})
                kg.add_relationship(subject_id, target_id, {predicate_id})

        return kg


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

    def generate_concept_combinations(self, parsed_query: Dict) -> List[Dict]:
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

        # Generate concept combinations and add relationship information in a dictionary
        concept_combinations = []
        
        for concept1 in concepts:
            for concept2 in concepts:
                if concept1 != concept2:
                    # The combination is a result of the cartesian product of concepts plus a random relationship
                    # TO-DO: Add more sophisticated relationship generation
                    # TO-DO: Add constraints based on the knowledge graph
                    # TO-DO: Support more than two concepts and more than one relationship
                    combination = {
                        "concept_pair": [concept1, concept2],
                        "relationship": random.choice(relationships),
                        "embeddings_pair": [
                            self.knowledge_graph.get_concept_embeddings(concept1),
                            self.knowledge_graph.get_concept_embeddings(concept2)
                        ],
                        "relationship_embedding": self.knowledge_graph.get_concept_embeddings(random.choice(relationships))
                    }
                    concept_combinations.append(combination)
        
        return concept_combinations

    def rank_concept_combinations(self, concept_combinations: List[Dict], kg: KnowledgeGraph) -> List[Dict]:
        """
        Ranks concept combinations based on semantic similarity and knowledge graph insights.

        Parameters:
        - concept_combinations (List[Dict]): A list of concept combinations with embeddings.
        - kg (KnowledgeGraph): The knowledge graph instance to use for retrieving similar concepts.

        Returns:
        - List[Dict]: A list of ranked concept combinations.
        """
        
        ranked_combinations = []
        for combination in concept_combinations:
            # Calculate semantic similarity score based on embeddings
            concept1_embeddings = combination['embeddings_pair'][0]
            concept2_embeddings = combination['embeddings_pair'][1]
            relationship_embeddings = combination['relationship_embedding']

            # Here we can compute a combined embedding for the concept pair and the relationship.
            # The reason why this works is because the embeddings are vectors in a high-dimensional space, and we can perform operations on them.
            # The most effective way to produce a resulting embedding that captures the relationship between the two concepts is to perform a simple 
            # operation like addition or concatenation. Let's go with addition as it preserves the dimensionality of the embeddings.
            # We can also normalize the combined embeddings to ensure that the magnitude of the embeddings does not affect the similarity score.
            combined_embeddings = concept1_embeddings + relationship_embeddings + concept2_embeddings
            combined_embeddings /= np.linalg.norm(combined_embeddings) if np.linalg.norm(combined_embeddings) != 0 else 1

            # Retrieve the most similar concept embeddings from the knowledge graph
            most_similar_concept_embeddings = kg.get_most_similar_combined_concepts(combined_embeddings, 1)
            similarity_score = self.calculate_similarity_score(combined_embeddings, most_similar_concept_embeddings)

            # Add similarity score to the combination dictionary
            combination["similarity_score"] = similarity_score
            ranked_combinations.append(combination)
        
        # Sort concept combinations based on similarity score
        ranked_combinations.sort(key=lambda x: x["similarity_score"], reverse=True)
        return ranked_combinations

    def craft_gpt_prompt(self, task: str, concept_combination: List[str]) -> str:
        """
        Crafts a GPT-4 prompt based on a concept combination and knowledge graph insights.

        Parameters:
        - concept_combination (list): A combination of concepts.

        Returns:
        str: A crafted prompt for GPT-4 incorporating the selected concept combination and insights from the knowledge graph.
        """

        # The reason the following prompt works is because it is a placeholder for the actual prompt that would be crafted based on the concept combination and the task.
        # Particularly, the task would be used to guide the prompt generation process, and the concept combination would be used to provide context and constraints for the creative text generation.
        prompt = f"Perform task '{task}' using concepts: [".join(concept_combination) + "], exploring their unique relationship."
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
            prompt = self.craft_gpt_prompt(parsed_query['task'], combination)
            creative_text = generate_text(prompt)
            return creative_text
        return "No creative text generated."