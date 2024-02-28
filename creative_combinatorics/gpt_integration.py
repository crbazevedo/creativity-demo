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
import json
import markdown
import rdflib
from rdflib import Graph, URIRef, Literal, Namespace, BNode
from rdflib.namespace import RDF, XSD
from urllib.parse import urlparse, quote
from creative_combinatorics.knowledge_graph import KnowledgeGraph
from creative_combinatorics.api_client import get_embeddings, generate_text

class CreativeTextGenerator:
    def __init__(self, knowledge_graph: KnowledgeGraph):
        self.knowledge_graph = knowledge_graph

    def parse_query(self, text, return_json=False):

        # Determine if the input is a file path or a string
        if text.endswith('.txt'):
            # Open and read the text contents
            with open(text, 'r') as file:
                text = file.read()
        
        api_key = api_client.configure_api_key()
        endpoint = 'https://api.openai.com/v1/chat/completions'
        
        headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json',
        }
        
        # Update the prompt to instruct GPT-4 to analyze the query for concepts, relationships, and tasks
        prompt = f"Given the query: '{text}', identify the 'concepts', 'relationships', and 'task'. Assume the relationships are formatted as a triple string "\
                 f"`(concept1, relation, concept2)`, within PARENTESHES. Do not use dicts to represent triples. Use parantheses notation, only. "\
                 f"All triple shall be not-null, i.e., no missing values are allowed. "\
                 f"Return the analysis in JSON format with lists of triples for each attribute. "\
                 f"Focus on the main topic of the input and deduce which concepts and relationships are most relevant. "\
                 f"If the query include examples, you can encode the concepts in the examples with the relationship 'an example of' ."\
                 f"Concepts can be single words or multi-word expressions. Relationships can be verbs or verb phrases. "\
                 f"Concepts can also be whole sentences or paragraphs to represent more complex ideas such as events or actions. "\
                 f"Relationships can also be encoded as a single word or multi-word expressions. "\
                 f"Relationships can also be temporal, spatial, or causal. "\
                 f"Add as many concepts and relationships as you can find in the input. "\
        
        data = {
            'model': 'gpt-4-turbo-preview', 
            'max_tokens': 4096,
            'temperature': 0.1,
            'top_p': 1.0,
            'frequency_penalty': 0.0,
            'presence_penalty': 0.0,
            'response_format': {'type': 'json_object'},
            'messages': [
                {
                "role": "system",
                "content": "You are a highly qualified knowledge graph extractor, an expert in extracting triples " +\
                           "(knowledge triplets composed of subject, predicate, object) from text."
                },
                {
                    "role": "user",
                    "content": prompt
                }]
        }
        
        response = requests.post(endpoint, headers=headers, json=data)
        print(response.json())    

        if response.status_code == 200:
            response_data = response.json()

            # Assuming `response_data['choices'][0]['message']['content']` contains the markdown string
            content = response_data['choices'][0]['message']['content']
            json_start = content.find('{')
            json_end = content.rfind('}') + 1  # Use rfind to ensure we get the last }
            json_block = content[json_start:json_end]
            print(f"JSON start: {json_start}, JSON end: {json_end}")
            print(f"JSON block: {json_block}")
            json_object = json.loads(json_block)
            # Convert the JSON block into a Python dictionary
            try:
                json_object = json.loads(json_block)
            except json.decoder.JSONDecodeError as e:
                print(f"Failed to decode JSON: {e}")
                print("Faulty JSON block:", json_block)
                

            #print(f"JSON object: {json_object}")

            # Process the JSON response to extract triples of concepts and relationships.
            # Extracted information should be in the format: {'triples': ['(concept1,relation,concept2)',...]}
            # Try-catch block that checks that it containts all exepcted keys
            if return_json == False:
                try:
                    # response_data contains the parsed query in JSON format, so we can extract the concepts, relationships, and task
                    extracted_info = {
                        # We get a list of triples in the format `(concept1, relation, concept2)`. So, we need to split each triple into its components.
                        # The concepts correspond to the first and third elements of each triple, and the relationships correspond to the second element.
                        # First, we get the list of triples from the response data.
                        # We transform response_data['choices'][0]['message']['content'], which contains a json string, to a json type object
                        'triples': [triple for triple in json_object['relationships']]
                    }

                    print (f"Extracted info: {extracted_info}")
                except KeyError:
                    return f"Error: {response.text}"
                return extracted_info
            else:
                return json_object
        else:
            return f"Error: {response.text}"

    @staticmethod
    def build_knowledge_graph_from_triples(triples: List[str]) -> KnowledgeGraph:
        kg = KnowledgeGraph()

        # Process each triple safely
        print(f"triple: {triples}")
        print(f"Processing {len(triples['triples'])} triples")
        for triple in triples['triples']:
            # Here we split the triple into its components and remove the initial and end brackets
            # List has no attribute strip, so instead we will use the replace method
            parts = triple.replace('[','').replace(']','').split(',')

            # Ensure exactly three parts are present
            if len(parts) == 3:
                concept1, relation, concept2 = [part.strip() for part in parts]
                
                # Add concepts if they're not already in the graph
                if concept1 not in kg.graph:
                    print (f"Adding concept: {concept1} to the graph")
                    embedding = kg.calculate_embeddings_for_text(concept1)
                    kg.add_concept(concept1, {'embedding': embedding})
                if concept2 not in kg.graph:
                    print (f"Adding concept: {concept2} to the graph")
                    embedding = kg.calculate_embeddings_for_text(concept2)
                    kg.add_concept(concept2, {'embedding': embedding})
                
                # Add the relationship
                print(f"Adding relationship: {concept1} -> {relation} -> {concept2}")
                kg.add_relationship(concept1, concept2, {relation})
            else:
                print(f"Skipping malformed triple: {triple}")

        return kg

    def save_knowledge_graph_as_turtle(self, kg: KnowledgeGraph, file_path: str):
        g = Graph()
        
        # Define your namespaces
        MY_NS = Namespace("http://example.org/my_knowledge_graph#")
        
        for concept, attrs in kg.get_all_concepts():
            safe_concept = quote(concept)
            concept_uri = URIRef(MY_NS[safe_concept])
            # Here, add the concept node to the RDF graph with a generic type or more specific based on your class's structure
            g.add((concept_uri, RDF.type, MY_NS.Concept))
            
            # Serialize embeddings as a single attribute, if present
            if 'embedding' in attrs:
                # Convert the embedding list to a string representation
                embedding_str = ','.join(map(str, attrs['embedding']))
                g.add((concept_uri, MY_NS.embedding, Literal(embedding_str, datatype=XSD.string)))
                
        for source, target, relation_type in kg.get_all_relations():
            print(f"Adding relation: {source} -> {list(relation_type['relationships'])[0]} -> {target}")
            safe_source = quote(source)
            safe_relation_type = quote(list(relation_type['relationships'])[0])
            safe_target = quote(target)

            source_uri = URIRef(MY_NS[safe_source])
            target_uri = URIRef(MY_NS[safe_target])
            relation_uri = MY_NS[safe_relation_type]  # This could be a more specific URI based on your application's ontology
            g.add((source_uri, relation_uri, target_uri))

        # Serialize the graph to Turtle format and save to file
        g.serialize(destination=file_path, format='turtle')

    @staticmethod
    def update_knowledge_graph_from_triples(kg: KnowledgeGraph, triples: List[str]) -> KnowledgeGraph:
        """
        Updates a knowledge graph with new concepts and relationships from a list of triples.

        Args:
            kg (KnowledgeGraph): The knowledge graph to update.
            triples (List[str]): A list of triples in the format '[concept1, relation, concept2]'.

        Returns:
            KnowledgeGraph: The updated knowledge graph.
        """
        # Process each triple safely
        for triple in triples["triples"]:
            # Here we split the triple into its components and remove the initial and end brackets
            # Triple will be a string in the format '[concept1, relation, concept2]' or '(concept1, relation, concept2)'.
            # We need to split it into its components and remove the initial and end brackets (or parentheses).

            embedding1, embedding2 = None, None

            # Check if the triple is in the correct format
            if triple[0] in ['[', '('] and triple[-1] in [']', ')']:
                parts = triple[1:-1].split(',')
                # Ensure exactly three parts are present
                if len(parts) == 3:
                    concept1, relation, concept2 = [part.strip() for part in parts]
                    # Add concepts if they're not already in the graph
                    if concept1 not in kg.graph:
                        embedding1 = kg.calculate_embeddings_for_text(concept1)
                        kg.add_concept(concept1, {'embedding': embedding1})
                    if concept2 not in kg.graph:
                        embedding2 = kg.calculate_embeddings_for_text(concept2)
                        kg.add_concept(concept2, {'embedding': embedding2})
                    # Add the relationship
                    if concept1 in kg.graph and concept2 in kg.graph:
                        kg.add_relationship(concept1, concept2, {relation})
                else:
                    print(f"Skipping malformed triple: {triple}")
            else:
                print(f"Skipping malformed triple: {triple}")

            # Search for similar concepts and add relationships to the graph
            similar_concepts1 = kg.get_most_similar_concepts([concept1], 3)
            similar_concepts2 = kg.get_most_similar_concepts([concept2], 3)

            for similar_concept1, similar_concept2 in zip(similar_concepts1, similar_concepts2):
                kg.add_relationship(similar_concept1, concept1, {'similar_to'})
                kg.add_relationship(similar_concept2, concept2, {'similar_to'})

        return kg

    @staticmethod
    def load_knowledge_graph_from_turtle(file_path: str) -> KnowledgeGraph:
        kg = KnowledgeGraph()
        g = Graph()
        g.parse(file_path, format="turtle")

        def clean_encoded_string(encoded_string):
            import re, urllib
            # Pattern to find all occurrences of percent encoding
            # finalizing with a space or 28 (hex for open parenthesis)
            pattern = re.compile(r'(%25)+20', re.IGNORECASE)
            
            # Replace multiple encodings with a single space
            cleaned_string = pattern.sub(' ', encoded_string)
            
            # Decode the cleaned string in case there are other encoded characters
            decoded_string = urllib.parse.unquote(cleaned_string)
            
            return decoded_string
        
        for subject, predicate, obj in g:
            # Simplify URIs for better readability in the visualization
            subject_id = subject.split('#')[-1]
            predicate_id = predicate.split('#')[-1]
            obj_str = obj.split('#')[-1] if isinstance(obj, rdflib.term.URIRef) else str(obj)
            
            # Correct handling based on predicate type
            if predicate_id == "embedding":
                embedding_list = np.array([float(x) for x in obj_str.split(',')])
                kg.add_concept(subject_id, {'embedding': embedding_list})
            elif isinstance(obj, Literal):
                kg.add_concept(subject_id, {predicate_id: obj_str.replace('%20', ' ')})
            else:
                target_id = obj_str
                kg.add_relationship(subject_id, target_id, {predicate_id.replace('%20', ' ')})
        
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

        # Ensure embeddings1 is a 1D array
        embeddings1 = np.ravel(embeddings1)  # This flattens embeddings1 to 1D if it's not already

        # Ensure embeddings2 is also a 1D array, assuming embeddings2 is the variable for most_similar_concept_embeddings
        embeddings2 = np.ravel(embeddings2)  # This flattens embeddings2 to 1D

        # Now calculate the similarity score
        similarity_score = np.dot(embeddings1, embeddings2) / (np.linalg.norm(embeddings1) * np.linalg.norm(embeddings2))

        return similarity_score

    def generate_concept_combinations(self, parsed_query: Dict, pct_combinations: float =1.0) -> List[Dict]:
        """
        Generates combinations of concepts based on the parsed query.

        Parameters:
        - parsed_query (dict): The parsed query containing concepts and relationships.

        Returns:
        - list: A list of concept combinations.
        """
        print(f"Parsed query: {parsed_query}")
        # Convert pased query to a dictionary
        # Find extracted concepts and relationships from the parsed query
        concepts = parsed_query.get('concepts')
        # WHat parsed_query.get('relationships', []) returns if the key is not present in the dictionary is an empty list
        relationships = parsed_query.get('relationships', [])

        print(f"Concepts: {concepts}, Relationships: {relationships}")

        # Generate concept combinations and add relationship information in a dictionary
        concept_combinations = []
        
        # Computes number of pairwise combinations of concepts
        num_combinations = len(concepts) * (len(concepts) - 1)
        num_combinations = int(num_combinations * pct_combinations)
        count = 0

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
                    print(f"Combination: {combination}")
                    concept_combinations.append(combination)
                    count += 1
            if count >= num_combinations:
                break
        
        return concept_combinations

    def rank_concept_combinations(self, concept_combinations: List[Dict]) -> List[Dict]:
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

            # Retrieve the most similar subgraphs  from the knowledge graph
            similarity_score, most_similar_subgraphs = self.knowledge_graph.get_most_similar_subgraphs(combined_embeddings, 1)

            # Add similarity score to the combination dictionary
            combination["similarity_score"] = similarity_score
            combination["most_similar_subgraphs"] = most_similar_subgraphs
            ranked_combinations.append(combination)
        
        # Sort concept combinations based on similarity score
        ranked_combinations.sort(key=lambda x: x["similarity_score"], reverse=True)
        return ranked_combinations

    def craft_gpt_prompt(self, task: str, content: str, concept_combination: List[str]) -> str:
        """
        Crafts a GPT-4 prompt based on a concept combination and knowledge graph insights.

        Parameters:
        - concept_combination (list): A combination of concepts.

        Returns:
        str: A crafted prompt for GPT-4 incorporating the selected concept combination and insights from the knowledge graph.
        """

        # Remove the 'embedding' key from the concept combination
        combinations = []
        for combination in concept_combination['most_similar_subgraphs']:
            del combination['embedding']
            combinations.append(combination)

        # The reason the following prompt works is because it is a placeholder for the actual prompt that would be crafted based on the concept combination and the task.
        # Particularly, the task would be used to guide the prompt generation process, and the concept combination would be used to provide context and constraints for the creative text generation.
        prompt = f"You goal is to generate a PROMPT instructing a LLM to perform the TASK ['{task}']. "\
                 f"You should write the prompt in plain natural language and rephrase the TASK in the context provided in the following KNOWLEDGE SUBGRAPH: ['{combinations}']. "\
                 f"Remember the list of concepts.  "\
                 f"The concepts should be combined with the MAIN CONCEPT PAIR: Concept Pair 1['{concept_combination['concept_pair'][0]}'] "\
                 f"and Concept Pair 2['{concept_combination['concept_pair'][1]}']. Their RELATIONSHIP is ['{concept_combination['relationship']}']. "\
                 f"Your goal is to generate write the PROMPT naturally combining the elements in the KNOWLEDGE SUGBRAGH and the MAIN CONCEPT PAIR to produce enriched and contextual instructions. "\
                 f"The enriched and contextual instructions will also consider  the extraction of concepts and relationships from the USER PROVIDED CONTENT: ['{content}']."
        return generate_text(prompt)
    
    def generate_creative_text(self, instruction: str, content: str) -> str:
        """
        Generates creative text based on the input query by leveraging a knowledge graph and GPT-4.

        Parameters:
        - query (str): The input query string.

        Returns:
        - str: The generated creative text.
        """
        creative_texts = {}
        parsed_query = self.parse_query(content,return_json=True)
        concept_combinations = self.generate_concept_combinations(parsed_query, pct_combinations=0.005)
        print(f"Generated {len(concept_combinations)} concept combinations")
        #print(f"Concept combinations: {concept_combinations}")

        ranked_combinations = self.rank_concept_combinations(concept_combinations)
        print(f"Rank: {ranked_combinations}")
        for combination in ranked_combinations:
            prompt = self.craft_gpt_prompt(instruction, content, combination)
            print(f"Combination: {combination}, Prompt: {prompt}")
            creative_text = generate_text(prompt)
            print(f"Creative text: {creative_text}")
            creative_texts.update({content: creative_text})
        return creative_texts