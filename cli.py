import argparse
from creative_combinatorics.knowledge_graph import KnowledgeGraph
from creative_combinatorics.gpt_integration import CreativeTextGenerator

"""
CLI for Creativity Demo

This module provides a command-line interface (CLI) for the Creativity Demo. It allows users to calculate embeddings for input text and generate creative text based on a prompt.

Example:
    $ python -m creative_combinatorics.cli embeddings "This is a test sentence."
    $ python -m creative_combinatorics.cli generate "Once upon a time, in a land far, far away..."

Attributes:
    parser (ArgumentParser): The argument parser for the CLI.
    subparsers (ArgumentParser): The subparsers for the CLI.
    parser_embeddings (ArgumentParser): The argument parser for the embeddings command.
    parser_generate (ArgumentParser): The argument parser for the generate command.
    args (Namespace): The parsed arguments from the CLI.
"""

# Assuming build_knowledge_graph_from_triples is properly imported or defined in this script
# from creative_combinatorics.knowledge_graph import build_knowledge_graph_from_triples

def main():
    parser = argparse.ArgumentParser(description='Creativity Demo CLI')
    subparsers = parser.add_subparsers(dest='command')

    # Command for embeddings
    parser_embeddings = subparsers.add_parser('embeddings', help='Calculate text embeddings')
    parser_embeddings.add_argument('text', type=str, help='Input text to calculate embeddings for')

    # Command for creative text generation
    parser_generate = subparsers.add_parser('generate', help='Generate creative text')
    parser_generate.add_argument('prompt', type=str, help='Prompt for creative text generation')

    # New command for building and saving the KG from ingested text
    parser_build_save_kg = subparsers.add_parser('build_save_kg', help='Build and save KG from ingested text')
    parser_build_save_kg.add_argument('text', type=str, help='Input text to build and save KG from')
    parser_build_save_kg.add_argument('file_path', type=str, help='Path to save the KG Turtle file')

    # New command for loading the KG and generating novel concepts
    parser_load_generate = subparsers.add_parser('load_generate', help='Load KG and generate novel concepts')
    parser_load_generate.add_argument('file_path', type=str, help='Path of the KG Turtle file to load')
    parser_load_generate.add_argument('prompt', type=str, help='Prompt for creative text generation')
    parser_load_generate.add_argument('content', type=str, help='Content to generate novel concepts from')

    # New command for updating the KG from ingested text
    parser_update_kg = subparsers.add_parser('update_kg', help='Update KG from ingested text')
    parser_update_kg.add_argument('file_path', type=str, help='Path of the KG Turtle file to update')
    parser_update_kg.add_argument('text', type=str, help='Input text to update KG from')


    args = parser.parse_args()

    if args.command == 'embeddings':
        embeddings = KnowledgeGraph.calculate_embeddings_for_text(args.text)
        print(embeddings)
    elif args.command == 'generate':
        creative_text = CreativeTextGenerator.generate_creative_text(args.prompt)
        print(creative_text)
    elif args.command == 'build_save_kg':
        kg = KnowledgeGraph()
        # Instantiate CreativeTextGenerator
        text_generator = CreativeTextGenerator(knowledge_graph=kg)

        # Use the instance to call parse_query
        parsed_triples = text_generator.parse_query(args.text)
        
        # Assuming build_knowledge_graph_from_triples is correctly defined/imported
        
        kg = text_generator.build_knowledge_graph_from_triples(parsed_triples)
        
        # Assuming save_knowledge_graph_as_turtle is correctly defined/imported
        text_generator.save_knowledge_graph_as_turtle(kg, args.file_path)
        print(f"KG built and saved to {args.file_path}")
    elif args.command == 'load_generate': 
        # Load the KnowledgeGraph from the Turtle file
        kg = CreativeTextGenerator.load_knowledge_graph_from_turtle(args.file_path)
        kg.encode_nodes()
        print(f"KG loaded from {args.file_path}")

        # Instantiate CreativeTextGenerator
        text_generator = CreativeTextGenerator(knowledge_graph=kg)
        
        # Now you can use text_generator to generate creative text based on the KG
        creative_texts = text_generator.generate_creative_text(args.prompt, args.content)
        print(creative_texts)
    elif args.command == 'update_kg':
        # Load the KnowledgeGraph from the Turtle file
        kg = CreativeTextGenerator.load_knowledge_graph_from_turtle(args.file_path)
        kg.encode_nodes()
        print(f"KG loaded from {args.file_path}")

        # Instantiate CreativeTextGenerator
        text_generator = CreativeTextGenerator(knowledge_graph=kg)

        # Use the instance to call parse_query
        parsed_triples = text_generator.parse_query(args.text)

        # Assuming update_knowledge_graph_from_triples is correctly defined/imported
        kg = text_generator.update_knowledge_graph_from_triples(kg, parsed_triples)

        # Assuming save_knowledge_graph_as_turtle is correctly defined/imported
        text_generator.save_knowledge_graph_as_turtle(kg, args.file_path)

        print(f"KG updated and saved to {args.file_path}")

    else:
        parser.print_help()

if __name__ == "__main__":
    main()

