import argparse
from creative_combinatorics.embeddings import calculate_embeddings_for_text
from creative_combinatorics.gpt_integration import generate_creative_text
from creative_combinatorics.knowledge_graph import KnowledgeGraph

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
def main():
    parser = argparse.ArgumentParser(description='Creativity Demo CLI')
    subparsers = parser.add_subparsers(dest='command')

    # Command for embeddings
    parser_embeddings = subparsers.add_parser('embeddings', help='Calculate text embeddings')
    parser_embeddings.add_argument('text', type=str, help='Input text to calculate embeddings for')

    # Command for creative text generation
    parser_generate = subparsers.add_parser('generate', help='Generate creative text')
    parser_generate.add_argument('prompt', type=str, help='Prompt for creative text generation')

    # Parse arguments
    args = parser.parse_args()

    # Handle commands
    if args.command == 'embeddings':
        embeddings = calculate_embeddings_for_text(args.text)
        print(embeddings)
    elif args.command == 'generate':
        creative_text = generate_creative_text(args.prompt)
        print(creative_text)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
