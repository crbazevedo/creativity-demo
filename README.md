# Creativity-Demo

Welcome to the Creativity-Demo repository, which hosts the Creative Combinatorics project. This project is designed to harness the power of Knowledge Graphs (KGs) and GPT-4 for enhanced creative text generation, providing a suite of tools for AI engineers and students to explore computational creativity.

## Features

- **Text Embeddings**: Calculate semantic embeddings for given text inputs.
- **Creative Text Generation**: Generate text based on prompts using an integrated LLM like GPT-4.
- **Knowledge Graph Construction**: Build and serialize KGs from text.
- **Knowledge Graph Update**: Enhance existing KGs with new data.
- **Novel Concept Generation**: Generate novel concepts by leveraging the semantic structure of KGs.

## Project Structure

- `cli.py`: The command-line interface for interacting with the project's features.
- `api_client.py`: Manages API interactions.
- `embeddings.py`: Handles the creation of semantic embeddings.
- `gpt_integration.py`: Facilitates the integration with GPT-4 for text generation.
- `knowledge_graph.py`: Contains the logic for KG management.
- `storage.py`: Implements data persistence and retrieval.
- `test_knowledge_graph.py`: Provides unit tests for KG functionality.

## Installation

```bash
git clone https://github.com/crbazevedo/creativity-demo.git
cd creativity-demo
pip install -r requirements.txt
```

# Usage

The CLI supports multiple commands for different functionalities:

- **Calculate embeddings for input text:**
  ```bash
  python -m creative_combinatorics.cli embeddings "This is a test sentence."
  ```

- **Generate creative text based on a prompt:**
  ```bash
  python -m creative_combinatorics.cli generate "Once upon a time..."
  ```

- **Build and save the KG from ingested text:**
  ```bash
  python -m creative_combinatorics.cli build_save_kg "Text to ingest" "path/to/kg.ttl"
  ```

- **Load KG and generate novel concepts:**
  ```bash
  python -m creative_combinatorics.cli load_generate "path/to/kg.ttl" "Prompt" "Content"
  ```

- **Update KG from ingested text:**
  ```bash
  python -m creative_combinatorics.cli update_kg "path/to/kg.ttl" "Text to update"
  ```

# Contributing

We encourage contributions! Please review `CONTRIBUTING.md` for guidelines on making contributions to this project.

# License

This project is released under the MIT License. See `LICENSE.md` for more details.

# Acknowledgments

- The contributors who have made this project possible.
- OpenAI for providing the GPT-4 API.
