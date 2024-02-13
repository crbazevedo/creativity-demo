"""
api_client.py

This module manages API interactions with OpenAI, abstracting the complexities of HTTP requests to fetch embeddings 
and generate text using GPT-4. It ensures secure API key management, constructs request payloads, processes responses,
and includes robust error handling and logging for reliable operations. Functions manage rate limits and provide a
simplified interface for semantic analysis and creative text generation within the system.

Functions:
- configure_api_key(api_key: str): Securely sets the OpenAI API key.
- get_embeddings(text: str) -> Dict: Fetches text embeddings.
- generate_text(prompt: str) -> str: Generates text based on prompts.
- handle_api_errors(response: requests.Response): Processes API errors.
- log_api_call(endpoint: str, payload: Dict): Logs API request details.

Usage requires a valid OpenAI API key configured via `configure_api_key`.
"""

from typing import Dict
import os
import requests
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import re
import openai

def configure_api_key():
    """
    Retrieves and sets the OpenAI API key from an environment variable.
    
    Validates the format of the API key to ensure it appears correct. Raises an exception if the key is not found or invalid.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OpenAI API key is missing. Please set the OPENAI_API_KEY environment variable.")
    
    if not re.match(r'^[A-Za-z0-9-_]+$', api_key):
        raise ValueError("Invalid OpenAI API key format.")
    
    openai.api_key = api_key

def get_embeddings(text: str) -> list:
    """
    Fetches embeddings for the given text using OpenAI's API with a specified model.

    Parameters:
    - text (str): The input text for which embeddings are requested.

    Returns:
    - list: A list containing the embeddings information for the input text.

    Raises:
    - Exception: If the API call fails or returns an error.
    """
    try:
        response = openai.Embedding.create(
            input=text,
            model="text-embedding-3-large"
        )
        embeddings = response.data[0].embedding  # Accessing the embeddings from the response
        return embeddings
    except Exception as e:
        logger.error(f"An error occurred while fetching embeddings: {e}")
        raise

def generate_text(prompt: str) -> str:
    """
    Generates text based on the given prompt using OpenAI's GPT-4 model.
    

    Parameters:
    - prompt (str): The input prompt for generating text.

    Returns:
    - str: The generated text based on the input prompt.

    Raises:
    - Exception: If the API call fails or returns an error.
    """


    try:
        response = openai.Completion.create(
            engine="gpt-4-turbo-preview",
            prompt=prompt,
            max_tokens=8000,
            messages=[
                {"role": "system", "content": "You are a highly creative writer."},
                {"role": "user", "content": prompt}
            ]
        )
        generated_text = response.choices[0].message  # Accessing the generated text from the response
        return generated_text
    except Exception as e:
        logger.error(f"An error occurred while generating text: {e}")
        raise

def handle_api_errors(response: requests.Response):
    """
    Processes API errors and raises exceptions for failed API calls.

    Parameters:
    - response (requests.Response): The response object from the API call.

    Raises:
    - Exception: If the API call returns an error.
    """
    if response.status_code != 200:
        error_message = response.json().get("error", "An error occurred while making an API call.")
        logger.error(f"API call failed with status code {response.status_code}: {error_message}")
        raise Exception(f"API call failed with status code {response.status_code}: {error_message}")
    
def log_api_call(endpoint: str, payload: Dict):
    """
    Logs details of an API request.

    Parameters:
    - endpoint (str): The API endpoint for the request.
    - payload (Dict): The payload of the request.

    Returns:
    - None
    """
    logger.info(f"Making an API call to {endpoint} with payload: {payload}")

