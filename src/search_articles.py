import json
import logging
from pathlib import Path
from typing import List, Dict, Any
import faiss
import numpy as np
import requests
from dotenv import load_dotenv
import time
import os
from requests.exceptions import RequestException
from datetime import datetime

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Azure endpoints
AZURE_EMBEDDING_ENDPOINT = os.getenv('AZURE_ENDPOINT')
OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')
OPENROUTER_MODEL_NAME = os.getenv('OPENROUTER_MODEL_NAME')
OPENROUTER_ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"

class ArticleSearcher:
    def __init__(self, vector_store_path: Path = Path("data/vector_store")):
        self.vector_store_path = vector_store_path
        self.index = None
        self.metadata = None
        self.load_vector_store()
    
    def load_vector_store(self):
        """Load the FAISS index and metadata from disk."""
        try:
            # Load FAISS index
            index_path = self.vector_store_path / "articles.index"
            self.index = faiss.read_index(str(index_path))
            
            # Load metadata
            metadata_path = self.vector_store_path / "metadata.json"
            with open(metadata_path, 'r', encoding='utf-8') as f:
                self.metadata = json.load(f)
                
            logger.info(f"Loaded vector store with {len(self.metadata)} chunks")
        except Exception as e:
            logger.error(f"Error loading vector store: {str(e)}")
            raise
    
    def get_embedding(self, text: str, max_retries: int = 3) -> np.ndarray:
        """Generate embedding for a single text using Azure OpenAI API."""
        headers = {
            "api-key": os.getenv('AZURE_OPENAI_API_KEY'),
            "Content-Type": "application/json"
        }
        
        retry_count = 0
        delay = 1.0
        
        while retry_count < max_retries:
            try:
                data = {
                    "input": [text]  # API expects a list
                }
                
                response = requests.post(
                    AZURE_EMBEDDING_ENDPOINT,
                    headers=headers,
                    json=data
                )
                
                if response.status_code == 200:
                    result = response.json()
                    return np.array(result['data'][0]['embedding'])
                    
                elif response.status_code == 429:  # Rate limit error
                    retry_after = int(response.headers.get('Retry-After', delay))
                    logger.warning(f"Rate limit hit. Waiting {retry_after} seconds...")
                    time.sleep(retry_after)
                    retry_count += 1
                    delay *= 2
                    
                else:
                    logger.error(f"Error from API: {response.status_code} - {response.text}")
                    raise Exception(f"API request failed with status {response.status_code}")
                    
            except RequestException as e:
                logger.warning(f"Request failed: {str(e)}. Retrying in {delay} seconds...")
                time.sleep(delay)
                retry_count += 1
                delay *= 2
        
        raise Exception("Max retries exceeded")
    
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for the most relevant articles given a query."""
        try:
            # Generate embedding for the query
            query_embedding = self.get_embedding(query)
            
            # Search in FAISS index
            distances, indices = self.index.search(
                query_embedding.reshape(1, -1).astype('float32'),
                top_k
            )
            
            # Get relevant chunks with their metadata
            results = []
            for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
                if idx < len(self.metadata):  # Ensure valid index
                    chunk = self.metadata[idx].copy()
                    chunk['similarity_score'] = float(1 / (1 + distance))  # Convert distance to similarity score
                    chunk['rank'] = i + 1
                    results.append(chunk)
            
            return results
            
        except Exception as e:
            logger.error(f"Error during search: {str(e)}")
            raise
    
    def format_results(self, results: List[Dict[str, Any]]) -> str:
        """Format search results into a readable string."""
        formatted_output = []
        
        for result in results:
            formatted_output.append(f"\nRank {result['rank']} (Score: {result['similarity_score']:.3f})")
            formatted_output.append(f"Title: {result['title']}")
            formatted_output.append(f"Author: {result['author']}")
            formatted_output.append(f"Published: {result['published_at']}")
            formatted_output.append(f"Category: {result['category']}")
            formatted_output.append(f"Section: {result['section']}")
            formatted_output.append("\nRelevant Text:")
            formatted_output.append(result.get('original_text', result['text']))  # Use original text if available
            formatted_output.append("-" * 80)
        
        return "\n".join(formatted_output)
    
    def generate_summary(self, results: List[Dict[str, Any]], prompt: str, max_retries: int = 3) -> str:
        """Generate a summary using the search results and a prompt."""
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "HTTP-Referer": "https://github.com/yourusername/yourrepo",  # Replace with your actual domain
            "Content-Type": "application/json"
        }
        
        # Prepare the context from search results using original text and metadata
        context = "\n\n".join([
            f"Article {i+1}:\nTitle: {result['title']}\nAuthor: {result['author']}\nPublished: {result['published_at']}\nCategory: {result['category']}\nSection: {result['section']}\nText: {result.get('original_text', result['text'])}"
            for i, result in enumerate(results)
        ])
        
        # Prepare the messages for the API
        messages = [
            {"role": "system", "content": "You are a helpful assistant that analyzes articles and provides relevant information based on the user's prompt. Please provide your response in English eventhough the articles are in German. Feel free to translate the text in the articles to English if needed."},
            {"role": "user", "content": f"Here are some relevant articles:\n\n{context}\n\nPlease analyze these articles and {prompt}"}
        ]
        
        data = {
            "model": OPENROUTER_MODEL_NAME,
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 1000
        }
        
        retry_count = 0
        delay = 1.0
        
        while retry_count < max_retries:
            try:
                response = requests.post(
                    OPENROUTER_ENDPOINT,
                    headers=headers,
                    json=data
                )
                
                if response.status_code == 200:
                    result = response.json()
                    return result['choices'][0]['message']['content']
                    
                elif response.status_code == 429:  # Rate limit error
                    retry_after = int(response.headers.get('Retry-After', delay))
                    logger.warning(f"Rate limit hit. Waiting {retry_after} seconds...")
                    time.sleep(retry_after)
                    retry_count += 1
                    delay *= 2
                    
                else:
                    logger.error(f"Error from API: {response.status_code} - {response.text}")
                    raise Exception(f"API request failed with status {response.status_code}")
                    
            except RequestException as e:
                logger.warning(f"Request failed: {str(e)}. Retrying in {delay} seconds...")
                time.sleep(delay)
                retry_count += 1
                delay *= 2
        
        raise Exception("Max retries exceeded")

def process_topic(searcher: ArticleSearcher, topic: Dict[str, str]) -> Dict[str, Any]:
    """Process a single topic and its prompts."""
    results = []
    
    # Search for relevant articles
    logger.info(f"Searching for articles related to: {topic['topic']}")
    search_results = searcher.search(topic['topic'])
    
    # Process each prompt
    for prompt in topic['prompts']:
        logger.info(f"Generating analysis for prompt: {prompt}")
        summary = searcher.generate_summary(search_results, prompt)
        
        # Store only essential information
        prompt_result = {
            "prompt": prompt,
            "analysis": summary
        }
        results.append(prompt_result)
    
    return {
        "topic": topic['topic'],
        "results": results
    }

def main():
    try:
        # Initialize searcher
        searcher = ArticleSearcher()
        
        # Read topics and prompts from JSON file
        input_file = Path("data/topics_prompts.json")
        logger.info(f"Reading topics and prompts from {input_file}")
        with open(input_file, 'r', encoding='utf-8') as f:
            topics = json.load(f)
        
        # Process each topic
        all_results = []
        for topic in topics:
            logger.info(f"\nProcessing topic: {topic['topic']}")
            result = process_topic(searcher, topic)
            all_results.append(result)
            
            # Display results for this topic
            print(f"\nResults for topic: {topic['topic']}")
            for prompt_result in result['results']:
                print(f"\nPrompt: {prompt_result['prompt']}")
                print("-" * 80)
                print(prompt_result['analysis'])
                print("-" * 80)
        
        # Save results to JSON file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = Path("data/analysis_results.json")
        logger.info(f"Saving results to {output_file}")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        
        logger.info("Processing complete!")
        
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main() 