import json
import logging
from pathlib import Path
import faiss
import numpy as np
from tqdm import tqdm
import os
from typing import List
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def get_embeddings(texts: List[str], batch_size: int = 100) -> np.ndarray:
    """Generate embeddings for a list of texts using Azure OpenAI API."""
    embeddings = []
    headers = {
        "api-key": os.getenv('AZURE_OPENAI_API_KEY'),
        "Content-Type": "application/json"
    }
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        try:
            data = {
                "input": batch
            }
            
            response = requests.post(
                os.getenv('AZURE_ENDPOINT'),
                headers=headers,
                json=data
            )
            
            if response.status_code == 200:
                result = response.json()
                batch_embeddings = [item['embedding'] for item in result['data']]
                embeddings.extend(batch_embeddings)
                logger.debug(f"Processed batch {i//batch_size + 1}")
            else:
                logger.error(f"Error from API: {response.status_code} - {response.text}")
                raise Exception(f"API request failed with status {response.status_code}")
                
        except Exception as e:
            logger.error(f"Error processing batch {i//batch_size + 1}: {str(e)}")
            raise
    
    return np.array(embeddings)

def main():
    try:
        # Read test articles
        logger.info("Reading test articles...")
        input_file = Path("data/articles_combined/test.json")
        with open(input_file, 'r', encoding='utf-8') as f:
            articles = json.load(f)
        
        logger.info(f"Loaded {len(articles)} articles")
        
        # Extract texts
        texts = [article.get('text', '') for article in articles]
        
        # Generate embeddings
        logger.info("Generating embeddings using Azure OpenAI API...")
        embeddings = get_embeddings(texts)
        
        # Create and save FAISS index
        logger.info("Creating FAISS index...")
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)
        
        # Save index and metadata
        output_dir = Path("data/vector_store")
        output_dir.mkdir(exist_ok=True)
        
        logger.info(f"Saving index to {output_dir}...")
        faiss.write_index(index, str(output_dir / "simple_articles.index"))
        
        # Save metadata
        metadata = [{
            'id': article.get('id'),
            'title': article.get('title'),
            'published_at': article.get('published_at')
        } for article in articles]
        
        with open(output_dir / "simple_metadata.json", 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        logger.info("Processing complete!")
        logger.info(f"Created embeddings for {len(articles)} articles")
        logger.info(f"Vector store saved to {output_dir}")
        
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main() 