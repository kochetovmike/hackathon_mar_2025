import json
import logging
from pathlib import Path
import re
from typing import List, Dict, Any, Generator
import faiss
import numpy as np
import spacy
from tqdm import tqdm
import os
import gc
import requests
from dotenv import load_dotenv
import time
from requests.exceptions import RequestException

# Load environment variables
load_dotenv()

# Set up logging with more verbose output
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize spaCy model
try:
    logger.debug("Loading spaCy model...")
    nlp = spacy.load('de_core_news_lg')
    logger.debug("spaCy model loaded successfully")
except Exception as e:
    logger.error(f"Error loading spaCy model: {str(e)}")
    raise

def get_embeddings(texts: List[str], batch_size: int = 100, max_retries: int = 5, initial_delay: float = 1.0) -> np.ndarray:
    """Generate embeddings for a list of texts using Azure OpenAI API with rate limit handling."""
    embeddings = []
    headers = {
        "api-key": os.getenv('AZURE_OPENAI_API_KEY'),
        "Content-Type": "application/json"
    }
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        retry_count = 0
        delay = initial_delay
        
        while retry_count < max_retries:
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
                    break  # Success, exit retry loop
                    
                elif response.status_code == 429:  # Rate limit error
                    retry_after = int(response.headers.get('Retry-After', delay))
                    logger.warning(f"Rate limit hit. Waiting {retry_after} seconds before retry {retry_count + 1}/{max_retries}")
                    time.sleep(retry_after)
                    retry_count += 1
                    delay *= 2  # Exponential backoff
                    
                else:
                    logger.error(f"Error from API: {response.status_code} - {response.text}")
                    raise Exception(f"API request failed with status {response.status_code}")
                    
            except RequestException as e:
                logger.warning(f"Request failed: {str(e)}. Retrying in {delay} seconds...")
                time.sleep(delay)
                retry_count += 1
                delay *= 2  # Exponential backoff
                
        if retry_count == max_retries:
            logger.error(f"Failed to process batch {i//batch_size + 1} after {max_retries} retries")
            raise Exception("Max retries exceeded")
            
        # Add a small delay between successful batches to prevent rate limiting
        time.sleep(0.5)
    
    return np.array(embeddings)

class ArticleProcessor:
    def __init__(self, chunk_size: int = 100):
        self.chunk_size = chunk_size
        
    def preprocess_text(self, text: str) -> str:
        """Preprocess text using spaCy's German model."""
        try:
            # Convert to lowercase
            text = text.lower()
            
            # Process with spaCy
            doc = nlp(text)
            
            # Lemmatize and remove stop words
            tokens = [
                token.lemma_ for token in doc 
                if not token.is_stop and not token.is_punct and token.lemma_.strip()
            ]
            
            return " ".join(tokens)
        except Exception as e:
            logger.error(f"Error preprocessing text: {str(e)}")
            return text  # Return original text if preprocessing fails
    
    def chunk_text(self, text: str, article_id: str) -> List[Dict[str, Any]]:
        """Split text into chunks of approximately chunk_size words."""
        try:
            words = text.split()
            chunks = []
            
            for i in range(0, len(words), self.chunk_size):
                chunk = " ".join(words[i:i + self.chunk_size])
                chunks.append({
                    "article_id": article_id,
                    "chunk_id": f"{article_id}_chunk_{len(chunks)}",
                    "text": chunk
                })
            
            return chunks
        except Exception as e:
            logger.error(f"Error chunking text: {str(e)}")
            return []

    def process_article(self, article: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Process a single article."""
        processed_chunks = []
        
        try:
            article_id = article.get('id')
            text = article.get('text', '')
            
            if not text or not article_id:
                return []
            
            # Preprocess text
            processed_text = self.preprocess_text(text)
            
            # Split into chunks
            chunks = self.chunk_text(processed_text, article_id)
            
            # Add metadata to each chunk
            for chunk in chunks:
                chunk.update({
                    "published_at": article.get('published_at'),
                    "author": article.get('author'),
                    "title": article.get('title'),
                    "category": article.get('category'),
                    "section": article.get('section')
                })
                processed_chunks.append(chunk)
                
        except Exception as e:
            logger.error(f"Error processing article {article_id}: {str(e)}")
            
        return processed_chunks

class VectorStore:
    def __init__(self, dimension: int):
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)
        self.metadata = []
        
    def add_embeddings(self, embeddings: np.ndarray, metadata_list: List[Dict[str, Any]]):
        """Add embeddings and metadata to the store."""
        self.index.add(embeddings)
        self.metadata.extend(metadata_list)
        
    def save(self, directory: Path):
        """Save the FAISS index and metadata."""
        directory.mkdir(exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, str(directory / "articles.index"))
        
        # Save metadata
        with open(directory / "metadata.json", 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)

def main():
    try:
        # Initialize processor and vector store
        processor = ArticleProcessor()
        vector_store = VectorStore(dimension=1536)  # Azure OpenAI embedding dimension
        
        # Read combined articles
        logger.info("Reading combined articles...")
        input_file = Path("data/articles_combined/combined_articles_20250322_144252.json")
        with open(input_file, 'r', encoding='utf-8') as f:
            articles = json.load(f)
        
        # Process articles and generate embeddings
        all_chunks = []
        logger.info("Processing articles and generating chunks...")
        for article in tqdm(articles, desc="Processing articles"):
            chunks = processor.process_article(article)
            all_chunks.extend(chunks)
        
        # Generate embeddings
        logger.info("Generating embeddings using Azure OpenAI API...")
        texts = [chunk['text'] for chunk in all_chunks]
        embeddings = get_embeddings(texts)
        
        # Add to vector store
        logger.info("Adding embeddings to vector store...")
        vector_store.add_embeddings(embeddings, all_chunks)
        
        # Save vector store
        output_dir = Path("data/vector_store")
        logger.info(f"Saving vector store to {output_dir}...")
        vector_store.save(output_dir)
        
        logger.info("Processing complete!")
        logger.info(f"Processed {len(articles)} articles into {len(all_chunks)} chunks")
        
    except Exception as e:
        logger.error(f"An error occurred in main: {str(e)}")
        raise

if __name__ == "__main__":
    main() 