import pandas as pd
import json
from pathlib import Path
from datetime import datetime
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def read_metadata():
    """Read and filter the metadata CSV file."""
    try:
        logger.info("Reading metadata.csv file...")
        metadata_path = Path("data/metadata.csv")
        df = pd.read_csv(metadata_path)
        
        # Convert published_at to datetime
        df['published_at'] = pd.to_datetime(df['published_at'])
        
        # Filter for years 2022-2024
        filtered_df = df[df['published_at'].dt.year.isin([2022])]
        logger.info(f"Found {len(filtered_df)} articles from 2022")
        
        return filtered_df
    except Exception as e:
        logger.error(f"Error reading metadata: {str(e)}")
        raise

def read_json_file(filename):
    """Read a single JSON file."""
    try:
        file_path = Path("data/articles_clean") / filename
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error reading file {filename}: {str(e)}")
        return None

def combine_articles(filenames):
    """Combine multiple JSON files into one."""
    combined_articles = []
    total_files = len(filenames)
    
    for i, filename in enumerate(filenames, 1):
        if i % 100 == 0:
            logger.info(f"Processing file {i}/{total_files}")
            
        article_data = read_json_file(filename)
        if article_data:
            combined_articles.append(article_data)
    
    return combined_articles

def save_combined_articles(articles):
    """Save the combined articles to a JSON file."""
    try:
        output_dir = Path("data/articles_combined")
        output_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f"combined_articles_{timestamp}.json"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(articles, f, ensure_ascii=False, indent=2)
            
        logger.info(f"Successfully saved combined articles to {output_file}")
    except Exception as e:
        logger.error(f"Error saving combined articles: {str(e)}")
        raise

def main():
    try:
        # Read and filter metadata
        filtered_df = read_metadata()
        
        # Get list of filenames to process
        filenames = filtered_df['filename'].tolist()
        
        # Combine articles
        logger.info("Starting to combine articles...")
        combined_articles = combine_articles(filenames)
        
        # Save combined articles
        logger.info(f"Saving {len(combined_articles)} combined articles...")
        save_combined_articles(combined_articles)
        
    except Exception as e:
        logger.error(f"An error occurred in main: {str(e)}")
        raise

if __name__ == "__main__":
    main() 