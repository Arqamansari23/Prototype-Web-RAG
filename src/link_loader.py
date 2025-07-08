from langchain_community.document_loaders import RecursiveUrlLoader
from bs4 import BeautifulSoup
import yaml
import logging
import os
from datetime import datetime
import sys
from pathlib import Path

# Set UTF-8 encoding for Windows compatibility
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

# Setup logging
os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/links_logs.txt'),
        logging.StreamHandler()
    ]
)

def bs4_extractor(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    return soup.get_text()

def main():
    try:
        # Load configuration
        config_path = 'config/config.yml'
        if not os.path.exists(config_path):
            print("[ERROR] Configuration file not found. Please run the Streamlit app first!")
            logging.error("Configuration file not found")
            sys.exit(1)
        
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        
        url = config['url']
        max_depth = config['max_depth']
        
        logging.info(f"Starting link extraction from: {url}")
        logging.info(f"Max depth: {max_depth}")
        
        print(f"Starting link extraction from: {url}")
        print(f"Max depth: {max_depth}")
        
        # Configure the loader
        loader = RecursiveUrlLoader(
            url=url,
            max_depth=max_depth,
            extractor=bs4_extractor,
            use_async=False,
            timeout=10
        )
        
        # Load documents
        documents = loader.load()
        
        # Log results
        logging.info(f"Total documents fetched: {len(documents)}")
        
        print(f"\n[SUCCESS] Total documents fetched: {len(documents)}\n")
        
        # Create data directory and save links
        os.makedirs('data', exist_ok=True)
        
        links = []
        for i, doc in enumerate(documents, start=1):
            link = doc.metadata['source']
            links.append(link)
            print(f"{i}. {link}")
            logging.info(f"Document {i}: {link}")
        
        # Save links to file with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        links_file = f"data/links_{timestamp}.txt"
        
        with open(links_file, 'w') as f:
            for link in links:
                f.write(f"{link}\n")
        
        # Also save as links.txt for backward compatibility
        with open('data/links.txt', 'w') as f:
            for link in links:
                f.write(f"{link}\n")
        
        logging.info(f"Links saved to: {links_file}")
        print(f"\n[SAVED] Links saved to: {links_file}")
        print(f"Total links discovered: {len(links)}")
        logging.info("Link extraction completed successfully")
        
    except Exception as e:
        logging.error(f"Error during link extraction: {str(e)}")
        print(f"[ERROR] Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()