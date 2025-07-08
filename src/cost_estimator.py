import os
import logging
import tiktoken
from datetime import datetime
from langchain_community.document_loaders import RecursiveUrlLoader
from bs4 import BeautifulSoup
import sys

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
        logging.FileHandler('logs/cost_estimation_logs.txt'),
        logging.StreamHandler()
    ]
)

def bs4_extractor(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    return soup.get_text()

def count_tokens(text: str) -> int:
    """Count tokens using tiktoken for OpenAI models"""
    try:
        encoding = tiktoken.encoding_for_model("text-embedding-3-small")
        return len(encoding.encode(text))
    except Exception as e:
        logging.error(f"Error counting tokens: {str(e)}")
        # Fallback: rough estimation (4 chars per token)
        return len(text) // 4

def estimate_embedding_cost(total_tokens: int) -> float:
    """
    Calculate cost for text-embedding-3-small model
    Pricing: $0.00002 per 1K tokens
    """
    cost_per_1k_tokens = 0.00002
    return (total_tokens / 1000) * cost_per_1k_tokens

def main():
    try:
        # Get the latest links file
        data_dir = 'data'
        if not os.path.exists(data_dir):
            print("[ERROR] No data directory found. Run link_loader.py first!")
            logging.error("No data directory found")
            sys.exit(1)
        
        links_files = [f for f in os.listdir(data_dir) if f.startswith('links') and f.endswith('.txt')]
        if not links_files:
            print("[ERROR] No links files found. Run link_loader.py first!")
            logging.error("No links files found")
            sys.exit(1)
        
        # Use the latest links file
        latest_links_file = sorted(links_files)[-1]
        links_file_path = os.path.join(data_dir, latest_links_file)
        
        print(f"[INFO] Using links file: {latest_links_file}")
        logging.info(f"Using links file: {latest_links_file}")
        
        # Read links from file
        with open(links_file_path, 'r') as f:
            links = [line.strip() for line in f if line.strip()]
        
        print(f"[INFO] Total links to process: {len(links)}")
        logging.info(f"Total links to process: {len(links)}")
        
        # Load documents and calculate tokens
        total_tokens = 0
        processed_docs = 0
        failed_docs = 0
        
        for i, url in enumerate(links, 1):
            print(f"Processing {i}/{len(links)}: {url}")
            logging.info(f"Processing document {i}: {url}")
            
            try:
                # Load single URL
                loader = RecursiveUrlLoader(
                    url=url,
                    max_depth=1,  # Just the single page
                    extractor=bs4_extractor,
                    use_async=False,
                    timeout=10
                )
                
                documents = loader.load()
                
                for doc in documents:
                    if doc.metadata['source'] == url:  # Only count the main URL
                        doc_tokens = count_tokens(doc.page_content)
                        total_tokens += doc_tokens
                        processed_docs += 1
                        
                        logging.info(f"Document {i} tokens: {doc_tokens}")
                        print(f"  [SUCCESS] Tokens: {doc_tokens:,}")
                        break
                        
            except Exception as e:
                print(f"  [WARNING] Error processing {url}: {str(e)}")
                logging.error(f"Error processing {url}: {str(e)}")
                failed_docs += 1
                continue
        
        # Calculate cost
        estimated_cost = estimate_embedding_cost(total_tokens)
        
        # Results
        print(f"\n{'='*50}")
        print(f"COST ESTIMATION RESULTS")
        print(f"{'='*50}")
        print(f"Documents processed: {processed_docs}/{len(links)}")
        print(f"Failed documents: {failed_docs}")
        print(f"Total tokens: {total_tokens:,}")
        print(f"Estimated cost: ${estimated_cost:.6f}")
        print(f"Estimated cost (rounded): ${estimated_cost:.2f}")
        
        # Log results
        logging.info(f"Cost estimation completed")
        logging.info(f"Documents processed: {processed_docs}/{len(links)}")
        logging.info(f"Failed documents: {failed_docs}")
        logging.info(f"Total tokens: {total_tokens}")
        logging.info(f"Estimated cost: ${estimated_cost:.6f}")
        
        # Save cost estimation
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        cost_file = f"data/cost_estimation_{timestamp}.txt"
        
        with open(cost_file, 'w') as f:
            f.write(f"Cost Estimation Report - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"=" * 50 + "\n")
            f.write(f"Model: text-embedding-3-small\n")
            f.write(f"Pricing: $0.00002 per 1K tokens\n")
            f.write(f"Links file: {latest_links_file}\n")
            f.write(f"Total links: {len(links)}\n")
            f.write(f"Documents processed: {processed_docs}\n")
            f.write(f"Failed documents: {failed_docs}\n")
            f.write(f"Success rate: {(processed_docs/len(links)*100):.1f}%\n")
            f.write(f"Total tokens: {total_tokens:,}\n")
            f.write(f"Estimated cost: ${estimated_cost:.6f}\n")
            f.write(f"Estimated cost (rounded): ${estimated_cost:.2f}\n")
            f.write(f"\nDetailed Breakdown:\n")
            f.write(f"- Cost per 1K tokens: ${0.00002}\n")
            f.write(f"- Total tokens: {total_tokens:,}\n")
            f.write(f"- Calculation: ({total_tokens:,} ÷ 1000) × $0.00002 = ${estimated_cost:.6f}\n")
        
        print(f"\n[SAVED] Cost estimation saved to: {cost_file}")
        logging.info(f"Cost estimation saved to: {cost_file}")
        
    except Exception as e:
        logging.error(f"Error during cost estimation: {str(e)}")
        print(f"[ERROR] Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()