
import os
import logging
from datetime import datetime
import sys
from langchain_community.document_loaders import RecursiveUrlLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from bs4 import BeautifulSoup
from dotenv import load_dotenv  
# Load environment variables
load_dotenv()

# Setup logging
os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/data_ingestion.txt'),
        logging.StreamHandler()
    ]
)




# Set UTF-8 encoding for Windows compatibility
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

def bs4_extractor(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    return soup.get_text()

# Initialize OpenAI embeddings
embedding_model = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=os.environ.get("OPENAI_API_KEY"),
    max_retries=0
)

# Initialize text splitter for chunking
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000,
    chunk_overlap=200,
    length_function=len,
)

# Get the latest links file
data_dir = 'data'
if not os.path.exists(data_dir):
    print("❌ No data directory found. Run link_loader.py first!")
    logging.error("No data directory found")
    exit(1)

links_files = [f for f in os.listdir(data_dir) if f.startswith('links') and f.endswith('.txt')]
if not links_files:
    print("❌ No links files found. Run link_loader.py first!")
    logging.error("No links files found")
    exit(1)

# Use the latest links file
latest_links_file = sorted(links_files)[-1]
links_file_path = os.path.join(data_dir, latest_links_file)

print(f"📊 Using links file: {latest_links_file}")
logging.info(f"Using links file: {latest_links_file}")

# Read links from file
with open(links_file_path, 'r') as f:
    links = [line.strip() for line in f if line.strip()]

print(f"🔗 Total links to process: {len(links)}")
logging.info(f"Total links to process: {len(links)}")

# Initialize vector store
vector_store = None
processed_docs = 0
total_chunks = 0

try:
    for i, url in enumerate(links, 1):
        print(f"Processing {i}/{len(links)}: {url}")
        logging.info(f"Processing document {i}: {url}")
        
        try:
            # Load single URL
            loader = RecursiveUrlLoader(
                url=url,
                max_depth=1,
                extractor=bs4_extractor,
                use_async=False,
                timeout=10
            )
            
            docs = loader.load()
            
            for doc in docs:
                if doc.metadata['source'] == url:
                    # Split document into chunks
                    chunks = text_splitter.split_text(doc.page_content)
                    
                    if chunks:
                        # Create or add to vector store
                        if vector_store is None:
                            # Create new vector store with first batch of chunks
                            vector_store = FAISS.from_texts(
                                texts=chunks,
                                embedding=embedding_model
                            )
                        else:
                            # Add chunks to existing vector store
                            chunk_embeddings = FAISS.from_texts(
                                texts=chunks,
                                embedding=embedding_model
                            )
                            vector_store.merge_from(chunk_embeddings)
                        
                        processed_docs += 1
                        total_chunks += len(chunks)
                        
                        print(f"✅ Document {i} processed: {len(chunks)} chunks")
                        logging.info(f"Document {i} added: {len(chunks)} chunks")
                    break
                    
        except Exception as e:
            print(f"⚠️  Error processing {url}: {str(e)}")
            logging.error(f"Error processing {url}: {str(e)}")
            continue
    
    if vector_store is not None:
        # Save FAISS index
        faiss_index_path = "data/vectorStores/store"
        os.makedirs(os.path.dirname(faiss_index_path), exist_ok=True)
        
        vector_store.save_local(faiss_index_path)
        
        print(f"✅ FAISS vector store created and saved to: {faiss_index_path}")
        logging.info(f"FAISS vector store saved to: {faiss_index_path}")
        
        # Summary
        print(f"\n📈 DATA INGESTION COMPLETED")
        print(f"=" * 40)
        print(f"📄 Documents processed: {processed_docs}/{len(links)}")
        print(f"📝 Total chunks created: {total_chunks}")
        print(f"🗂️  FAISS index saved: {faiss_index_path}")
        
        logging.info("Data ingestion completed successfully")
        logging.info(f"Final stats - Documents: {processed_docs}, Chunks: {total_chunks}")
    else:
        print("❌ No documents were successfully processed")
        logging.error("No documents were successfully processed")

except Exception as e:
    logging.error(f"Error during data ingestion: {str(e)}")
    print(f"❌ Error: {str(e)}")