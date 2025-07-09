import streamlit as st
import os
import yaml
import subprocess
import sys
from datetime import datetime
import logging
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="GeneTech Solutions ",
    page_icon="🔗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #1f77b4;
        margin-bottom: 30px;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .error-box {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .info-box {
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        color: #0c5460;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #dee2e6;
        text-align: center;
    }
    .graph-container {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #dee2e6;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

def setup_directories():
    """Create necessary directories"""
    directories = ['logs', 'data', 'config', 'src', 'data/vectorStores']
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)

def create_config_file(url, max_depth):
    """Create configuration file for the pipeline"""
    config = {
        'url': url,
        'max_depth': max_depth
    }
    
    config_path = 'config/config.yml'
    with open(config_path, 'w') as file:
        yaml.dump(config, file, default_flow_style=False)
    
    return config_path

def run_link_loader():
    """Execute the link loader pipeline"""
    try:
        # Change to src directory and run the script
        result = subprocess.run(
            [sys.executable, 'src/link_loader.py'],
            capture_output=True,
            text=True,
            check=True
        )
        return True, result.stdout, result.stderr
    except subprocess.CalledProcessError as e:
        return False, e.stdout, e.stderr
    except Exception as e:
        return False, "", str(e)

def run_cost_estimator():
    """Execute the cost estimator pipeline"""
    try:
        result = subprocess.run(
            [sys.executable, 'src/cost_estimator.py'],
            capture_output=True,
            text=True,
            check=True
        )
        return True, result.stdout, result.stderr
    except subprocess.CalledProcessError as e:
        return False, e.stdout, e.stderr
    except Exception as e:
        return False, "", str(e)

def run_data_ingestion():
    """Execute the data ingestion pipeline"""
    try:
        result = subprocess.run(
            [sys.executable, 'src/data_ingestion.py'],
            capture_output=True,
            text=True,
            check=True
        )
        return True, result.stdout, result.stderr
    except subprocess.CalledProcessError as e:
        return False, e.stdout, e.stderr
    except Exception as e:
        return False, "", str(e)

def run_graph_compile():
    """Execute the graph compile pipeline"""
    try:
        result = subprocess.run(
            [sys.executable, 'src/graph_compile.py'],
            capture_output=True,
            text=True,
            check=True
        )
        return True, result.stdout, result.stderr
    except subprocess.CalledProcessError as e:
        return False, e.stdout, e.stderr
    except Exception as e:
        return False, "", str(e)

def generate_graph_visualization():
    """Generate graph visualization using mermaid"""
    try:
        # Import the GraphBuilder from the graph_compile module
        import sys
        sys.path.append('src')
        from graph_compile import GraphBuilder
        
        # Initialize the graph builder
        graph_builder = GraphBuilder()
        
        # Get the mermaid representation
        try:
            mermaid_code = graph_builder.app.get_graph().draw_mermaid()
            return True, mermaid_code
        except Exception as e:
            # Fallback to a simple mermaid representation
            fallback_mermaid = """
            graph TD
                A[START] --> B[AI Assistant]
                B --> C{Tool Decision}
                C -->|Use Tools| D[Tools]
                C -->|No Tools| E[END]
                D --> F[Grade Documents]
                F -->|Relevant| G[Output Generator]
                F -->|Not Relevant| H[Handle Irrelevant]
                G --> E
                H --> E
            """
            return True, fallback_mermaid
    except Exception as e:
        return False, str(e)

def check_graph_ready():
    """Check if graph is ready to be built"""
    # Check if vector database exists
    vector_store_path = 'data/vectorStores/store'
    return os.path.exists(vector_store_path) and os.listdir(vector_store_path)

def read_links_file():
    """Read the generated links file"""
    data_dir = 'data'
    if not os.path.exists(data_dir):
        return []
    
    links_files = [f for f in os.listdir(data_dir) if f.startswith('links') and f.endswith('.txt')]
    if not links_files:
        return []
    
    latest_links_file = sorted(links_files)[-1]
    links_file_path = os.path.join(data_dir, latest_links_file)
    
    try:
        with open(links_file_path, 'r') as f:
            links = [line.strip() for line in f if line.strip()]
        return links
    except:
        return []

def read_cost_estimation():
    """Read the latest cost estimation file"""
    data_dir = 'data'
    if not os.path.exists(data_dir):
        return None
    
    cost_files = [f for f in os.listdir(data_dir) if f.startswith('cost_estimation') and f.endswith('.txt')]
    if not cost_files:
        return None
    
    latest_cost_file = sorted(cost_files)[-1]
    cost_file_path = os.path.join(data_dir, latest_cost_file)
    
    try:
        with open(cost_file_path, 'r') as f:
            content = f.read()
        return content
    except:
        return None

def check_vector_database_exists():
    """Check if vector database exists"""
    vector_store_path = 'data/vectorStores/store'
    return os.path.exists(vector_store_path) and os.listdir(vector_store_path)

def check_graph_compiled():
    """Check if graph has been compiled by looking for log files or other indicators"""
    # Check for graph compilation logs
    graph_log_path = 'logs/graph_builder.log'
    if os.path.exists(graph_log_path):
        try:
            with open(graph_log_path, 'r') as f:
                content = f.read()
                if 'GraphBuilder initialized successfully' in content:
                    return True
        except:
            pass
    return False

def initialize_chat_system():
    """Initialize the chat system with the built graph"""
    try:
        # Import the GraphBuilder from the graph_compile module
        import sys
        sys.path.append('src')
        from graph_compile import GraphBuilder
        
        # Initialize the graph builder
        if 'graph_builder' not in st.session_state:
            with st.spinner("🔄 Initializing chat system..."):
                st.session_state.graph_builder = GraphBuilder()
                st.session_state.chat_initialized = True
        
        return True, "Chat system initialized successfully!"
    except Exception as e:
        return False, f"Error initializing chat system: {str(e)}"

def process_chat_query(query: str):
    """Process a chat query through the graph"""
    try:
        if 'graph_builder' not in st.session_state:
            return "Error: Chat system not initialized. Please initialize the chat system first."
        
        # Process the query through the graph
        response = st.session_state.graph_builder.query(query)
        return response
    except Exception as e:
        return f"Error processing query: {str(e)}"

def clear_chat_history():
    """Clear the chat history"""
    if 'chat_messages' in st.session_state:
        st.session_state.chat_messages = []
    if 'chat_history' in st.session_state:
        st.session_state.chat_history = []

def main():
    # Setup
    setup_directories()
    
    # Header
    st.markdown('<h1 class="main-header">🔗 Web Scraping Cost Estimator</h1>', unsafe_allow_html=True)
    
    # Sidebar for configuration
    st.sidebar.header("⚙️ Configuration")
    
    # Input fields
    url = st.sidebar.text_input(
        "🌐 Enter Website URL",
        placeholder="https://example.com",
        help="Enter the base URL to scrape"
    )
    
    max_depth = st.sidebar.number_input(
        "🔍 Max Depth",
        min_value=1,
        max_value=10,
        value=2,
        help="Maximum depth for recursive crawling"
    )
    
    # Validation
    url_valid = url and url.startswith(('http://', 'https://'))
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("📊 Pipeline Execution")
        
        # Step 1: Link Loading
        st.subheader("Step 1: Link Discovery")
        
        if st.button("🚀 Start Link Loading", disabled=not url_valid, type="primary"):
            if not url_valid:
                st.error("Please enter a valid URL starting with http:// or https://")
            else:
                # Create config file
                config_path = create_config_file(url, max_depth)
                
                with st.spinner("🔍 Discovering links..."):
                    # Initialize session state
                    if 'link_loading_complete' not in st.session_state:
                        st.session_state.link_loading_complete = False
                    
                    # Run link loader
                    success, stdout, stderr = run_link_loader()
                    
                    if success:
                        st.session_state.link_loading_complete = True
                        st.markdown('<div class="success-box">✅ Link loading completed successfully!</div>', unsafe_allow_html=True)
                        
                        # Display results
                        if stdout:
                            st.text_area("📋 Link Loader Output", value=stdout, height=200)
                        
                        # Read and display links
                        links = read_links_file()
                        if links:
                            st.success(f"📄 Total links discovered: {len(links)}")
                            
                            with st.expander("🔗 View All Links"):
                                for i, link in enumerate(links, 1):
                                    st.write(f"{i}. {link}")
                    else:
                        st.markdown('<div class="error-box">❌ Link loading failed!</div>', unsafe_allow_html=True)
                        if stderr:
                            st.error(f"Error: {stderr}")
                        if stdout:
                            st.text_area("Output", value=stdout, height=100)
        
        # Step 2: Cost Estimation
        st.subheader("Step 2: Cost Estimation")
        
        # Check if links are available
        links = read_links_file()
        links_available = len(links) > 0
        
        if not links_available:
            st.info("⚠️ Please complete link loading first to proceed with cost estimation.")
        
        if st.button("💰 Calculate Data Ingestion Cost", disabled=not links_available, type="secondary"):
            if not links_available:
                st.error("No links found. Please run link loading first.")
            else:
                with st.spinner("💵 Calculating costs..."):
                    success, stdout, stderr = run_cost_estimator()
                    
                    if success:
                        st.markdown('<div class="success-box">✅ Cost estimation completed successfully!</div>', unsafe_allow_html=True)
                        
                        # Display results
                        if stdout:
                            st.text_area("📊 Cost Estimation Results", value=stdout, height=200)
                        
                        # Read and display detailed cost estimation
                        cost_content = read_cost_estimation()
                        if cost_content:
                            st.markdown("### 📈 Detailed Cost Report")
                            st.text_area("Cost Report", value=cost_content, height=300)
                    else:
                        st.markdown('<div class="error-box">❌ Cost estimation failed!</div>', unsafe_allow_html=True)
                        if stderr:
                            st.error(f"Error: {stderr}")
                        if stdout:
                            st.text_area("Output", value=stdout, height=100)
        
        # Step 3: Data Ingestion
        st.subheader("Step 3: Data Ingestion")
        
        # Check if cost estimation is available
        cost_content = read_cost_estimation()
        cost_available = cost_content is not None
        
        if not cost_available:
            st.info("⚠️ Please complete cost estimation first to proceed with data ingestion.")
        
        if st.button("📁 Start Data Ingestion", disabled=not cost_available, type="secondary"):
            if not cost_available:
                st.error("No cost estimation found. Please run cost estimation first.")
            else:
                with st.spinner("📊 Ingesting data and creating vector database..."):
                    success, stdout, stderr = run_data_ingestion()
                    
                    if success:
                        st.markdown('<div class="success-box">✅ Successfully Created Vector Database in data/vectorStores/store and data ingested successfully!</div>', unsafe_allow_html=True)
                        
                        # Display results
                        if stdout:
                            st.text_area("📁 Data Ingestion Output", value=stdout, height=200)
                            
                            # Parse additional stats from output
                            lines = stdout.split('\n')
                            for line in lines:
                                if 'Documents processed:' in line:
                                    st.info(f"📄 {line.strip()}")
                                elif 'Total chunks created:' in line:
                                    st.info(f"📝 {line.strip()}")
                                elif 'FAISS index saved:' in line:
                                    st.info(f"🗂️ {line.strip()}")
                        
                        # Check if vector database was created
                        if check_vector_database_exists():
                            st.success("🎉 Vector database created successfully!")
                            st.info("📍 Location: data/vectorStores/store")
                        else:
                            st.warning("⚠️ Vector database directory not found, but process completed successfully.")
                    else:
                        st.markdown('<div class="error-box">❌ Data ingestion failed!</div>', unsafe_allow_html=True)
                        if stderr:
                            st.error(f"Error: {stderr}")
                        if stdout:
                            st.text_area("Output", value=stdout, height=100)
        
        # Step 4: Graph Building
        st.subheader("Step 4: Graph Building")
        
        # Check if graph is ready to be built
        graph_ready = check_graph_ready()
        
        if not graph_ready:
            st.info("⚠️ Please complete data ingestion first to proceed with graph building.")
        
        if st.button("🔧 Build Graph", disabled=not graph_ready, type="secondary"):
            if not graph_ready:
                st.error("Vector database not found. Please run data ingestion first.")
            else:
                with st.spinner("🔧 Building graph..."):
                    success, stdout, stderr = run_graph_compile()
                    
                    if success:
                        st.markdown('<div class="success-box">✅ Graph Build Successfully!</div>', unsafe_allow_html=True)
                        
                        # Display results
                        if stdout:
                            st.text_area("🔧 Graph Building Output", value=stdout, height=200)
                        
                        # Store in session state that graph is built
                        st.session_state.graph_built = True
                        
                        # Show success message and graph info
                        st.success("🎉 Agentic RAG Graph compiled successfully!")
                        st.info("📍 Graph is ready for query processing")
                        
                    else:
                        st.markdown('<div class="error-box">❌ Graph building failed!</div>', unsafe_allow_html=True)
                        if stderr:
                            st.error(f"Error: {stderr}")
                        if stdout:
                            st.text_area("Output", value=stdout, height=100)
        
        # Show Graph Button
        if st.session_state.get('graph_built', False) or check_graph_compiled():
            if st.button("📊 Show Graph", type="primary"):
                st.markdown("### 🗺️ Graph Visualization")
                
                with st.spinner("Generating graph visualization..."):
                    success, mermaid_code = generate_graph_visualization()
                    
                    if success:
                        st.markdown('<div class="graph-container">', unsafe_allow_html=True)
                        st.markdown("**Agentic RAG Workflow Graph:**")
                        
                        # Display mermaid diagram
                        try:
                            import streamlit.components.v1 as components
                            
                            # Create HTML with mermaid
                            mermaid_html = f"""
                            <div class="mermaid">
                                {mermaid_code}
                            </div>
                            <script src="https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js"></script>
                            <script>
                                mermaid.initialize({{startOnLoad: true}});
                            </script>
                            """
                            
                            components.html(mermaid_html, height=600)
                            
                        except Exception as e:
                            # Fallback to text representation
                            st.code(mermaid_code, language="text")
                        
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        # Additional graph information
                        st.markdown("### 📋 Graph Components")
                        st.markdown("""
                        - **AI Assistant**: Main decision-making node
                        - **Tools**: Retrieval and processing tools
                        - **Grade Documents**: Document relevance checker
                        - **Output Generator**: Final response generator
                        - **Handle Irrelevant**: Handles irrelevant queries
                        """)
                        
                    else:
                        st.error(f"Failed to generate graph visualization: {mermaid_code}")
        
        # Step 5: Chat Interface
        st.subheader("Step 5: Chat with Your Data")
        
        # Check if graph is compiled and ready for chat
        graph_compiled = check_graph_compiled() or st.session_state.get('graph_built', False)
        
        if not graph_compiled:
            st.info("⚠️ Please complete graph building first to start chatting.")
        else:
            # Initialize chat system button
            if not st.session_state.get('chat_initialized', False):
                if st.button("💬 Initialize Chat System", type="primary"):
                    success, message = initialize_chat_system()
                    if success:
                        st.success(message)
                        st.rerun()
                    else:
                        st.error(message)
            
            # Chat interface
            if st.session_state.get('chat_initialized', False):
                st.markdown("### 💬 Chat Interface")
                
                # Initialize chat messages in session state
                if 'chat_messages' not in st.session_state:
                    st.session_state.chat_messages = []
                
                # Display chat history
                chat_container = st.container()
                with chat_container:
                    for message in st.session_state.chat_messages:
                        if message["role"] == "user":
                            st.markdown(f"**You:** {message['content']}")
                        else:
                            st.markdown(f"**Assistant:** {message['content']}")
                        st.markdown("---")
                
                # Chat input area
                col_input, col_send, col_clear = st.columns([6, 1, 1])
                
                with col_input:
                    user_input = st.text_input(
                        "Ask anything about your data:",
                        key="chat_input",
                        placeholder="e.g., What services does the company offer?"
                    )
                
                with col_send:
                    send_button = st.button("Send", type="primary")
                
                with col_clear:
                    clear_button = st.button("Clear")
                
                # Process user input
                if send_button and user_input:
                    # Add user message to chat history
                    st.session_state.chat_messages.append({"role": "user", "content": user_input})
                    
                    # Process the query
                    with st.spinner("🤔 Processing your query..."):
                        response = process_chat_query(user_input)
                    
                    # Add assistant response to chat history
                    st.session_state.chat_messages.append({"role": "assistant", "content": response})
                    
                    # Clear the input and rerun to show new messages
                    st.rerun()
                
                # Clear chat history
                if clear_button:
                    clear_chat_history()
                    st.rerun()
                
                # Sample questions
                if not st.session_state.chat_messages:
                    st.markdown("### 💡 Sample Questions")
                    sample_questions = [
                        "What services does the company offer?",
                        "Tell me about the company's history",
                        "What are the contact details?",
                        "Who are the key clients?",
                        "What technologies does the company use?"
                    ]
                    
                    for question in sample_questions:
                        if st.button(f"📝 {question}", key=f"sample_{question}"):
                            st.session_state.chat_messages.append({"role": "user", "content": question})
                            
                            with st.spinner("🤔 Processing your query..."):
                                response = process_chat_query(question)
                            
                            st.session_state.chat_messages.append({"role": "assistant", "content": response})
                            st.rerun()
                
                # Chat statistics
                if st.session_state.chat_messages:
                    st.markdown("### 📊 Chat Statistics")
                    user_messages = len([msg for msg in st.session_state.chat_messages if msg["role"] == "user"])
                    assistant_messages = len([msg for msg in st.session_state.chat_messages if msg["role"] == "assistant"])
                    
                    col_stat1, col_stat2 = st.columns(2)
                    with col_stat1:
                        st.metric("User Messages", user_messages)
                    with col_stat2:
                        st.metric("Assistant Responses", assistant_messages)
    
    with col2:
        st.header("📋 Status")
        
        # Current configuration
        st.markdown("### ⚙️ Current Config")
        st.markdown(f"**URL:** {url if url else 'Not set'}")
        st.markdown(f"**Max Depth:** {max_depth}")
        st.markdown(f"**Status:** {'✅ Valid' if url_valid else '❌ Invalid URL'}")
        
        # Links status
        st.markdown("### 🔗 Links Status")
        links = read_links_file()
        if links:
            st.markdown(f"**Total Links:** {len(links)}")
            st.markdown("**Status:** ✅ Ready for cost estimation")
        else:
            st.markdown("**Status:** ⏳ No links discovered yet")
        
        # Cost estimation status
        st.markdown("### 💰 Cost Status")
        cost_content = read_cost_estimation()
        if cost_content:
            # Parse cost from content
            lines = cost_content.split('\n')
            for line in lines:
                if 'Estimated cost:' in line and '$' in line:
                    cost = line.split('$')[1].strip()
                    st.markdown(f"**Latest Cost:** ${cost}")
                    break
            st.markdown("**Status:** ✅ Cost calculated")
        else:
            st.markdown("**Status:** ⏳ No cost estimation yet")
        
        # Data ingestion status
        st.markdown("### 📁 Data Ingestion Status")
        vector_db_exists = check_vector_database_exists()
        if vector_db_exists:
            st.markdown("**Vector Database:** ✅ Created")
            st.markdown("**Location:** data/vectorStores/store")
            st.markdown("**Status:** ✅ Data ingested successfully")
            
            # Try to show additional stats if available
            logs_path = 'logs/data_ingestion.txt'
            if os.path.exists(logs_path):
                try:
                    with open(logs_path, 'r') as f:
                        log_content = f.read()
                        if 'Final stats' in log_content:
                            lines = log_content.split('\n')
                            for line in lines:
                                if 'Final stats' in line:
                                    st.markdown(f"**Stats:** {line.split('Final stats - ')[1]}")
                                    break
                except:
                    pass
        else:
            st.markdown("**Status:** ⏳ No vector database created yet")
        
        # Graph building status
        st.markdown("### 🔧 Graph Status")
        graph_compiled = check_graph_compiled()
        if graph_compiled or st.session_state.get('graph_built', False):
            st.markdown("**Graph:** ✅ Built successfully")
            st.markdown("**Status:** ✅ Ready for queries")
            st.markdown("**Type:** Agentic RAG Workflow")
        else:
            st.markdown("**Status:** ⏳ No graph built yet")
        
        # Chat status
        st.markdown("### 💬 Chat Status")
        chat_initialized = st.session_state.get('chat_initialized', False)
        if chat_initialized:
            st.markdown("**Chat System:** ✅ Initialized")
            st.markdown("**Status:** ✅ Ready for queries")
            
            # Show chat message count
            chat_messages = st.session_state.get('chat_messages', [])
            if chat_messages:
                user_msgs = len([msg for msg in chat_messages if msg["role"] == "user"])
                st.markdown(f"**Messages:** {user_msgs} queries")
        else:
            if check_graph_compiled() or st.session_state.get('graph_built', False):
                st.markdown("**Status:** ⏳ Ready to initialize")
            else:
                st.markdown("**Status:** ⏳ Waiting for graph")
        
        # Quick stats
        if links:
            st.markdown("### 📊 Quick Stats")
            st.markdown(f"🔗 **Links Found:** {len(links)}")
            
            # Show recent files
            data_dir = 'data'
            if os.path.exists(data_dir):
                files = os.listdir(data_dir)
                if files:
                    st.markdown("### 📁 Recent Files")
                    for file in sorted(files)[-3:]:  # Show last 3 files
                        st.markdown(f"📄 {file}")
    
    # Footer
    st.markdown("---")
    st.markdown("### 📝 Instructions")
    st.markdown("""
    1. **Enter URL**: Provide the base website URL you want to scrape
    2. **Set Max Depth**: Choose how deep to crawl (1 = single page, 2 = one level deep, etc.)
    3. **Load Links**: Click 'Start Link Loading' to discover all URLs
    4. **Calculate Cost**: Once links are loaded, click 'Calculate Data Ingestion Cost'
    5. **Ingest Data**: After cost estimation, click 'Start Data Ingestion' to create vector database
    6. **Build Graph**: Once data is ingested, click 'Build Graph' to create the agentic RAG workflow
    7. **Show Graph**: Click 'Show Graph' to visualize the workflow diagram
    8. **Review Results**: Check the detailed reports and graph visualization
    """)
    
    # Debug information (collapsible)
    with st.expander("🔧 Debug Information"):
        st.markdown("**Current Working Directory:**")
        st.code(os.getcwd())
        
        st.markdown("**Directory Structure:**")
        for root, dirs, files in os.walk('.'):
            level = root.replace('.', '').count(os.sep)
            indent = ' ' * 2 * level
            st.text(f"{indent}{os.path.basename(root)}/")
            subindent = ' ' * 2 * (level + 1)
            for file in files:
                st.text(f"{subindent}{file}")
            if level > 2:  # Limit depth for readability
                break

if __name__ == "__main__":
    main()