import os
import json
from typing import Annotated, Sequence, Literal, Dict, List, Any, Optional
from typing_extensions import TypedDict
from pathlib import Path
from datetime import datetime
import logging

# LangChain imports
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import Tool
from langchain.tools.retriever import create_retriever_tool
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# LangGraph imports
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from load_dotenv import load_dotenv 
# Load environment variables
load_dotenv()



import sys

# Set UTF-8 encoding for Windows compatibility
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')





### ___________Logging _____________________________________

def setup_logging(log_file_path: str = "logs/graph_builder.log"):
    """Setup logging for graph builder."""
    log_dir = Path(log_file_path).parent
    log_dir.mkdir(parents=True, exist_ok=True)
    
    logger = logging.getLogger(f"GraphBuilder_{id(log_file_path)}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    
    file_handler = logging.FileHandler(log_file_path, mode='a', encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.propagate = False
    
    return logger

# _________Agent State_______________________

class AgentState(TypedDict):
    """State for the RAG agent workflow."""
    messages: Annotated[Sequence[BaseMessage], add_messages]
    current_category: Optional[str]
    retrieved_docs: Optional[str]

# ________________   Grade ___________________ 

class Grade(BaseModel):
    """Document relevance grading model."""
    binary_score: str = Field(description="Relevance score 'yes' or 'no'")

class GraphBuilder:
    """
    Dynamic Agentic RAG Graph Builder.
    Uses a single vector store for all document retrieval.
    """
    
    def __init__(self):
        """Initialize GraphBuilder with default configuration."""
        
        # Setup logging
        self.logger = setup_logging()
        self.logger.info("🚀 GraphBuilder initialized successfully")
        
        # Set up paths
        self.vector_store_path = Path("data/vectorStores/store")
        
        # Initialize embeddings
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        
        # Initialize LLM
        self.llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.3, max_tokens=500)
        
        # Components for single vector store
        self.vector_store = None
        self.retriever = None
        self.tools = []
        
        # Load the single vector store
        self._load_vector_store()
        
        # Create retrieval tool
        self._create_retrieval_tool()
        
        # Create greeting and feedback tool
        self._create_greeting_feedback_tool()
        
        # Build the graph
        self.app = self._build_graph()
    
    def _load_vector_store(self):
        """Load the single vector store from data/vectorStore/store."""
        try:
            if not self.vector_store_path.exists():
                self.logger.warning(f"⚠️ Vector store path does not exist: {self.vector_store_path}")
                return
            
            # Load the single vector store
            self.vector_store = FAISS.load_local(
                str(self.vector_store_path), 
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            
            self.logger.info(f"✅ Loaded vector store from: {self.vector_store_path}")
            
        except Exception as e:
            self.logger.error(f"❌ Error loading vector store: {e}")
            raise
    
    def _create_retrieval_tool(self):
        """Create a single retrieval tool for the vector store."""
        if not self.vector_store:
            self.logger.warning("⚠️ No vector store available to create tool")
            return
        
        try:
            # Create retriever with fixed parameters
            self.retriever = self.vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 3}  # Fixed value instead of config
            )
            
            # Create retrieval tool with comprehensive description
            tool = create_retriever_tool(
                self.retriever,
                "retrieve_company_info",
                "Search and retrieve comprehensive information about the company including: company details, history, mission, vision, services, offerings, portfolio, contact information, addresses, phone numbers, emails, products, product features, specifications, clients, testimonials, client relationships, job openings, hiring processes, career opportunities, projects, case studies, and any other company-related information."
            )
            
            self.tools.append(tool)
            
            self.logger.info("✅ Created comprehensive retrieval tool")
            
        except Exception as e:
            self.logger.error(f"❌ Error creating retrieval tool: {e}")
    
    def _create_greeting_feedback_tool(self):
        """Create greeting and feedback handling tool."""
        def greeting_feedback_handler(user_message: str) -> str:
            """
            Handle greeting and feedback queries with LLM-based responses.
            """
            print("---GREETING AND FEEDBACK HANDLER---")
            
            # Use LLM-based prompt for greeting and feedback detection
            prompt = PromptTemplate(
                template="""You are a customer service representative for Genetech Solution company.
                            
                            Analyze the user's message and determine if it's a greeting or feedback-related query.
                            
                            User's message: {user_message}
                            
                            Instructions:
                            1. If the message contains greetings (hi, hello, hey, good morning, etc.), respond with:
                               "Hello! This is Genetech Solution. How can I help you today?"
                            
                            2. If the message is about feedback, reviews, ratings, opinions, suggestions, or comments, respond with:
                               "Thank you for wanting to provide feedback! We appreciate your input. Please share your thoughts about our services, and we'll make sure to consider your suggestions for improvement."
                            
                            3. For any other general conversational starters, respond with:
                               "This is Genetech Solution. How can I help you today?"
                            
                            Provide only the appropriate response without any additional explanation.""",
                input_variables=["user_message"]
            )
            
            chain = prompt | self.llm
            llm_response = chain.invoke({"user_message": user_message})
            
            response_content = llm_response.content if hasattr(llm_response, 'content') else str(llm_response)
            
            return response_content
        
        # Create the greeting and feedback tool
        greeting_tool = Tool(
            name="greeting_and_feedback_handler",
            description="Use this tool for greeting messages (hi, hello, hey, good morning, etc.) and feedback-related queries (reviews, ratings, opinions, suggestions, comments). This tool handles conversational starters and feedback requests.",
            func=greeting_feedback_handler
        )
        
        self.tools.append(greeting_tool)
        self.logger.info("✅ Created greeting and feedback tool")
    
    def _build_graph(self) -> StateGraph:
        """Build the agentic RAG workflow graph."""
        if not self.tools:
            self.logger.error("❌ No tools available to build graph")
            raise ValueError("No retrieval tools available")
        
        # Create tool node
        tool_node = ToolNode(self.tools)
        
        # Create workflow
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("AI_Assistant", self._ai_assistant)
        workflow.add_node("Tools", tool_node)
        workflow.add_node("Output_Generator", self._generate_response)
        workflow.add_node("Handle_Irrelevant", self._handle_irrelevant_query)
        
        # Add edges
        workflow.add_edge(START, "AI_Assistant")
        
        # Conditional edge from assistant
        workflow.add_conditional_edges(
            "AI_Assistant",
            tools_condition,
            {
                "tools": "Tools",
                END: END,
            }
        )
        
        # Grade documents after tool execution
        workflow.add_conditional_edges(
            "Tools",
            self._grade_documents,
            {
                "Output_Generator": "Output_Generator",
                "Handle_Irrelevant": "Handle_Irrelevant"
            }
        )
        
        # Final edges
        workflow.add_edge("Output_Generator", END)
        workflow.add_edge("Handle_Irrelevant", END)
        
        # Compile workflow
        app = workflow.compile()
        
        self.logger.info("✅ RAG workflow graph built successfully")
        return app
    
    def _ai_assistant(self, state: AgentState):
        """AI assistant node that decides which tools to use."""
        print("---CALL AI ASSISTANT---")
        messages = state['messages']
        
        # Bind tools to LLM
        llm_with_tools = self.llm.bind_tools(self.tools)
        
        # Get response
        response = llm_with_tools.invoke(messages)
        
        return {"messages": [response]}
    
    def _grade_documents(self, state: AgentState) -> Literal["Output_Generator", "Handle_Irrelevant"]:
        """Grade document relevance."""
        print("---GRADE DOCUMENTS---")
        
        # Get the last message to check if it's from greeting tool
        messages = state["messages"]
        last_message = messages[-1]
        
        # If the last message is from the greeting tool, skip grading
        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            tool_call = last_message.tool_calls[0]
            if tool_call['name'] == 'greeting_and_feedback_handler':
                print("---DECISION: GREETING/FEEDBACK RESPONSE - SKIP GRADING---")
                return "Output_Generator"
        
        # Check if the message content is a direct response (not retrieval results)
        if isinstance(last_message.content, str) and not last_message.content.startswith("Retrieved"):
            print("---DECISION: DIRECT RESPONSE - SKIP GRADING---")
            return "Output_Generator"
        
        # Create structured output LLM for document grading
        llm_with_structure = self.llm.with_structured_output(Grade)
        
        # Create grading prompt
        prompt = PromptTemplate(
            template="""You are a grader assessing relevance of a retrieved document to a user question.
            Here is the retrieved document: {context}
            Here is the user question: {question}
            
            If the document contains information relevant to the user question, grade it as relevant.
            Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.""",
            input_variables=["context", "question"]
        )
        
        # Create chain
        chain = prompt | llm_with_structure
        
        question = messages[0].content
        
        # Get document content
        docs = last_message.content
        
        # Grade documents
        scored_result = chain.invoke({"question": question, "context": docs})
        score = scored_result.binary_score
        
        if score == "yes":
            print("---DECISION: DOCS RELEVANT---")
            return "Output_Generator"
        else:
            print("---DECISION: DOCS NOT RELEVANT---")
            return "Handle_Irrelevant"
    
    def _generate_response(self, state: AgentState):
        """Generate final response using retrieved documents or direct responses."""
        print("---GENERATE RESPONSE---")
        messages = state["messages"]
        
        question = messages[0].content
        last_message = messages[-1]
        
        # Check if it's a greeting/feedback response that doesn't need RAG processing
        if isinstance(last_message.content, str) and any(phrase in last_message.content.lower() for phrase in 
                                                        ["hello! this is genetech solution", 
                                                         "thank you for wanting to provide feedback",
                                                         "this is genetech solution"]):
            print("---DIRECT GREETING/FEEDBACK RESPONSE---")
            return {"messages": [AIMessage(content=last_message.content)]}
        
        # For retrieval-based responses, process with RAG
        docs = last_message.content
        
        # Use custom prompt for Genetech Solution
        prompt = PromptTemplate(
            template="""You are Genetech Solution chatbot. Answer queries based on the provided context.
            
                    Use the following pieces of retrieved context to answer the question.
                    If you don't know the answer, just say that you don't know.
                    Provide detailed answers with appropriate headings to organize the information.
                    Start your answer with "Genetech Solution:"

                    Format your response with clear headings like:
                    - ## Overview
                    - ## Key Points
                    - ## Technical Details
                    - ## Applications
                    - ## Recommendations

                    Context: {context}
                    Question: {question}

                    Answer:""",
            input_variables=["context", "question"]
        )
        
        # Create RAG chain
        rag_chain = prompt | self.llm
        
        # Generate response
        response = rag_chain.invoke({"context": docs, "question": question})
        
        print(f"Generated response: {response}")
        
        return {"messages": [response]}
    
    def _handle_irrelevant_query(self, state: AgentState):
        """Handle queries with irrelevant documents."""
        print("---HANDLE IRRELEVANT QUERY---")
        
        response = AIMessage(
            content="I don't have enough relevant information to answer your question based on the available documents. Could you please rephrase your question or ask about something else I can help you with?"
        )
        
        return {"messages": [response]}
    
    def query(self, question: str) -> str:
        """
        Process a query through the RAG system.
        
        Args:
            question: User's question
            
        Returns:
            Generated response
        """
        try:
            # Create initial state
            initial_state = {
                "messages": [HumanMessage(content=question)]
            }
            
            # Run the workflow
            result = self.app.invoke(initial_state)
            
            # Extract the final response
            final_message = result["messages"][-1]
            
            if isinstance(final_message, AIMessage):
                return final_message.content
            else:
                return str(final_message.content)
                
        except Exception as e:
            self.logger.error(f"❌ Error processing query: {e}")
            return f"Sorry, I encountered an error while processing your question: {str(e)}"
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get information about the RAG system."""
        return {
            "vector_store_path": str(self.vector_store_path),
            "total_tools": len(self.tools),
            "tool_names": [tool.name for tool in self.tools],
            "llm_model": "gpt-3.5-turbo",
            "retrieval_method": "Single Vector Store",
            "created_at": datetime.now().isoformat()
        }


def main():
    """Main function to test the GraphBuilder."""
    try:
        # Initialize GraphBuilder
        print("🚀 Initializing GraphBuilder...")
        graph_builder = GraphBuilder()
        
        # Get system information
        system_info = graph_builder.get_system_info()
        
        # Print system summary
        print("\n" + "="*60)
        print("AGENTIC RAG SYSTEM SUMMARY")
        print("="*60)
        print(f"Vector Store Path: {system_info['vector_store_path']}")
        print(f"Tools created: {system_info['total_tools']}")
        print(f"LLM model: {system_info['llm_model']}")
        print(f"Retrieval method: {system_info['retrieval_method']}")
        print()
        
        print("Available tools:")
        for tool_name in system_info['tool_names']:
            print(f"├── {tool_name}")
        
        print("="*60)
        print("✅ GraphBuilder initialized successfully")
        print("📝 Ready to process queries!")
        
  
        
    except Exception as e:
        print(f"❌ Error in main execution: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()