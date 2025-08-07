import streamlit as st
import os
from dotenv import load_dotenv
from typing import List, Dict, Any
import tempfile

# LangChain imports
from langchain.document_loaders import PyPDFLoader, TextLoader, UnstructuredWordDocumentLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
from langchain.vectorstores import Chroma, Pinecone
from langchain.llms import OpenAI, Ollama
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory
from langchain.schema import Document

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🤖 RAG Chatbot")
st.markdown("Ask questions about your uploaded documents!")

class RAGChatbot:
    def __init__(self):
        self.vectorstore = None
        self.qa_chain = None
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    
    def load_documents(self, uploaded_files) -> List[Document]:
        """Load and process uploaded documents."""
        documents = []
        
        for uploaded_file in uploaded_files:
            # Save uploaded file to temporary location
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name
            
            try:
                # Load document based on file type
                if uploaded_file.name.endswith('.pdf'):
                    loader = PyPDFLoader(tmp_file_path)
                elif uploaded_file.name.endswith('.txt'):
                    loader = TextLoader(tmp_file_path)
                elif uploaded_file.name.endswith(('.docx', '.doc')):
                    loader = UnstructuredWordDocumentLoader(tmp_file_path)
                else:
                    st.warning(f"Unsupported file type: {uploaded_file.name}")
                    continue
                
                docs = loader.load()
                
                # Add source metadata
                for doc in docs:
                    doc.metadata['source'] = uploaded_file.name
                
                documents.extend(docs)
                
            except Exception as e:
                st.error(f"Error loading {uploaded_file.name}: {str(e)}")
            
            finally:
                # Clean up temporary file
                os.unlink(tmp_file_path)
        
        return documents
    
    def setup_vectorstore(self, documents: List[Document], embedding_choice: str, vector_db: str):
        """Set up vector store with documents."""
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""]
        )
        texts = text_splitter.split_documents(documents)
        
        # Choose embedding model
        if embedding_choice == "OpenAI":
            embeddings = OpenAIEmbeddings()
        else:  # Local embeddings
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
        
        # Choose vector database
        if vector_db == "Chroma":
            self.vectorstore = Chroma.from_documents(
                texts, 
                embeddings,
                persist_directory="./chroma_db"
            )
        elif vector_db == "Pinecone":
            import pinecone
            pinecone.init(
                api_key=os.getenv("PINECONE_API_KEY"),
                environment=os.getenv("PINECONE_ENVIRONMENT")
            )
            index_name = "rag-chatbot"
            self.vectorstore = Pinecone.from_documents(texts, embeddings, index_name=index_name)
        
        st.success(f"Vector store created with {len(texts)} chunks!")
    
    def setup_qa_chain(self, llm_choice: str, temperature: float):
        """Set up the QA chain."""
        # Choose LLM
        if llm_choice == "OpenAI GPT-4":
            llm = OpenAI(model_name="gpt-4", temperature=temperature)
        elif llm_choice == "OpenAI GPT-3.5":
            llm = OpenAI(model_name="gpt-3.5-turbo", temperature=temperature)
        else:  # Local Ollama
            model_name = llm_choice.replace("Ollama: ", "")
            llm = Ollama(model=model_name, temperature=temperature)
        
        # Create QA chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(search_kwargs={"k": 5}),
            memory=self.memory,
            return_source_documents=True
        )
    
    def query(self, question: str) -> Dict[str, Any]:
        """Query the RAG system."""
        if not self.qa_chain:
            return {"error": "QA chain not initialized"}
        
        try:
            response = self.qa_chain({"query": question})
            return response
        except Exception as e:
            return {"error": str(e)}

# Initialize chatbot
if 'chatbot' not in st.session_state:
    st.session_state.chatbot = RAGChatbot()

if 'messages' not in st.session_state:
    st.session_state.messages = []

# Sidebar configuration
st.sidebar.header("⚙️ Configuration")

# File upload
st.sidebar.subheader("📁 Upload Documents")
uploaded_files = st.sidebar.file_uploader(
    "Choose files",
    type=['pdf', 'txt', 'docx', 'doc'],
    accept_multiple_files=True
)

# Model configuration
st.sidebar.subheader("🤖 Model Settings")

embedding_choice = st.sidebar.selectbox(
    "Embedding Model",
    ["OpenAI", "Local (all-MiniLM-L6-v2)"]
)

vector_db = st.sidebar.selectbox(
    "Vector Database",
    ["Chroma", "Pinecone"]
)

llm_models = ["OpenAI GPT-4", "OpenAI GPT-3.5", "Ollama: llama2", "Ollama: mistral"]
llm_choice = st.sidebar.selectbox("LLM Model", llm_models)

temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.7, 0.1)

# Process documents button
if st.sidebar.button("🔄 Process Documents") and uploaded_files:
    with st.spinner("Processing documents..."):
        # Load documents
        documents = st.session_state.chatbot.load_documents(uploaded_files)
        
        if documents:
            # Setup vector store
            st.session_state.chatbot.setup_vectorstore(documents, embedding_choice, vector_db)
            
            # Setup QA chain
            st.session_state.chatbot.setup_qa_chain(llm_choice, temperature)
            
            st.sidebar.success("✅ Documents processed successfully!")
        else:
            st.sidebar.error("❌ No documents could be processed.")

# Main chat interface
col1, col2 = st.columns([3, 1])

with col1:
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Show sources if available
            if "sources" in message:
                with st.expander("📚 Sources"):
                    for i, source in enumerate(message["sources"]):
                        st.markdown(f"**Source {i+1}:** {source['source']}")
                        st.markdown(f"```\n{source['content'][:200]}...\n```")

    # Chat input
    if prompt := st.chat_input("Ask a question about your documents..."):
        # Check if documents are processed
        if not st.session_state.chatbot.qa_chain:
            st.error("Please upload and process documents first!")
        else:
            # Add user message
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Get response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = st.session_state.chatbot.query(prompt)
                
                if "error" in response:
                    st.error(f"Error: {response['error']}")
                else:
                    answer = response["result"]
                    st.markdown(answer)
                    
                    # Prepare sources for storage
                    sources = []
                    if "source_documents" in response:
                        sources = [
                            {
                                "source": doc.metadata.get("source", "Unknown"),
                                "content": doc.page_content
                            }
                            for doc in response["source_documents"]
                        ]
                        
                        # Show sources
                        with st.expander("📚 Sources"):
                            for i, source in enumerate(sources):
                                st.markdown(f"**Source {i+1}:** {source['source']}")
                                st.markdown(f"```\n{source['content'][:200]}...\n```")
                    
                    # Add assistant message
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": answer,
                        "sources": sources
                    })

with col2:
    st.subheader("📊 Stats")
    
    if st.session_state.chatbot.vectorstore:
        # Vector store info
        st.metric("Documents Processed", len(uploaded_files) if uploaded_files else 0)
        st.metric("Messages", len(st.session_state.messages))
        
        # Clear chat button
        if st.button("🗑️ Clear Chat"):
            st.session_state.messages = []
            st.rerun()
    
    # Help section
    st.subheader("💡 Tips")
    st.markdown("""
    - Upload PDF, TXT, or DOCX files
    - Ask specific questions about the content
    - Use the sources to verify answers
    - Try different temperature settings for varied responses
    """)

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using LangChain and Streamlit")