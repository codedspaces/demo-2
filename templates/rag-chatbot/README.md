# Basic RAG Chatbot

A simple yet powerful Retrieval-Augmented Generation (RAG) chatbot implementation that combines the power of vector search with large language models.

## Features

- 📚 **Document Ingestion**: Support for multiple document formats (PDF, TXT, DOCX, Markdown)
- 🔍 **Vector Search**: Fast similarity search using embeddings
- 🤖 **LLM Integration**: Compatible with OpenAI, Anthropic, and local models
- 💬 **Conversational Interface**: Streamlit-based chat interface
- 🔒 **Privacy-First**: Option to run completely offline with local models

## Architecture

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  Documents  │───▶│   Vector    │───▶│     RAG     │
│  (PDF/TXT)  │    │  Database   │    │   Engine    │
└─────────────┘    └─────────────┘    └─────────────┘
                                             │
                                             ▼
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│    User     │◀───│  Streamlit  │◀───│     LLM     │
│   Query     │    │    UI       │    │  Response   │
└─────────────┘    └─────────────┘    └─────────────┘
```

## Quick Start

### Prerequisites

```bash
pip install -r requirements.txt
```

### Environment Setup

Create a `.env` file:

```bash
# OpenAI API (optional - for cloud LLM)
OPENAI_API_KEY=your_openai_api_key

# Anthropic API (optional - alternative cloud LLM)
ANTHROPIC_API_KEY=your_anthropic_api_key

# Vector Database (choose one)
PINECONE_API_KEY=your_pinecone_key
PINECONE_ENVIRONMENT=your_pinecone_env

# Or use local Chroma (no setup needed)
VECTOR_DB=chroma
```

### Usage

1. **Start the application:**
   ```bash
   streamlit run app.py
   ```

2. **Upload documents** through the sidebar

3. **Ask questions** about your documents

## Configuration

### Vector Database Options

**Chroma (Local - Default)**
```python
from langchain.vectorstores import Chroma

vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings
)
```

**Pinecone (Cloud)**
```python
import pinecone
from langchain.vectorstores import Pinecone

pinecone.init(api_key="your-key", environment="your-env")
vectorstore = Pinecone.from_documents(docs, embeddings, index_name="your-index")
```

### LLM Options

**OpenAI GPT-4**
```python
from langchain.llms import OpenAI

llm = OpenAI(
    model_name="gpt-4",
    temperature=0.7,
    max_tokens=500
)
```

**Local Ollama**
```python
from langchain.llms import Ollama

llm = Ollama(
    model="llama2",
    temperature=0.7
)
```

## File Structure

```
rag-chatbot/
├── app.py                 # Main Streamlit application
├── src/
│   ├── document_loader.py # Document processing utilities
│   ├── embeddings.py      # Embedding model configurations
│   ├── vector_store.py    # Vector database setup
│   └── llm_config.py      # LLM configurations
├── requirements.txt       # Python dependencies
├── .env.example          # Environment variables template
├── config.yaml           # Application configuration
└── README.md             # This file
```

## Advanced Features

### Custom Document Processors

```python
from src.document_loader import CustomDocumentLoader

loader = CustomDocumentLoader()
loader.add_processor(".pdf", PDFProcessor())
loader.add_processor(".docx", DocxProcessor())
loader.add_processor(".md", MarkdownProcessor())
```

### Chunking Strategies

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", " ", ""]
)
```

### Retrieval Enhancement

```python
# Similarity search with metadata filtering
docs = vectorstore.similarity_search(
    query,
    k=5,
    filter={"source": "specific_document.pdf"}
)

# Maximum marginal relevance search
docs = vectorstore.max_marginal_relevance_search(
    query,
    k=5,
    fetch_k=20
)
```

## Deployment

### Docker

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### Docker Compose

```yaml
version: '3.8'
services:
  rag-chatbot:
    build: .
    ports:
      - "8501:8501"
    volumes:
      - ./data:/app/data
      - ./chroma_db:/app/chroma_db
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
```

## Performance Tips

1. **Optimize Chunk Size**: Experiment with chunk sizes (500-2000 tokens)
2. **Use Appropriate Embeddings**: 
   - `text-embedding-ada-002` for general use
   - `all-MiniLM-L6-v2` for local deployment
3. **Vector Database Tuning**:
   - Increase `k` for better recall
   - Use metadata filtering for large document sets
4. **Caching**: Implement response caching for repeated queries

## Troubleshooting

### Common Issues

**Slow Vector Search**
```python
# Solution: Reduce embedding dimensions or use approximate search
vectorstore = Chroma(
    embedding_function=embeddings,
    search_kwargs={"nprobe": 10}  # For approximate search
)
```

**Out of Memory**
```python
# Solution: Process documents in batches
def process_documents_in_batches(docs, batch_size=10):
    for i in range(0, len(docs), batch_size):
        batch = docs[i:i + batch_size]
        vectorstore.add_documents(batch)
```

## Next Steps

- Try the [Multi-Modal RAG](../multimodal-rag/) template for images and audio
- Explore [Enterprise RAG](../enterprise-rag/) for production deployment
- Check out [Memory-Enabled Chatbot](../memory-chatbot/) for conversation history

## License

MIT License - see [LICENSE](../../LICENSE) for details.