# Web Scraper for RAG Systems

A comprehensive web scraping pipeline that collects, processes, and indexes web content for Retrieval-Augmented Generation (RAG) systems.

## Features

- 🕷️ **Multi-Tool Scraping**: Supports Firecrawl, Scrapy, Playwright, and BeautifulSoup
- 🧹 **Content Processing**: Clean and structure scraped content for LLM consumption
- 📚 **RAG Integration**: Direct indexing into vector databases (Chroma, Qdrant, Pinecone)
- 🔄 **Incremental Updates**: Track and update changed content
- 📊 **Monitoring**: Built-in analytics and error tracking
- ⚡ **Async Processing**: High-performance concurrent scraping

## Architecture

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Target    │───▶│   Scraper   │───▶│  Content    │
│  Websites   │    │  Engines    │    │ Processor   │
└─────────────┘    └─────────────┘    └─────────────┘
                                             │
                                             ▼
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│    RAG      │◀───│   Vector    │◀───│  Document   │
│   System    │    │  Database   │    │  Chunker    │
└─────────────┘    └─────────────┘    └─────────────┘
```

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Configuration

Create a `config.yaml` file:

```yaml
scrapers:
  firecrawl:
    api_key: "your_firecrawl_api_key"
    enabled: true
  
  playwright:
    headless: true
    timeout: 30000
    enabled: true
  
  scrapy:
    concurrent_requests: 16
    download_delay: 1
    enabled: true

sources:
  - url: "https://example.com/docs"
    scraper: "firecrawl"
    depth: 2
    selectors:
      content: "article, .content, main"
      title: "h1, .title"
    
  - url: "https://blog.example.com"
    scraper: "playwright"
    pagination: true
    frequency: "daily"

processing:
  chunk_size: 1000
  chunk_overlap: 200
  min_chunk_size: 100
  
vector_store:
  provider: "chroma"
  collection: "scraped_content"
  embedding_model: "sentence-transformers/all-MiniLM-L6-v2"
```

### Usage

```bash
# Run complete scraping pipeline
python scrape_rag.py --config config.yaml

# Scrape specific source
python scrape_rag.py --source "https://example.com" --scraper firecrawl

# Update existing content
python scrape_rag.py --update-only

# Monitor scraping status
python scrape_rag.py --status
```

## Scraper Implementations

### 1. Firecrawl (AI-Powered)

```python
from firecrawl import FirecrawlApp

class FirecrawlScraper:
    def __init__(self, api_key):
        self.app = FirecrawlApp(api_key=api_key)
    
    def scrape_url(self, url, options=None):
        """Scrape single URL with AI extraction"""
        params = {
            'crawlerOptions': {
                'includes': [],
                'excludes': [],
                'generateImgAltText': True,
                'returnOnlyUrls': False
            },
            'pageOptions': {
                'onlyMainContent': True,
                'includeHtml': False,
                'screenshot': False
            }
        }
        
        if options:
            params.update(options)
        
        return self.app.scrape_url(url, params)
    
    def crawl_site(self, url, max_pages=100):
        """Crawl entire site with smart discovery"""
        job = self.app.crawl_url(url, {
            'crawlerOptions': {
                'maxDepth': 3,
                'limit': max_pages
            },
            'pageOptions': {
                'onlyMainContent': True
            }
        })
        
        return self.monitor_crawl_job(job['jobId'])
```

### 2. Playwright (Dynamic Content)

```python
from playwright.async_api import async_playwright
import asyncio

class PlaywrightScraper:
    def __init__(self, headless=True, timeout=30000):
        self.headless = headless
        self.timeout = timeout
    
    async def scrape_spa(self, url, selectors):
        """Scrape Single Page Applications"""
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=self.headless)
            page = await browser.new_page()
            
            try:
                await page.goto(url, timeout=self.timeout)
                await page.wait_for_load_state('networkidle')
                
                content = {}
                for key, selector in selectors.items():
                    elements = await page.query_selector_all(selector)
                    content[key] = [await el.inner_text() for el in elements]
                
                return content
                
            finally:
                await browser.close()
    
    async def scrape_with_pagination(self, base_url, next_selector):
        """Handle paginated content"""
        all_content = []
        current_url = base_url
        
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=self.headless)
            page = await browser.new_page()
            
            try:
                while current_url:
                    await page.goto(current_url)
                    content = await self.extract_content(page)
                    all_content.extend(content)
                    
                    # Find next page
                    next_link = await page.query_selector(next_selector)
                    current_url = await next_link.get_attribute('href') if next_link else None
                    
            finally:
                await browser.close()
        
        return all_content
```

### 3. Scrapy (Large Scale)

```python
import scrapy
from scrapy.crawler import CrawlerProcess

class DocumentSpider(scrapy.Spider):
    name = 'documents'
    
    def __init__(self, start_urls, selectors, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.start_urls = start_urls
        self.selectors = selectors
    
    def parse(self, response):
        # Extract content using selectors
        title = response.css(self.selectors.get('title', 'h1::text')).get()
        content = response.css(self.selectors.get('content', 'p::text')).getall()
        
        yield {
            'url': response.url,
            'title': title,
            'content': ' '.join(content),
            'timestamp': response.headers.get('Date'),
            'word_count': len(' '.join(content).split())
        }
        
        # Follow links
        for link in response.css('a[href]'):
            next_url = response.urljoin(link.attrib['href'])
            if self.should_follow(next_url):
                yield response.follow(next_url, self.parse)
    
    def should_follow(self, url):
        """Implement link filtering logic"""
        excluded_extensions = ['.pdf', '.jpg', '.png', '.zip']
        return not any(url.endswith(ext) for ext in excluded_extensions)

class ScrapyRunner:
    def __init__(self, settings=None):
        self.settings = settings or {
            'ROBOTSTXT_OBEY': True,
            'CONCURRENT_REQUESTS': 16,
            'DOWNLOAD_DELAY': 1,
            'AUTOTHROTTLE_ENABLED': True,
        }
    
    def run_spider(self, start_urls, selectors):
        process = CrawlerProcess(self.settings)
        process.crawl(DocumentSpider, start_urls=start_urls, selectors=selectors)
        process.start()
```

## Content Processing Pipeline

### Text Cleaning & Normalization

```python
import re
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter

class ContentProcessor:
    def __init__(self, chunk_size=1000, chunk_overlap=200):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )
    
    def clean_html(self, html_content):
        """Remove HTML tags and clean text"""
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style", "nav", "footer"]):
            script.decompose()
        
        # Get text and clean whitespace
        text = soup.get_text()
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def extract_metadata(self, content, url):
        """Extract useful metadata"""
        return {
            'url': url,
            'word_count': len(content.split()),
            'char_count': len(content),
            'estimated_read_time': len(content.split()) // 200,  # ~200 wpm
            'language': self.detect_language(content),
            'content_type': self.classify_content(content)
        }
    
    def chunk_content(self, content, metadata):
        """Split content into chunks with metadata"""
        chunks = self.text_splitter.split_text(content)
        
        documents = []
        for i, chunk in enumerate(chunks):
            doc_metadata = metadata.copy()
            doc_metadata.update({
                'chunk_index': i,
                'total_chunks': len(chunks),
                'chunk_id': f"{metadata['url']}#{i}"
            })
            
            documents.append({
                'content': chunk,
                'metadata': doc_metadata
            })
        
        return documents
```

### Vector Database Integration

```python
from langchain.vectorstores import Chroma, Qdrant
from langchain.embeddings import HuggingFaceEmbeddings
import chromadb

class VectorStoreManager:
    def __init__(self, provider='chroma', collection_name='scraped_content'):
        self.provider = provider
        self.collection_name = collection_name
        self.embeddings = HuggingFaceEmbeddings(
            model_name='sentence-transformers/all-MiniLM-L6-v2'
        )
        self.setup_vector_store()
    
    def setup_vector_store(self):
        """Initialize vector store"""
        if self.provider == 'chroma':
            self.client = chromadb.PersistentClient(path="./chroma_db")
            self.vectorstore = Chroma(
                client=self.client,
                collection_name=self.collection_name,
                embedding_function=self.embeddings
            )
        elif self.provider == 'qdrant':
            from qdrant_client import QdrantClient
            client = QdrantClient(host="localhost", port=6333)
            self.vectorstore = Qdrant(
                client=client,
                collection_name=self.collection_name,
                embeddings=self.embeddings
            )
    
    def add_documents(self, documents):
        """Add processed documents to vector store"""
        texts = [doc['content'] for doc in documents]
        metadatas = [doc['metadata'] for doc in documents]
        
        # Check for existing documents to avoid duplicates
        existing_urls = self.get_existing_urls()
        new_documents = [
            (text, meta) for text, meta in zip(texts, metadatas)
            if meta['url'] not in existing_urls
        ]
        
        if new_documents:
            new_texts, new_metadatas = zip(*new_documents)
            self.vectorstore.add_texts(
                texts=list(new_texts),
                metadatas=list(new_metadatas)
            )
            
        return len(new_documents)
    
    def search_similar(self, query, k=5):
        """Search for similar content"""
        return self.vectorstore.similarity_search(query, k=k)
    
    def get_existing_urls(self):
        """Get list of already indexed URLs"""
        # Implementation depends on vector store
        if self.provider == 'chroma':
            collection = self.client.get_collection(self.collection_name)
            metadata = collection.get()['metadatas']
            return set(meta.get('url') for meta in metadata if meta.get('url'))
        return set()
```

## Monitoring & Analytics

```python
import sqlite3
from datetime import datetime
import logging

class ScrapingMonitor:
    def __init__(self, db_path='scraping.db'):
        self.db_path = db_path
        self.init_database()
        self.setup_logging()
    
    def init_database(self):
        """Initialize monitoring database"""
        conn = sqlite3.connect(self.db_path)
        conn.execute('''
            CREATE TABLE IF NOT EXISTS scraping_jobs (
                id INTEGER PRIMARY KEY,
                url TEXT,
                scraper TEXT,
                status TEXT,
                pages_scraped INTEGER,
                documents_added INTEGER,
                started_at TIMESTAMP,
                completed_at TIMESTAMP,
                error_message TEXT
            )
        ''')
        conn.commit()
        conn.close()
    
    def log_job_start(self, url, scraper):
        """Log start of scraping job"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO scraping_jobs (url, scraper, status, started_at)
            VALUES (?, ?, 'running', ?)
        ''', (url, scraper, datetime.now()))
        job_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return job_id
    
    def log_job_completion(self, job_id, pages_scraped, documents_added, error=None):
        """Log completion of scraping job"""
        conn = sqlite3.connect(self.db_path)
        status = 'completed' if not error else 'failed'
        conn.execute('''
            UPDATE scraping_jobs 
            SET status=?, pages_scraped=?, documents_added=?, 
                completed_at=?, error_message=?
            WHERE id=?
        ''', (status, pages_scraped, documents_added, datetime.now(), error, job_id))
        conn.commit()
        conn.close()
    
    def get_statistics(self):
        """Get scraping statistics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        stats = {}
        
        # Total jobs
        cursor.execute('SELECT COUNT(*) FROM scraping_jobs')
        stats['total_jobs'] = cursor.fetchone()[0]
        
        # Success rate
        cursor.execute('SELECT COUNT(*) FROM scraping_jobs WHERE status="completed"')
        stats['successful_jobs'] = cursor.fetchone()[0]
        stats['success_rate'] = (stats['successful_jobs'] / stats['total_jobs']) * 100 if stats['total_jobs'] > 0 else 0
        
        # Total documents
        cursor.execute('SELECT SUM(documents_added) FROM scraping_jobs WHERE status="completed"')
        result = cursor.fetchone()[0]
        stats['total_documents'] = result if result else 0
        
        conn.close()
        return stats
```

## Deployment & Scheduling

### Docker Setup

```dockerfile
FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    wget \
    gnupg \
    && rm -rf /var/lib/apt/lists/*

# Install Playwright browsers
RUN pip install playwright
RUN playwright install chromium

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD ["python", "scrape_rag.py", "--config", "config.yaml"]
```

### Cron Schedule

```bash
# Daily scraping at 2 AM
0 2 * * * cd /path/to/scraper && python scrape_rag.py --update-only

# Weekly full crawl on Sundays at 1 AM  
0 1 * * 0 cd /path/to/scraper && python scrape_rag.py --full-crawl
```

## Performance Optimization

### Async Processing

```python
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor

class AsyncScrapeManager:
    def __init__(self, max_concurrent=10):
        self.max_concurrent = max_concurrent
        self.semaphore = asyncio.Semaphore(max_concurrent)
    
    async def scrape_urls_concurrent(self, urls, scraper_func):
        """Scrape multiple URLs concurrently"""
        tasks = [self.scrape_with_semaphore(url, scraper_func) for url in urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return results
    
    async def scrape_with_semaphore(self, url, scraper_func):
        """Scrape single URL with concurrency control"""
        async with self.semaphore:
            try:
                return await scraper_func(url)
            except Exception as e:
                logging.error(f"Error scraping {url}: {e}")
                return None
```

### Caching Strategy

```python
import hashlib
import pickle
from functools import wraps

def cache_content(cache_dir='./cache', ttl_hours=24):
    """Cache scraped content to avoid re-scraping"""
    def decorator(func):
        @wraps(func)
        def wrapper(url, *args, **kwargs):
            # Generate cache key
            cache_key = hashlib.md5(f"{url}_{str(args)}_{str(kwargs)}".encode()).hexdigest()
            cache_file = f"{cache_dir}/{cache_key}.pkl"
            
            # Check if cached version exists and is fresh
            if os.path.exists(cache_file):
                mtime = os.path.getmtime(cache_file)
                if (time.time() - mtime) < (ttl_hours * 3600):
                    with open(cache_file, 'rb') as f:
                        return pickle.load(f)
            
            # Scrape and cache
            result = func(url, *args, **kwargs)
            os.makedirs(cache_dir, exist_ok=True)
            with open(cache_file, 'wb') as f:
                pickle.dump(result, f)
            
            return result
        return wrapper
    return decorator
```

## Usage Examples

### News Article Scraping

```python
# Scrape news articles for current events RAG
config = {
    'sources': [
        {
            'url': 'https://techcrunch.com',
            'scraper': 'playwright',
            'selectors': {
                'title': 'h1.article__title',
                'content': '.article-content',
                'date': '.article__meta time'
            }
        }
    ]
}

scraper = WebScraperRAG(config)
results = scraper.scrape_all()
```

### Documentation Scraping

```python
# Scrape technical documentation
config = {
    'sources': [
        {
            'url': 'https://docs.python.org',
            'scraper': 'firecrawl',
            'depth': 3,
            'filters': {
                'include_patterns': ['/library/', '/tutorial/'],
                'exclude_patterns': ['/downloads/', '/bugs/']
            }
        }
    ]
}
```

## Best Practices

1. **Respect robots.txt**: Always check and follow robots.txt rules
2. **Rate Limiting**: Implement delays between requests
3. **Error Handling**: Robust error handling and retry logic
4. **Content Validation**: Verify scraped content quality
5. **Incremental Updates**: Only scrape changed content
6. **Monitoring**: Track scraping performance and success rates

## Troubleshooting

### Common Issues

**JavaScript-heavy sites**: Use Playwright instead of simple HTTP requests
**Rate limiting**: Implement exponential backoff and respect rate limits
**Content changes**: Set up monitoring for content structure changes
**Memory issues**: Process content in batches for large sites

## Next Steps

- Explore [Enterprise RAG](../enterprise-rag/) for production deployment
- Check out [Evaluation Tools](../../README.md#evaluation--testing) for content quality assessment
- Review [Infrastructure](../../infrastructure/) for scaling scraping operations

## License

MIT License - see [LICENSE](../../LICENSE) for details.