# 🐳 Docker Infrastructure Templates

Production-ready Docker configurations for LLM applications, from simple single-service containers to complex multi-service stacks.

## 📁 Available Templates

### 1. LLM Stack (`llm-stack/`)
Complete production environment with:
- **Local LLM serving** (Ollama)
- **Vector databases** (Chroma, Qdrant, Milvus)
- **Application backend** (FastAPI)
- **Frontend interface** (Streamlit)
- **Caching layer** (Redis)
- **Database** (PostgreSQL)
- **Monitoring** (Prometheus + Grafana)
- **Reverse proxy** (Nginx)

### 2. Simple RAG App (`simple-rag/`)
Minimal RAG application with:
- Streamlit frontend
- Chroma vector database
- OpenAI integration

### 3. Local LLM Only (`local-llm/`)
Just local LLM serving:
- Ollama container
- GPU support
- Model management

### 4. Vector DB Only (`vector-db/`)
Standalone vector databases:
- Chroma
- Qdrant
- Milvus
- Pinecone (configuration)

## 🚀 Quick Start

### Prerequisites

```bash
# Install Docker and Docker Compose
curl -fsSL https://get.docker.com -o get-docker.sh
sh get-docker.sh

# For GPU support (NVIDIA)
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

### Deploy Full LLM Stack

```bash
cd llm-stack/
cp .env.example .env
# Edit .env with your API keys
docker-compose up -d
```

### Deploy Simple RAG App

```bash
cd simple-rag/
docker-compose up -d
```

## 🔧 Configuration

### Environment Variables

Create a `.env` file in each template directory:

```bash
# API Keys
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key

# Database
POSTGRES_PASSWORD=secure_password
REDIS_PASSWORD=redis_password

# Monitoring
GRAFANA_ADMIN_PASSWORD=admin_password

# Application
DEBUG=false
LOG_LEVEL=info
```

### GPU Configuration

For GPU-enabled containers, ensure:

1. **NVIDIA Docker runtime** installed
2. **GPU resources** allocated in docker-compose.yml:
   ```yaml
   deploy:
     resources:
       reservations:
         devices:
           - driver: nvidia
             count: 1
             capabilities: [gpu]
   ```

### Resource Limits

Adjust memory and CPU limits:

```yaml
services:
  ollama:
    deploy:
      resources:
        limits:
          memory: 8G
          cpus: '4'
        reservations:
          memory: 4G
          cpus: '2'
```

## 📊 Monitoring & Observability

### Included Dashboards

The full stack includes pre-configured dashboards:

- **LLM Metrics**: Token usage, latency, error rates
- **Vector DB Performance**: Query times, index size
- **Application Health**: API response times, error rates
- **System Resources**: CPU, memory, GPU utilization

### Access URLs

After deployment:
- **Application**: http://localhost:8501
- **API Docs**: http://localhost:8080/docs
- **Grafana**: http://localhost:3000 (admin/admin)
- **Prometheus**: http://localhost:9090
- **Chroma**: http://localhost:8000
- **Qdrant**: http://localhost:6333/dashboard

## 🔒 Security

### Production Hardening

1. **Change default passwords**:
   ```bash
   # Generate secure passwords
   openssl rand -base64 32
   ```

2. **Use SSL certificates**:
   ```bash
   # Place certificates in nginx/ssl/
   nginx/ssl/
   ├── cert.pem
   └── key.pem
   ```

3. **Network security**:
   ```yaml
   # Restrict external access
   ports:
     - "127.0.0.1:5432:5432"  # PostgreSQL internal only
   ```

4. **Environment isolation**:
   ```yaml
   # Use Docker secrets for sensitive data
   secrets:
     openai_key:
       file: ./secrets/openai_key.txt
   ```

### API Security

```yaml
# API rate limiting in nginx
location /api/ {
    limit_req zone=api burst=10 nodelay;
    proxy_pass http://llm-api:8080;
}
```

## 📈 Scaling

### Horizontal Scaling

Scale individual services:

```bash
# Scale API servers
docker-compose up -d --scale llm-api=3

# Scale with load balancing
# (requires nginx configuration update)
```

### Vertical Scaling

Adjust resource allocation:

```yaml
services:
  llm-api:
    deploy:
      replicas: 3
      resources:
        limits:
          memory: 2G
          cpus: '1'
```

### Database Scaling

For high-traffic applications:

```yaml
# PostgreSQL with read replicas
postgres-primary:
  image: postgres:15-alpine
  environment:
    - POSTGRES_REPLICATION_MODE=master

postgres-replica:
  image: postgres:15-alpine
  environment:
    - POSTGRES_REPLICATION_MODE=slave
    - POSTGRES_MASTER_HOST=postgres-primary
```

## 🔧 Customization

### Adding New Services

1. **Define service** in docker-compose.yml:
   ```yaml
   my-service:
     build: ./my-service
     ports:
       - "8082:8082"
     depends_on:
       - postgres
   ```

2. **Create Dockerfile**:
   ```dockerfile
   FROM python:3.9-slim
   WORKDIR /app
   COPY requirements.txt .
   RUN pip install -r requirements.txt
   COPY . .
   CMD ["python", "app.py"]
   ```

3. **Update nginx configuration** if needed

### Custom Vector Databases

Add alternative vector databases:

```yaml
# Weaviate
weaviate:
  image: semitechnologies/weaviate:latest
  ports:
    - "8080:8080"
  environment:
    - AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED=true
    - PERSISTENCE_DATA_PATH=/var/lib/weaviate
```

## 🚨 Troubleshooting

### Common Issues

1. **GPU not detected**:
   ```bash
   # Check NVIDIA runtime
   docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi
   ```

2. **Port conflicts**:
   ```bash
   # Check port usage
   netstat -tulpn | grep :8501
   ```

3. **Memory issues**:
   ```bash
   # Monitor container resources
   docker stats
   ```

4. **Network connectivity**:
   ```bash
   # Test service connectivity
   docker exec llm-api curl http://chroma:8000/api/v1/heartbeat
   ```

### Logs & Debugging

```bash
# View service logs
docker-compose logs -f llm-api

# Debug container
docker exec -it llm-api bash

# Check docker-compose configuration
docker-compose config
```

### Performance Optimization

1. **Use multi-stage builds**:
   ```dockerfile
   FROM python:3.9-slim as builder
   # Build dependencies
   
   FROM python:3.9-slim as runtime
   # Copy only runtime files
   ```

2. **Optimize layer caching**:
   ```dockerfile
   # Copy requirements first for better caching
   COPY requirements.txt .
   RUN pip install -r requirements.txt
   COPY . .
   ```

3. **Use .dockerignore**:
   ```
   .git
   __pycache__
   *.pyc
   .env
   ```

## 📚 Next Steps

- Explore [Kubernetes deployments](../k8s/) for orchestration
- Check [Terraform modules](../terraform/) for cloud infrastructure
- Review [monitoring setup](../../tutorials/08-optimization/) for production
- Learn about [security best practices](../../tutorials/07-production/)

## 🤝 Contributing

Have improvements for these Docker templates?

1. Test your changes locally
2. Update documentation
3. Submit a pull request

We welcome contributions for:
- New service integrations
- Performance optimizations
- Security improvements
- Additional monitoring

---

*These templates provide a solid foundation for containerized LLM applications. Customize them based on your specific needs and scale requirements.*