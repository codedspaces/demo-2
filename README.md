# 🧰 AI Engineering Toolkit

**Build better LLM apps — faster, smarter, production-ready.**

A curated, practical resource for AI engineers building with Large Language Models. This toolkit includes battle-tested tools, frameworks, templates, and reference implementations for developing, deploying, and optimizing LLM-powered systems.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](http://makeapullrequest.com)
[![GitHub stars](https://img.shields.io/github/stars/yourusername/ai-engineering-toolkit.svg?style=social)](https://github.com/yourusername/ai-engineering-toolkit/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/yourusername/ai-engineering-toolkit.svg?style=social)](https://github.com/yourusername/ai-engineering-toolkit/network/members)
[![Twitter Follow](https://img.shields.io/twitter/follow/yourusername?style=social)](https://twitter.com/yourusername)

## 📋 Table of Contents

- [🛠️ Tooling for AI Engineers](#%EF%B8%8F-tooling-for-ai-engineers)
  - [Vector Databases](#vector-databases)
  - [Orchestration & Workflows](#orchestration--workflows)
  - [RAG (Retrieval-Augmented Generation)](#rag-retrieval-augmented-generation)
  - [Evaluation & Testing](#evaluation--testing)
  - [Model Management](#model-management)
  - [Data Collection & Web Scraping](#data-collection--web-scraping)
- [🤖 Agent Frameworks](#-agent-frameworks)
- [📦 LLM App Templates](#-llm-app-templates)
- [🚀 Infrastructure & Deployment](#-infrastructure--deployment)
- [📚 Tutorials & Notebooks](#-tutorials--notebooks)
- [📰 Newsletter Companion Demos](#-newsletter-companion-demos)
- [🤝 Contributing](#-contributing)
- [📄 License](#-license)

## 🛠️ Tooling for AI Engineers

### Vector Databases

| Tool | Description | Language | License | Stars |
|------|-------------|----------|---------|-------|
| [Pinecone](https://www.pinecone.io/?utm_source=ai-engineering-toolkit&utm_medium=github&utm_campaign=vector-db) | Managed vector database for production AI applications | API/SDK | Commercial | - |
| [Weaviate](https://github.com/weaviate/weaviate?utm_source=ai-engineering-toolkit&utm_medium=github&utm_campaign=vector-db) | Open-source vector database with GraphQL API | Go | BSD-3 | [![GitHub stars](https://img.shields.io/github/stars/weaviate/weaviate.svg)](https://github.com/weaviate/weaviate/stargazers) |
| [Qdrant](https://github.com/qdrant/qdrant?utm_source=ai-engineering-toolkit&utm_medium=github&utm_campaign=vector-db) | Vector similarity search engine with extended filtering | Rust | Apache-2.0 | [![GitHub stars](https://img.shields.io/github/stars/qdrant/qdrant.svg)](https://github.com/qdrant/qdrant/stargazers) |
| [Chroma](https://github.com/chroma-core/chroma?utm_source=ai-engineering-toolkit&utm_medium=github&utm_campaign=vector-db) | Open-source embedding database for LLM apps | Python | Apache-2.0 | [![GitHub stars](https://img.shields.io/github/stars/chroma-core/chroma.svg)](https://github.com/chroma-core/chroma/stargazers) |
| [Milvus](https://github.com/milvus-io/milvus?utm_source=ai-engineering-toolkit&utm_medium=github&utm_campaign=vector-db) | Cloud-native vector database for scalable similarity search | Go/C++ | Apache-2.0 | [![GitHub stars](https://img.shields.io/github/stars/milvus-io/milvus.svg)](https://github.com/milvus-io/milvus/stargazers) |
| [FAISS](https://github.com/facebookresearch/faiss?utm_source=ai-engineering-toolkit&utm_medium=github&utm_campaign=vector-db) | Library for efficient similarity search and clustering | C++/Python | MIT | [![GitHub stars](https://img.shields.io/github/stars/facebookresearch/faiss.svg)](https://github.com/facebookresearch/faiss/stargazers) |

### Orchestration & Workflows

| Tool | Description | Language | License | Stars |
|------|-------------|----------|---------|-------|
| [LangChain](https://github.com/langchain-ai/langchain?utm_source=ai-engineering-toolkit&utm_medium=github&utm_campaign=orchestration) | Framework for developing LLM applications | Python/JS | MIT | [![GitHub stars](https://img.shields.io/github/stars/langchain-ai/langchain.svg)](https://github.com/langchain-ai/langchain/stargazers) |
| [LlamaIndex](https://github.com/run-llama/llama_index?utm_source=ai-engineering-toolkit&utm_medium=github&utm_campaign=orchestration) | Data framework for LLM applications | Python | MIT | [![GitHub stars](https://img.shields.io/github/stars/run-llama/llama_index.svg)](https://github.com/run-llama/llama_index/stargazers) |
| [Haystack](https://github.com/deepset-ai/haystack?utm_source=ai-engineering-toolkit&utm_medium=github&utm_campaign=orchestration) | End-to-end NLP framework for production | Python | Apache-2.0 | [![GitHub stars](https://img.shields.io/github/stars/deepset-ai/haystack.svg)](https://github.com/deepset-ai/haystack/stargazers) |
| [DSPy](https://github.com/stanfordnlp/dspy?utm_source=ai-engineering-toolkit&utm_medium=github&utm_campaign=orchestration) | Framework for algorithmically optimizing LM prompts | Python | MIT | [![GitHub stars](https://img.shields.io/github/stars/stanfordnlp/dspy.svg)](https://github.com/stanfordnlp/dspy/stargazers) |
| [Semantic Kernel](https://github.com/microsoft/semantic-kernel?utm_source=ai-engineering-toolkit&utm_medium=github&utm_campaign=orchestration) | SDK for integrating AI into conventional programming languages | C#/Python/Java | MIT | [![GitHub stars](https://img.shields.io/github/stars/microsoft/semantic-kernel.svg)](https://github.com/microsoft/semantic-kernel/stargazers) |

### PDF Extraction Tools

| Tool | Description | Language | License | Stars |
|------|-------------|----------|---------|-------|
| [Docling](https://github.com/docling-project/docling) | AI-powered toolkit converting PDF, DOCX, PPTX, HTML, images into structured JSON/Markdown with layout, OCR, table, and code recognition | Python | MIT | [![GitHub stars](https://img.shields.io/github/stars/docling-project/docling.svg)](https://github.com/docling-project/docling/stargazers) |
| [pdfplumber](https://github.com/jsvine/pdfplumber) | Drill through PDFs at a character level, extract text & tables, and visually debug extraction | Python | MIT | [![GitHub stars](https://img.shields.io/github/stars/jsvine/pdfplumber.svg)](https://github.com/jsvine/pdfplumber/stargazers) |
| [PyMuPDF (fitz)](https://github.com/pymupdf/PyMuPDF) | Lightweight, high-performance PDF parser for text/image extraction and manipulation | Python / C | AGPL-3.0 | [![GitHub stars](https://img.shields.io/github/stars/pymupdf/PyMuPDF.svg)](https://github.com/pymupdf/PyMuPDf/stargazers) |
| [PDF.js](https://github.com/mozilla/pdf.js) | Browser-based PDF renderer with text extraction capabilities | JavaScript | Apache-2.0 | [![GitHub stars](https://img.shields.io/github/stars/mozilla/pdf.js.svg)](https://github.com/mozilla/pdf.js/stargazers) |
| [Camelot](https://github.com/camelot-dev/camelot) | Extracts structured tabular data from PDFs into DataFrames and CSVs | Python | MIT | [![GitHub stars](https://img.shields.io/github/stars/camelot-dev/camelot.svg)](https://github.com/camelot-dev/camelot/stargazers) |

### RAG (Retrieval-Augmented Generation)

| Tool | Description | Language | License | Stars |
|------|-------------|----------|---------|-------|
| [RAGFlow](https://github.com/infiniflow/ragflow) | Open-source RAG engine based on deep document understanding | Python | Apache-2.0 | ![GitHub stars](https://img.shields.io/github/stars/infiniflow/ragflow.svg) |
| [Verba](https://github.com/weaviate/Verba) | Retrieval Augmented Generation (RAG) chatbot | Python | BSD-3 | ![GitHub stars](https://img.shields.io/github/stars/weaviate/Verba.svg) |
| [PrivateGPT](https://github.com/imartinez/privateGPT) | Interact with documents using local LLMs | Python | Apache-2.0 | ![GitHub stars](https://img.shields.io/github/stars/imartinez/privateGPT.svg) |
| [AnythingLLM](https://github.com/Mintplex-Labs/anything-llm) | All-in-one AI application for any LLM | JavaScript | MIT | ![GitHub stars](https://img.shields.io/github/stars/Mintplex-Labs/anything-llm.svg) |
| [Quivr](https://github.com/QuivrHQ/quivr) | Your GenAI second brain | Python/TypeScript | Apache-2.0 | ![GitHub stars](https://img.shields.io/github/stars/QuivrHQ/quivr.svg) |

### Evaluation & Testing

| Tool | Description | Language | License | Stars |
|------|-------------|----------|---------|-------|
| [Ragas](https://github.com/explodinggradients/ragas) | Evaluation framework for RAG pipelines | Python | Apache-2.0 | ![GitHub stars](https://img.shields.io/github/stars/explodinggradients/ragas.svg) |
| [LangSmith](https://smith.langchain.com/) | Platform for debugging, testing, and monitoring LLM applications | API/SDK | Commercial | - |
| [Phoenix](https://github.com/Arize-ai/phoenix) | ML observability for LLM, vision, language, and tabular models | Python | Apache-2.0 | ![GitHub stars](https://img.shields.io/github/stars/Arize-ai/phoenix.svg) |
| [DeepEval](https://github.com/confident-ai/deepeval) | LLM evaluation framework for unit testing LLM outputs | Python | Apache-2.0 | ![GitHub stars](https://img.shields.io/github/stars/confident-ai/deepeval.svg) |
| [TruLens](https://github.com/truera/trulens) | Evaluation and tracking for LLM experiments | Python | MIT | ![GitHub stars](https://img.shields.io/github/stars/truera/trulens.svg) |

### Model Management

| Tool | Description | Language | License | Stars |
|------|-------------|----------|---------|-------|
| [Hugging Face Hub](https://github.com/huggingface/huggingface_hub?utm_source=ai-engineering-toolkit&utm_medium=github&utm_campaign=model-management) | Client library for Hugging Face Hub | Python | Apache-2.0 | [![GitHub stars](https://img.shields.io/github/stars/huggingface/huggingface_hub.svg)](https://github.com/huggingface/huggingface_hub/stargazers) |
| [MLflow](https://github.com/mlflow/mlflow?utm_source=ai-engineering-toolkit&utm_medium=github&utm_campaign=model-management) | Platform for ML lifecycle management | Python | Apache-2.0 | [![GitHub stars](https://img.shields.io/github/stars/mlflow/mlflow.svg)](https://github.com/mlflow/mlflow/stargazers) |
| [Weights & Biases](https://github.com/wandb/wandb?utm_source=ai-engineering-toolkit&utm_medium=github&utm_campaign=model-management) | Developer tools for ML | Python | MIT | [![GitHub stars](https://img.shields.io/github/stars/wandb/wandb.svg)](https://github.com/wandb/wandb/stargazers) |
| [DVC](https://github.com/iterative/dvc?utm_source=ai-engineering-toolkit&utm_medium=github&utm_campaign=model-management) | Data version control for ML projects | Python | Apache-2.0 | [![GitHub stars](https://img.shields.io/github/stars/iterative/dvc.svg)](https://github.com/iterative/dvc/stargazers) |

### Data Collection & Web Scraping

| Tool | Description | Language | License | Stars |
|------|-------------|----------|---------|-------|
| [Firecrawl](https://github.com/mendableai/firecrawl?utm_source=ai-engineering-toolkit&utm_medium=github&utm_campaign=data-collection) | AI-powered web crawler that extracts and structures content for LLM pipelines | TypeScript | MIT | [![GitHub stars](https://img.shields.io/github/stars/mendableai/firecrawl.svg)](https://github.com/mendableai/firecrawl/stargazers) |
| [Scrapy](https://github.com/scrapy/scrapy?utm_source=ai-engineering-toolkit&utm_medium=github&utm_campaign=data-collection) | Fast, high-level web crawling & scraping framework | Python | BSD-3 | [![GitHub stars](https://img.shields.io/github/stars/scrapy/scrapy.svg)](https://github.com/scrapy/scrapy/stargazers) |
| [Playwright](https://github.com/microsoft/playwright?utm_source=ai-engineering-toolkit&utm_medium=github&utm_campaign=data-collection) | Web automation & scraping with headless browsers | TypeScript/Python/Java/.NET | Apache-2.0 | [![GitHub stars](https://img.shields.io/github/stars/microsoft/playwright.svg)](https://github.com/microsoft/playwright/stargazers) |
| [BeautifulSoup](https://www.crummy.com/software/BeautifulSoup/?utm_source=ai-engineering-toolkit&utm_medium=github&utm_campaign=data-collection) | Easy HTML/XML parsing for quick scraping tasks | Python | MIT | – |
| [Selenium](https://github.com/SeleniumHQ/selenium?utm_source=ai-engineering-toolkit&utm_medium=github&utm_campaign=data-collection) | Browser automation framework (supports scraping) | Multiple | Apache-2.0 | [![GitHub stars](https://img.shields.io/github/stars/SeleniumHQ/selenium.svg)](https://github.com/SeleniumHQ/selenium/stargazers) |
| [Apify SDK](https://github.com/apify/apify-sdk-python?utm_source=ai-engineering-toolkit&utm_medium=github&utm_campaign=data-collection) | Web scraping & automation platform SDK | Python/JavaScript | Apache-2.0 | [![GitHub stars](https://img.shields.io/github/stars/apify/apify-sdk-python.svg)](https://github.com/apify/apify-sdk-python/stargazers) |
| [Newspaper3k](https://github.com/codelucas/newspaper?utm_source=ai-engineering-toolkit&utm_medium=github&utm_campaign=data-collection) | News & article extraction library | Python | MIT | [![GitHub stars](https://img.shields.io/github/stars/codelucas/newspaper.svg)](https://github.com/codelucas/newspaper/stargazers) |

## 🤖 Agent Frameworks

| Framework | Description | Language | License | Stars |
|-----------|-------------|----------|---------|-------|
| [AutoGen](https://github.com/microsoft/autogen?utm_source=ai-engineering-toolkit&utm_medium=github&utm_campaign=agent-frameworks) | Multi-agent conversation framework | Python | CC-BY-4.0 | [![GitHub stars](https://img.shields.io/github/stars/microsoft/autogen.svg)](https://github.com/microsoft/autogen/stargazers) |
| [CrewAI](https://github.com/joaomdmoura/crewAI?utm_source=ai-engineering-toolkit&utm_medium=github&utm_campaign=agent-frameworks) | Framework for orchestrating role-playing autonomous AI agents | Python | MIT | [![GitHub stars](https://img.shields.io/github/stars/joaomdmoura/crewAI.svg)](https://github.com/joaomdmoura/crewAI/stargazers) |
| [LangGraph](https://github.com/langchain-ai/langgraph?utm_source=ai-engineering-toolkit&utm_medium=github&utm_campaign=agent-frameworks) | Build resilient language agents as graphs | Python | MIT | [![GitHub stars](https://img.shields.io/github/stars/langchain-ai/langgraph.svg)](https://github.com/langchain-ai/langgraph/stargazers) |
| [AgentOps](https://github.com/AgentOps-AI/agentops?utm_source=ai-engineering-toolkit&utm_medium=github&utm_campaign=agent-frameworks) | Python SDK for AI agent monitoring, LLM cost tracking, benchmarking | Python | MIT | [![GitHub stars](https://img.shields.io/github/stars/AgentOps-AI/agentops.svg)](https://github.com/AgentOps-AI/agentops/stargazers) |
| [Swarm](https://github.com/openai/swarm?utm_source=ai-engineering-toolkit&utm_medium=github&utm_campaign=agent-frameworks) | Educational framework for exploring ergonomic, lightweight multi-agent orchestration | Python | MIT | [![GitHub stars](https://img.shields.io/github/stars/openai/swarm.svg)](https://github.com/openai/swarm/stargazers) |
| [Agency Swarm](https://github.com/VRSEN/agency-swarm) | An open-source agent framework designed to automate your workflows | Python | MIT | ![GitHub stars](https://img.shields.io/github/stars/VRSEN/agency-swarm.svg) |
| [Multi-Agent Systems](https://github.com/microsoft/multi-agent-systems) | Research into multi-agent systems and applications | Python | MIT | ![GitHub stars](https://img.shields.io/github/stars/microsoft/multi-agent-systems.svg) |

## 📦 LLM Development & Optimization

### LLM Training and Fine-Tuning

| Tool | Description | Language | License | Stars |
|------|-------------|----------|---------|-------|
| [PyTorch Lightning](https://github.com/Lightning-AI/pytorch-lightning?utm_source=ai-engineering-toolkit&utm_medium=github&utm_campaign=llm-training) | High-level PyTorch interface for LLMs | Python | Apache-2.0 | [![GitHub stars](https://img.shields.io/github/stars/Lightning-AI/pytorch-lightning.svg)](https://github.com/Lightning-AI/pytorch-lightning/stargazers) |
| [unsloth](https://github.com/unslothai/unsloth?utm_source=ai-engineering-toolkit&utm_medium=github&utm_campaign=llm-training) | Fine-tune LLMs faster with less memory | Python | Apache-2.0 | [![GitHub stars](https://img.shields.io/github/stars/unslothai/unsloth.svg)](https://github.com/unslothai/unsloth/stargazers) |
| [Axolotl](https://github.com/OpenAccess-AI-Collective/axolotl?utm_source=ai-engineering-toolkit&utm_medium=github&utm_campaign=llm-training) | Post-training pipeline for AI models | Python | Apache-2.0 | [![GitHub stars](https://img.shields.io/github/stars/OpenAccess-AI-Collective/axolotl.svg)](https://github.com/OpenAccess-AI-Collective/axolotl/stargazers) |
| [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory?utm_source=ai-engineering-toolkit&utm_medium=github&utm_campaign=llm-training) | Easy & efficient LLM fine-tuning | Python | Apache-2.0 | [![GitHub stars](https://img.shields.io/github/stars/hiyouga/LLaMA-Factory.svg)](https://github.com/hiyouga/LLaMA-Factory/stargazers) |
| [PEFT](https://github.com/huggingface/peft?utm_source=ai-engineering-toolkit&utm_medium=github&utm_campaign=llm-training) | Parameter-Efficient Fine-Tuning library | Python | Apache-2.0 | [![GitHub stars](https://img.shields.io/github/stars/huggingface/peft.svg)](https://github.com/huggingface/peft/stargazers) |
| [DeepSpeed](https://github.com/microsoft/DeepSpeed?utm_source=ai-engineering-toolkit&utm_medium=github&utm_campaign=llm-training) | Distributed training & inference optimization | Python | MIT | [![GitHub stars](https://img.shields.io/github/stars/microsoft/DeepSpeed.svg)](https://github.com/microsoft/DeepSpeed/stargazers) |
| [TRL](https://github.com/huggingface/trl?utm_source=ai-engineering-toolkit&utm_medium=github&utm_campaign=llm-training) | Train transformer LMs with reinforcement learning | Python | Apache-2.0 | [![GitHub stars](https://img.shields.io/github/stars/huggingface/trl.svg)](https://github.com/huggingface/trl/stargazers) |
| [Transformers](https://github.com/huggingface/transformers?utm_source=ai-engineering-toolkit&utm_medium=github&utm_campaign=llm-training) | Pretrained models for text, vision, and audio tasks | Python | Apache-2.0 | [![GitHub stars](https://img.shields.io/github/stars/huggingface/transformers.svg)](https://github.com/huggingface/transformers/stargazers) |
| [LLMBox](https://github.com/microsoft/LLMBox?utm_source=ai-engineering-toolkit&utm_medium=github&utm_campaign=llm-training) | Unified training pipeline & model evaluation | Python | MIT | [![GitHub stars](https://img.shields.io/github/stars/microsoft/LLMBox.svg)](https://github.com/microsoft/LLMBox/stargazers) |
| [LitGPT](https://github.com/Lightning-AI/LitGPT?utm_source=ai-engineering-toolkit&utm_medium=github&utm_campaign=llm-training) | Train and fine-tune LLMs lightning fast | Python | Apache-2.0 | [![GitHub stars](https://img.shields.io/github/stars/Lightning-AI/LitGPT.svg)](https://github.com/Lightning-AI/LitGPT/stargazers) |
| [Mergoo](https://github.com/mlfoundations/mergoo?utm_source=ai-engineering-toolkit&utm_medium=github&utm_campaign=llm-training) | Merge multiple LLM experts efficiently | Python | Apache-2.0 | [![GitHub stars](https://img.shields.io/github/stars/mlfoundations/mergoo.svg)](https://github.com/mlfoundations/mergoo/stargazers) |
| [Ludwig](https://github.com/ludwig-ai/ludwig?utm_source=ai-engineering-toolkit&utm_medium=github&utm_campaign=llm-training) | Low-code framework for custom LLMs | Python | Apache-2.0 | [![GitHub stars](https://img.shields.io/github/stars/ludwig-ai/ludwig.svg)](https://github.com/ludwig-ai/ludwig/stargazers) |
| [txtinstruct](https://github.com/allenai/txtinstruct?utm_source=ai-engineering-toolkit&utm_medium=github&utm_campaign=llm-training) | Framework for training instruction-tuned models | Python | Apache-2.0 | [![GitHub stars](https://img.shields.io/github/stars/allenai/txtinstruct.svg)](https://github.com/allenai/txtinstruct/stargazers) |
| [xTuring](https://github.com/stochasticai/xTuring?utm_source=ai-engineering-toolkit&utm_medium=github&utm_campaign=llm-training) | Fast fine-tuning of open-source LLMs | Python | Apache-2.0 | [![GitHub stars](https://img.shields.io/github/stars/stochasticai/xTuring.svg)](https://github.com/stochasticai/xTuring/stargazers) |
| [RL4LMs](https://github.com/allenai/RL4LMs?utm_source=ai-engineering-toolkit&utm_medium=github&utm_campaign=llm-training) | RL library to fine-tune LMs to human preferences | Python | Apache-2.0 | [![GitHub stars](https://img.shields.io/github/stars/allenai/RL4LMs.svg)](https://github.com/allenai/RL4LMs/stargazers) |
| [torchtune](https://github.com/pytorch/torchtune?utm_source=ai-engineering-toolkit&utm_medium=github&utm_campaign=llm-training) | PyTorch-native library for fine-tuning LLMs | Python | BSD-3 | [![GitHub stars](https://img.shields.io/github/stars/pytorch/torchtune.svg)](https://github.com/pytorch/torchtune/stargazers) |

## 📦 LLM App Templates

### RAG Applications

- **[Basic RAG Chatbot](./templates/rag-chatbot/)** - Simple RAG implementation with vector search
- **[Multi-Modal RAG](./templates/multimodal-rag/)** - RAG with support for documents, images, and audio
- **[Enterprise RAG](./templates/enterprise-rag/)** - Production-ready RAG with authentication and monitoring

### Conversational AI

- **[Memory-Enabled Chatbot](./templates/memory-chatbot/)** - Chatbot with conversation memory and context
- **[Function Calling Agent](./templates/function-agent/)** - Agent with tool use capabilities
- **[Multi-Turn Assistant](./templates/assistant/)** - Advanced conversational assistant

### Specialized Applications

- **[Code Generation Assistant](./templates/code-assistant/)** - AI-powered code generation and analysis
- **[Document Analysis Tool](./templates/doc-analyzer/)** - Extract insights from documents
- **[Content Generation Pipeline](./templates/content-pipeline/)** - Automated content creation workflow
- **[Web Scraper for RAG](./templates/web-scraper-rag/)** - Comprehensive web scraping pipeline for data collection

## 🚀 Infrastructure & Deployment

### Local Development & Serving

| Tool | Description | Language | License | Stars |
|------|-------------|----------|---------|-------|
| [Ollama](https://github.com/ollama/ollama) | Get up and running with large language models locally | Go | MIT | ![GitHub stars](https://img.shields.io/github/stars/ollama/ollama.svg) |
| [LM Studio](https://lmstudio.ai/) | Desktop app for running local LLMs | - | Commercial | - |
| [GPT4All](https://github.com/nomic-ai/gpt4all) | Open-source chatbot ecosystem | C++ | MIT | ![GitHub stars](https://img.shields.io/github/stars/nomic-ai/gpt4all.svg) |
| [LocalAI](https://github.com/mudler/LocalAI) | Self-hosted OpenAI-compatible API | Go | MIT | ![GitHub stars](https://img.shields.io/github/stars/mudler/LocalAI.svg) |

### Production Serving

| Tool | Description | Language | License | Stars |
|------|-------------|----------|---------|-------|
| [vLLM](https://github.com/vllm-project/vllm) | High-throughput and memory-efficient inference engine | Python | Apache-2.0 | ![GitHub stars](https://img.shields.io/github/stars/vllm-project/vllm.svg) |
| [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) | TensorRT toolbox for optimized LLM inference | Python/C++ | Apache-2.0 | ![GitHub stars](https://img.shields.io/github/stars/NVIDIA/TensorRT-LLM.svg) |
| [LMDeploy](https://github.com/InternLM/lmdeploy) | Toolkit for compressing, deploying, and serving LLMs | Python | Apache-2.0 | ![GitHub stars](https://img.shields.io/github/stars/InternLM/lmdeploy.svg) |
| [Text Generation Inference](https://github.com/huggingface/text-generation-inference) | Large Language Model Text Generation Inference | Rust/Python | Apache-2.0 | ![GitHub stars](https://img.shields.io/github/stars/huggingface/text-generation-inference.svg) |

### Inference Platforms

| Platform | Description | Pricing | Features |
|----------|-------------|---------|----------|
| [Clarifai](https://www.clarifai.com/) | AI platform for computer vision, NLP, and generative AI | Free tier + Pay-as-you-go | Pre-trained models, Model training, Workflow automation |
| [Modal](https://modal.com/) | Serverless platform for AI/ML workloads | Pay-per-use | Serverless GPU, Auto-scaling |
| [Replicate](https://replicate.com/) | Run open-source models with a cloud API | Pay-per-use | Pre-built models, Custom training |
| [Together AI](https://www.together.ai/) | Cloud platform for open-source models | Various | Open models, Fine-tuning |
| [Anyscale](https://www.anyscale.com/) | Ray-based platform for AI applications | Enterprise | Distributed training, Serving |

### Containerization & Orchestration

- **[Docker Templates](./infrastructure/docker/)** - Containerized LLM applications
- **[Kubernetes Manifests](./infrastructure/k8s/)** - K8s deployments for LLM services
- **[Terraform Modules](./infrastructure/terraform/)** - IaC for cloud deployments

## 📚 Tutorials & Notebooks

### Getting Started

- **[LLM Basics](./tutorials/01-llm-basics/)** - Understanding LLMs and prompt engineering
- **[Setting Up Your Environment](./tutorials/02-environment-setup/)** - Development environment configuration
- **[Your First LLM App](./tutorials/03-first-app/)** - Build a simple LLM application

### Intermediate Guides

- **[Building RAG Systems](./tutorials/04-rag-systems/)** - Comprehensive RAG implementation guide
- **[Agent Development](./tutorials/05-agent-development/)** - Creating intelligent agents
- **[Fine-tuning Models](./tutorials/06-fine-tuning/)** - Custom model training

### Advanced Topics

- **[Production Deployment](./tutorials/07-production/)** - Deploying LLM apps at scale
- **[Performance Optimization](./tutorials/08-optimization/)** - Optimizing inference and costs
- **[Multi-Agent Systems](./tutorials/09-multi-agent/)** - Building complex agent networks

### Jupyter Notebooks

- **[Quick Start Notebooks](./notebooks/quickstart/)** - Interactive tutorials
- **[Use Case Examples](./notebooks/examples/)** - Real-world implementations
- **[Benchmarking](./notebooks/benchmarks/)** - Performance comparisons

## 📰 Newsletter Companion Demos

Code examples and configurations featured in the AI Engineering newsletter:

- **[Latest Demos](./newsletter-demos/latest/)** - Most recent newsletter content
- **[Tool Reviews](./newsletter-demos/tool-reviews/)** - Hands-on tool evaluations
- **[Architecture Deep Dives](./newsletter-demos/architecture/)** - System design examples
- **[Performance Studies](./newsletter-demos/performance/)** - Benchmarking and optimization

## 🤝 Contributing

We welcome contributions! This toolkit grows stronger with community input.

### How to Contribute

1. **Fork the repository**
2. **Create a feature branch** (`git checkout -b feature/amazing-tool`)
3. **Add your contribution** (new tool, template, or tutorial)
4. **Follow our guidelines** (see [CONTRIBUTING.md](./CONTRIBUTING.md))
5. **Submit a pull request**

### Contribution Guidelines

- **Quality over quantity** - Focus on tools and resources that provide real value
- **Production-ready** - Include tools that work in real-world scenarios
- **Well-documented** - Provide clear descriptions and usage examples
- **Up-to-date** - Ensure tools are actively maintained

### Areas We Need Help With

- [ ] More production deployment examples
- [ ] Additional evaluation frameworks
- [ ] Multi-modal AI tools
- [ ] Cost optimization strategies
- [ ] Security and privacy tools

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 📧 Stay Connected

### Newsletter
Get weekly AI engineering insights, tool reviews, and exclusive demos delivered to your inbox:

**[📧 Subscribe to AI Engineering Newsletter →](https://aiengineering.substack.com/subscribe?utm_source=ai-engineering-toolkit&utm_medium=github&utm_campaign=readme-footer)**

*Join 10,000+ engineers building better LLM applications*

### Social Media
[![Twitter Follow](https://img.shields.io/twitter/follow/ai_engineering?style=social)](https://twitter.com/ai_engineering?utm_source=ai-engineering-toolkit&utm_medium=github&utm_campaign=social)
[![LinkedIn Follow](https://img.shields.io/badge/LinkedIn-Follow-blue?style=social&logo=linkedin)](https://linkedin.com/in/yourusername?utm_source=ai-engineering-toolkit&utm_medium=github&utm_campaign=social)

### Share This Repository
[![Share on Twitter](https://img.shields.io/twitter/url?url=https%3A//github.com/yourusername/ai-engineering-toolkit&style=social)](https://twitter.com/intent/tweet?url=https%3A//github.com/yourusername/ai-engineering-toolkit&text=🧰%20AI%20Engineering%20Toolkit%20-%20Build%20better%20LLM%20apps%20faster&via=ai_engineering&hashtags=AI,LLM,Engineering,OpenSource)

[![Share on LinkedIn](https://img.shields.io/badge/Share%20on-LinkedIn-blue?style=social&logo=linkedin)](https://www.linkedin.com/sharing/share-offsite/?url=https%3A//github.com/yourusername/ai-engineering-toolkit&summary=AI%20Engineering%20Toolkit%20-%20A%20curated%20collection%20of%20tools%20and%20resources%20for%20building%20production-ready%20LLM%20applications)

[![Share on Reddit](https://img.shields.io/badge/Share%20on-Reddit-orange?style=social&logo=reddit)](https://reddit.com/submit?url=https%3A//github.com/yourusername/ai-engineering-toolkit&title=AI%20Engineering%20Toolkit%20-%20Build%20better%20LLM%20apps)

---

**Built with ❤️ for the AI Engineering community**

*Star ⭐ this repo if you find it helpful!*
