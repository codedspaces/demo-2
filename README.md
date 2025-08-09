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
| [Hugging Face Hub](https://github.com/huggingface/huggingface_hub) | Client library for Hugging Face Hub | Python | Apache-2.0 | ![GitHub stars](https://img.shields.io/github/stars/huggingface/huggingface_hub.svg) |
| [MLflow](https://github.com/mlflow/mlflow) | Platform for ML lifecycle management | Python | Apache-2.0 | ![GitHub stars](https://img.shields.io/github/stars/mlflow/mlflow.svg) |
| [Weights & Biases](https://github.com/wandb/wandb) | Developer tools for ML | Python | MIT | ![GitHub stars](https://img.shields.io/github/stars/wandb/wandb.svg) |
| [DVC](https://github.com/iterative/dvc) | Data version control for ML projects | Python | Apache-2.0 | ![GitHub stars](https://img.shields.io/github/stars/iterative/dvc.svg) |

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

### Cloud Platforms

| Platform | Description | Pricing | Features |
|----------|-------------|---------|----------|
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
