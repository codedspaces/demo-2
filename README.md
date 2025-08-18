
# AI Engineering Toolkit🔥

**Build better LLM apps — faster, smarter, production-ready.**

A curated, list of 100+ libraries and frameworks for AI engineers building with Large Language Models. This toolkit includes battle-tested tools, frameworks, templates, and reference implementations for developing, deploying, and optimizing LLM-powered systems.

[![Toolkit banner](https://github.com/codedspaces/demo-2/blob/d9442b179eba2856e8c6e62bb1c6a1bb8c676b89/2.jpg?raw=true)](https://aiengineering.beehiiv.com/subscribe)

<p align="center">
  <a href="https://opensource.org/licenses/MIT">
    <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT">
  </a>
  <a href="https://x.com/Sumanth_077">
    <img src="https://img.shields.io/twitter/follow/Sumanth_077?style=social&logo=x" alt="X Follow">
  </a>
  <a href="https://www.linkedin.com/company/theaiengineering/">
    <img src="https://img.shields.io/badge/LinkedIn-Connect-blue?logo=linkedin" alt="LinkedIn">
  </a>
</p>

## 📋 Table of Contents

- [🛠️ Tooling for AI Engineers](#%EF%B8%8F-tooling-for-ai-engineers)
  - [Vector Databases](#vector-databases)
  - [Orchestration & Workflows](#orchestration--workflows)
  - [RAG (Retrieval-Augmented Generation)](#rag-retrieval-augmented-generation)
  - [Evaluation & Testing](#evaluation--testing)
  - [Model Management](#model-management)
  - [Data Collection & Web Scraping](#data-collection--web-scraping)
- [🤖 Agent Frameworks](#-agent-frameworks)
- [📦 LLM Development & Optimization](#llm-development--optimization)
  - [LLM Inference](#llm-inference)
  - [LLM Safety & Security](#llm-safety--security)
  - [Prototyping & Enhancement Tools](#prototyping--enhancement-tools)
- [🚀 Infrastructure & Deployment](#-infrastructure--deployment)
  - [Local Development & Serving](#local-development--serving)
  - [Production Serving](#production-serving)
  - [Cloud Inference Platforms](#cloud-inference-platforms)
- [🤝 Contributing](#-contributing)
- [📄 License](#-license)

## 🛠️ Tooling for AI Engineers

### Vector Databases

| Tool | Description | Language | License |
|------|-------------|----------|---------|
| [Pinecone](https://www.pinecone.io/?utm_source=ai-engineering-toolkit&utm_medium=github&utm_campaign=vector-db) | Managed vector database for production AI applications | API/SDK | Commercial |
| [Weaviate](https://github.com/weaviate/weaviate?utm_source=ai-engineering-toolkit&utm_medium=github&utm_campaign=vector-db) | Open-source vector database with GraphQL API | Go | BSD-3 | 
| [Qdrant](https://github.com/qdrant/qdrant?utm_source=ai-engineering-toolkit&utm_medium=github&utm_campaign=vector-db) | Vector similarity search engine with extended filtering | Rust | Apache-2.0 |
| [Chroma](https://github.com/chroma-core/chroma?utm_source=ai-engineering-toolkit&utm_medium=github&utm_campaign=vector-db) | Open-source embedding database for LLM apps | Python | Apache-2.0 |
| [Milvus](https://github.com/milvus-io/milvus?utm_source=ai-engineering-toolkit&utm_medium=github&utm_campaign=vector-db) | Cloud-native vector database for scalable similarity search | Go/C++ | Apache-2.0 | 
| [FAISS](https://github.com/facebookresearch/faiss?utm_source=ai-engineering-toolkit&utm_medium=github&utm_campaign=vector-db) | Library for efficient similarity search and clustering | C++/Python | MIT | 

### Orchestration & Workflows

| Tool | Description | Language | License | 
|------|-------------|----------|---------|
| [LangChain](https://github.com/langchain-ai/langchain?utm_source=ai-engineering-toolkit&utm_medium=github&utm_campaign=orchestration) | Framework for developing LLM applications | Python/JS | MIT | 
| [LlamaIndex](https://github.com/run-llama/llama_index?utm_source=ai-engineering-toolkit&utm_medium=github&utm_campaign=orchestration) | Data framework for LLM applications | Python | MIT | 
| [Haystack](https://github.com/deepset-ai/haystack?utm_source=ai-engineering-toolkit&utm_medium=github&utm_campaign=orchestration) | End-to-end NLP framework for production | Python | Apache-2.0 | 
| [DSPy](https://github.com/stanfordnlp/dspy?utm_source=ai-engineering-toolkit&utm_medium=github&utm_campaign=orchestration) | Framework for algorithmically optimizing LM prompts | Python | MIT |
| [Semantic Kernel](https://github.com/microsoft/semantic-kernel?utm_source=ai-engineering-toolkit&utm_medium=github&utm_campaign=orchestration) | SDK for integrating AI into conventional programming languages | C#/Python/Java | MIT | 
| [Langflow](https://github.com/langflow-ai/langflow) | Visual no-code platform for building and deploying LLM workflows | Python/TypeScript | MIT |
| [Flowise](https://github.com/FlowiseAI/Flowise) | Drag-and-drop UI for creating LLM chains and agents | TypeScript | MIT |

### PDF Extraction Tools

| Tool | Description | Language | License |
|------|-------------|----------|---------|
| [Docling](https://github.com/docling-project/docling) | AI-powered toolkit converting PDF, DOCX, PPTX, HTML, images into structured JSON/Markdown with layout, OCR, table, and code recognition | Python | MIT |
| [pdfplumber](https://github.com/jsvine/pdfplumber) | Drill through PDFs at a character level, extract text & tables, and visually debug extraction | Python | MIT | 
| [PyMuPDF (fitz)](https://github.com/pymupdf/PyMuPDF) | Lightweight, high-performance PDF parser for text/image extraction and manipulation | Python / C | AGPL-3.0 |
| [PDF.js](https://github.com/mozilla/pdf.js) | Browser-based PDF renderer with text extraction capabilities | JavaScript | Apache-2.0 | 
| [Camelot](https://github.com/camelot-dev/camelot) | Extracts structured tabular data from PDFs into DataFrames and CSVs | Python | MIT |

### RAG (Retrieval-Augmented Generation)

| Tool | Description | Language | License |
|------|-------------|----------|---------|
| [RAGFlow](https://github.com/infiniflow/ragflow) | Open-source RAG engine based on deep document understanding | Python | Apache-2.0 | 
| [Verba](https://github.com/weaviate/Verba) | Retrieval Augmented Generation (RAG) chatbot | Python | BSD-3 | 
| [PrivateGPT](https://github.com/imartinez/privateGPT) | Interact with documents using local LLMs | Python | Apache-2.0 | 
| [AnythingLLM](https://github.com/Mintplex-Labs/anything-llm) | All-in-one AI application for any LLM | JavaScript | MIT |
| [Quivr](https://github.com/QuivrHQ/quivr) | Your GenAI second brain | Python/TypeScript | Apache-2.0 |
| [Jina](https://github.com/jina-ai/jina) | Cloud-native neural search framework for multimodal RAG | Python | Apache-2.0 |
| [txtai](https://github.com/neuml/txtai) | All-in-one embeddings database for semantic search and workflows | Python | Apache-2.0 |

### Evaluation & Testing

| Tool | Description | Language | License |
|------|-------------|----------|---------|
| [Ragas](https://github.com/explodinggradients/ragas) | Evaluation framework for RAG pipelines | Python | Apache-2.0 |
| [LangSmith](https://smith.langchain.com/) | Platform for debugging, testing, and monitoring LLM applications | API/SDK | Commercial |
| [Phoenix](https://github.com/Arize-ai/phoenix) | ML observability for LLM, vision, language, and tabular models | Python | Apache-2.0 |
| [DeepEval](https://github.com/confident-ai/deepeval) | LLM evaluation framework for unit testing LLM outputs | Python | Apache-2.0 |
| [TruLens](https://github.com/truera/trulens) | Evaluation and tracking for LLM experiments | Python | MIT |
| [Inspect](https://github.com/ukaisi/inspect) | Framework for large language model evaluations | Python | Apache-2.0 |
| [UpTrain](https://github.com/uptrain-ai/uptrain) | Open-source tool to evaluate and improve LLM applications | Python | Apache-2.0 |

### Model Management

| Tool | Description | Language | License |
|------|-------------|----------|---------|
| [Hugging Face Hub](https://github.com/huggingface/huggingface_hub?utm_source=ai-engineering-toolkit&utm_medium=github&utm_campaign=model-management) | Client library for Hugging Face Hub | Python | Apache-2.0 | 
| [MLflow](https://github.com/mlflow/mlflow?utm_source=ai-engineering-toolkit&utm_medium=github&utm_campaign=model-management) | Platform for ML lifecycle management | Python | Apache-2.0 |
| [Weights & Biases](https://github.com/wandb/wandb?utm_source=ai-engineering-toolkit&utm_medium=github&utm_campaign=model-management) | Developer tools for ML | Python | MIT |
| [DVC](https://github.com/iterative/dvc?utm_source=ai-engineering-toolkit&utm_medium=github&utm_campaign=model-management) | Data version control for ML projects | Python | Apache-2.0 |
| [Comet ML](https://github.com/comet-ml/comet-ml) | Experiment tracking and visualization for ML/LLM workflows | Python | MIT |
| [ClearML](https://github.com/allegroai/clearml) | End-to-end MLOps platform with LLM support | Python | Apache-2.0 |

### Data Collection & Web Scraping

| Tool | Description | Language | License |
|------|-------------|----------|---------|
| [Firecrawl](https://github.com/mendableai/firecrawl?utm_source=ai-engineering-toolkit&utm_medium=github&utm_campaign=data-collection) | AI-powered web crawler that extracts and structures content for LLM pipelines | TypeScript | MIT |
| [Scrapy](https://github.com/scrapy/scrapy?utm_source=ai-engineering-toolkit&utm_medium=github&utm_campaign=data-collection) | Fast, high-level web crawling & scraping framework | Python | BSD-3 |
| [Playwright](https://github.com/microsoft/playwright?utm_source=ai-engineering-toolkit&utm_medium=github&utm_campaign=data-collection) | Web automation & scraping with headless browsers | TypeScript/Python/Java/.NET | Apache-2.0 | 
| [BeautifulSoup](https://www.crummy.com/software/BeautifulSoup/?utm_source=ai-engineering-toolkit&utm_medium=github&utm_campaign=data-collection) | Easy HTML/XML parsing for quick scraping tasks | Python | MIT |
| [Selenium](https://github.com/SeleniumHQ/selenium?utm_source=ai-engineering-toolkit&utm_medium=github&utm_campaign=data-collection) | Browser automation framework (supports scraping) | Multiple | Apache-2.0 |
| [Apify SDK](https://github.com/apify/apify-sdk-python?utm_source=ai-engineering-toolkit&utm_medium=github&utm_campaign=data-collection) | Web scraping & automation platform SDK | Python/JavaScript | Apache-2.0 |
| [Newspaper3k](https://github.com/codelucas/newspaper?utm_source=ai-engineering-toolkit&utm_medium=github&utm_campaign=data-collection) | News & article extraction library | Python | MIT |

## 🤖 Agent Frameworks

| Framework | Description | Language | License |
|-----------|-------------|----------|---------|
| [AutoGen](https://github.com/microsoft/autogen?utm_source=ai-engineering-toolkit&utm_medium=github&utm_campaign=agent-frameworks) | Multi-agent conversation framework | Python | CC-BY-4.0 | 
| [CrewAI](https://github.com/joaomdmoura/crewAI?utm_source=ai-engineering-toolkit&utm_medium=github&utm_campaign=agent-frameworks) | Framework for orchestrating role-playing autonomous AI agents | Python | MIT | 
| [LangGraph](https://github.com/langchain-ai/langgraph?utm_source=ai-engineering-toolkit&utm_medium=github&utm_campaign=agent-frameworks) | Build resilient language agents as graphs | Python | MIT |
| [AgentOps](https://github.com/AgentOps-AI/agentops?utm_source=ai-engineering-toolkit&utm_medium=github&utm_campaign=agent-frameworks) | Python SDK for AI agent monitoring, LLM cost tracking, benchmarking | Python | MIT |
| [Swarm](https://github.com/openai/swarm?utm_source=ai-engineering-toolkit&utm_medium=github&utm_campaign=agent-frameworks) | Educational framework for exploring ergonomic, lightweight multi-agent orchestration | Python | MIT | 
| [Agency Swarm](https://github.com/VRSEN/agency-swarm) | An open-source agent framework designed to automate your workflows | Python | MIT | 
| [Multi-Agent Systems](https://github.com/microsoft/multi-agent-systems) | Research into multi-agent systems and applications | Python | MIT | 
| [Auto-GPT](https://github.com/Significant-Gravitas/Auto-GPT) | Autonomous AI agent for task execution using GPT models | Python | MIT |
| [BabyAGI](https://github.com/yoheinakajima/babyagi) | Task-driven autonomous agent inspired by AGI | Python | MIT |
| [SuperAGI](https://github.com/TransformerOptimus/SuperAGI) | Infrastructure for building and managing autonomous agents | Python | MIT |
| [Phidata](https://github.com/phidatahq/phidata) | Build AI agents with memory, tools, and knowledge | Python | MIT |
| [MemGPT](https://github.com/cpacker/MemGPT) | Self-improving agents with infinite context via memory management | Python | MIT |

## 📦 LLM Development & Optimization

### LLM Training and Fine-Tuning

| Tool | Description | Language | License |
|------|-------------|----------|---------|
| [PyTorch Lightning](https://github.com/Lightning-AI/pytorch-lightning?utm_source=ai-engineering-toolkit&utm_medium=github&utm_campaign=llm-training) | High-level PyTorch interface for LLMs | Python | Apache-2.0 | 
| [unsloth](https://github.com/unslothai/unsloth?utm_source=ai-engineering-toolkit&utm_medium=github&utm_campaign=llm-training) | Fine-tune LLMs faster with less memory | Python | Apache-2.0 |
| [Axolotl](https://github.com/OpenAccess-AI-Collective/axolotl?utm_source=ai-engineering-toolkit&utm_medium=github&utm_campaign=llm-training) | Post-training pipeline for AI models | Python | Apache-2.0 |
| [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory?utm_source=ai-engineering-toolkit&utm_medium=github&utm_campaign=llm-training) | Easy & efficient LLM fine-tuning | Python | Apache-2.0 |
| [PEFT](https://github.com/huggingface/peft?utm_source=ai-engineering-toolkit&utm_medium=github&utm_campaign=llm-training) | Parameter-Efficient Fine-Tuning library | Python | Apache-2.0 |
| [DeepSpeed](https://github.com/microsoft/DeepSpeed?utm_source=ai-engineering-toolkit&utm_medium=github&utm_campaign=llm-training) | Distributed training & inference optimization | Python | MIT | 
| [TRL](https://github.com/huggingface/trl?utm_source=ai-engineering-toolkit&utm_medium=github&utm_campaign=llm-training) | Train transformer LMs with reinforcement learning | Python | Apache-2.0 |
| [Transformers](https://github.com/huggingface/transformers?utm_source=ai-engineering-toolkit&utm_medium=github&utm_campaign=llm-training) | Pretrained models for text, vision, and audio tasks | Python | Apache-2.0 |
| [LLMBox](https://github.com/microsoft/LLMBox?utm_source=ai-engineering-toolkit&utm_medium=github&utm_campaign=llm-training) | Unified training pipeline & model evaluation | Python | MIT | 
| [LitGPT](https://github.com/Lightning-AI/LitGPT?utm_source=ai-engineering-toolkit&utm_medium=github&utm_campaign=llm-training) | Train and fine-tune LLMs lightning fast | Python | Apache-2.0 |
| [Mergoo](https://github.com/mlfoundations/mergoo?utm_source=ai-engineering-toolkit&utm_medium=github&utm_campaign=llm-training) | Merge multiple LLM experts efficiently | Python | Apache-2.0 | 
| [Ludwig](https://github.com/ludwig-ai/ludwig?utm_source=ai-engineering-toolkit&utm_medium=github&utm_campaign=llm-training) | Low-code framework for custom LLMs | Python | Apache-2.0 |
| [txtinstruct](https://github.com/allenai/txtinstruct?utm_source=ai-engineering-toolkit&utm_medium=github&utm_campaign=llm-training) | Framework for training instruction-tuned models | Python | Apache-2.0 |
| [xTuring](https://github.com/stochasticai/xTuring?utm_source=ai-engineering-toolkit&utm_medium=github&utm_campaign=llm-training) | Fast fine-tuning of open-source LLMs | Python | Apache-2.0 |
| [RL4LMs](https://github.com/allenai/RL4LMs?utm_source=ai-engineering-toolkit&utm_medium=github&utm_campaign=llm-training) | RL library to fine-tune LMs to human preferences | Python | Apache-2.0 |
| [torchtune](https://github.com/pytorch/torchtune?utm_source=ai-engineering-toolkit&utm_medium=github&utm_campaign=llm-training) | PyTorch-native library for fine-tuning LLMs | Python | BSD-3 |
| [Accelerate](https://github.com/huggingface/accelerate) | Library to easily train on multiple GPUs/TPUs with mixed precision | Python | Apache-2.0 |
| [BitsandBytes](https://github.com/TimDettmers/bitsandbytes) | 8-bit optimizers and quantization for efficient LLM training | Python | MIT |

### LLM Inference

| Tool | Description | Language | License | 
|------|-------------|----------|---------|
| [LLM Compressor](https://github.com/mit-han-lab/llm-compressor?utm_source=ai-engineering-toolkit&utm_medium=github&utm_campaign=llm-inference) | Transformers-compatible library for applying various compression algorithms to LLMs for optimized deployment | Python | Apache-2.0 |
| [LightLLM](https://github.com/ModelTC/lightllm?utm_source=ai-engineering-toolkit&utm_medium=github&utm_campaign=llm-inference) | Lightweight Python-based LLM inference and serving framework with easy scalability and high performance | Python | Apache-2.0 |
| [vLLM](https://github.com/vllm-project/vllm?utm_source=ai-engineering-toolkit&utm_medium=github&utm_campaign=llm-inference) | High-throughput and memory-efficient inference and serving engine for LLMs | Python | Apache-2.0 |
| [torchchat](https://github.com/facebookresearch/torchchat?utm_source=ai-engineering-toolkit&utm_medium=github&utm_campaign=llm-inference) | Run PyTorch LLMs locally on servers, desktop, and mobile | Python | MIT |
| [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM?utm_source=ai-engineering-toolkit&utm_medium=github&utm_campaign=llm-inference) | NVIDIA library for optimizing LLM inference with TensorRT | C++/Python | Apache-2.0 |
| [WebLLM](https://github.com/mlc-ai/web-llm?utm_source=ai-engineering-toolkit&utm_medium=github&utm_campaign=llm-inference) | High-performance in-browser LLM inference engine | TypeScript/Python | Apache-2.0 |

### LLM Safety and Security

| Tool | Description | Language | License |
|------|-------------|----------|---------|
| [JailbreakEval](https://github.com/centerforaisafety/JailbreakEval?utm_source=ai-engineering-toolkit&utm_medium=github&utm_campaign=llm-safety) | Automated evaluators for assessing jailbreak attempts | Python | MIT |
| [EasyJailbreak](https://github.com/thu-coai/EasyJailbreak?utm_source=ai-engineering-toolkit&utm_medium=github&utm_campaign=llm-safety) | Easy-to-use Python framework to generate adversarial jailbreak prompts | Python | Apache-2.0 |
| [Guardrails](https://github.com/ShreyaR/guardrails?utm_source=ai-engineering-toolkit&utm_medium=github&utm_campaign=llm-safety) | Add guardrails to large language models | Python | MIT |
| [LLM Guard](https://github.com/deadbits/llm-guard?utm_source=ai-engineering-toolkit&utm_medium=github&utm_campaign=llm-safety) | Security toolkit for LLM interactions | Python | Apache-2.0 |
| [AuditNLG](https://github.com/Alex-Fabbri/AuditNLG?utm_source=ai-engineering-toolkit&utm_medium=github&utm_campaign=llm-safety) | Reduce risks in generative AI systems for language | Python | MIT |
| [NeMo Guardrails](https://github.com/NVIDIA/NeMo-Guardrails?utm_source=ai-engineering-toolkit&utm_medium=github&utm_campaign=llm-safety) | Toolkit for adding programmable guardrails to LLM conversational systems | Python | Apache-2.0 |
| [Garak](https://github.com/leondz/garak?utm_source=ai-engineering-toolkit&utm_medium=github&utm_campaign=llm-safety) | LLM vulnerability scanner | Python | MIT |
| [DeepTeam](https://github.com/DeepTeamAI/deepteam?utm_source=ai-engineering-toolkit&utm_medium=github&utm_campaign=llm-safety) | LLM red teaming framework | Python | Apache-2.0 |

### Prototyping & Enhancement Tools

| Tool | Description | Language | License |
|------|-------------|----------|---------|
| [Text Machina](https://github.com/Genaios/TextMachina) | Modular Python framework for building unbiased datasets for MGT-related tasks (detection, attribution, boundary detection) | Python | - |
| [LLM Reasoners](https://github.com/maitrix-org/llm-reasoners) | Library for advanced large language model reasoning | Python | - |
| [EasyEdit](https://github.com/zjunlp/EasyEdit) | Easy-to-use knowledge editing framework for LLMs | Python | Apache-2.0 |
| [CodeTF](https://github.com/salesforce/CodeTF) | One-stop Transformer Library for state-of-the-art Code LLM | Python | Apache-2.0 |
[Reflex](https://github.com/reflex-dev/reflex) | Python framework for building full-stack web apps for LLM interfaces and dashboards | Python | Apache-2.0 |
| [spacy-llm](https://github.com/explosion/spacy-llm) | Integrates LLMs into spaCy for prototyping, prompting, and structured NLP outputs | Python | MIT |
| [pandas-ai](https://github.com/Sinaptik-AI/pandas-ai) | Chat with your database (SQL, CSV, pandas, polars, MongoDB, NoSQL, etc.) | Python | MIT |
| [LLM Transparency Tool](https://github.com/facebookresearch/llm-transparency-tool) | Interactive toolkit for analyzing internal workings of Transformer-based LMs | Python | - |
| [Vanna](https://github.com/vanna-ai/vanna) | Text-to-SQL generation via LLMs using RAG for accurate database queries | Python | Apache-2.0 |
| [mergekit](https://github.com/arcee-ai/MergeKit) | Tools for merging pretrained LLMs | Python | Apache-2.0 |
| [MarkLLM](https://github.com/THU-BPM/MarkLLM) | Toolkit for LLM watermarking | Python | - |
| [LLMSanitize](https://github.com/ntunlp/LLMSanitize) | Contamination detection in NLP datasets and LLMs | Python | - |
| [Annotateai](https://github.com/neuml/annotateai) | Automatically annotate papers using LLMs | Python | - |
| [LLM Reasoner](https://github.com/harishsg993010/LLM-Reasoner) | Makes any LLM think like OpenAI o1 and DeepSeek R1 | Python | - |
| [Guidance](https://github.com/guidance-ai/guidance) | Guidance language for controlling large language models | Python | MIT |
| [Gradio](https://github.com/gradio-app/gradio) | Quickly create UIs for LLM demos and prototypes | Python | Apache-2.0 |

## 🚀 Infrastructure & Deployment

### Local Development & Serving

| Tool | Description | Language | License |
|------|-------------|----------|---------|
| [Ollama](https://github.com/ollama/ollama) | Get up and running with large language models locally | Go | MIT |
| [LM Studio](https://lmstudio.ai/) | Desktop app for running local LLMs | - | Commercial |
| [GPT4All](https://github.com/nomic-ai/gpt4all) | Open-source chatbot ecosystem | C++ | MIT |
| [LocalAI](https://github.com/mudler/LocalAI) | Self-hosted OpenAI-compatible API | Go | MIT |

### Production Serving

| Tool | Description | Language | License |
|------|-------------|----------|---------|
| [vLLM](https://github.com/vllm-project/vllm) | High-throughput and memory-efficient inference engine | Python | Apache-2.0 |
| [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) | TensorRT toolbox for optimized LLM inference | Python/C++ | Apache-2.0 |
| [LMDeploy](https://github.com/InternLM/lmdeploy) | Toolkit for compressing, deploying, and serving LLMs | Python | Apache-2.0 |
| [Text Generation Inference](https://github.com/huggingface/text-generation-inference) | Large Language Model Text Generation Inference | Rust/Python | Apache-2.0 |
| [BentoML](https://github.com/bentoml/BentoML) | Framework for serving and deploying ML/LLM models | Python | Apache-2.0 |
| [Ray Serve](https://github.com/ray-project/ray) | Scalable model serving library built on Ray | Python | Apache-2.0 |

### Cloud Inference Platforms

| Platform | Description | Pricing | Features |
|----------|-------------|---------|----------|
| [Clarifai](https://www.clarifai.com/) | AI platform for computer vision, NLP, and generative AI | Free tier + Pay-as-you-go | Pre-trained models, Model training, Workflow automation |
| [Modal](https://modal.com/) | Serverless platform for AI/ML workloads | Pay-per-use | Serverless GPU, Auto-scaling |
| [Replicate](https://replicate.com/) | Run open-source models with a cloud API | Pay-per-use | Pre-built models, Custom training |
| [Together AI](https://www.together.ai/) | Cloud platform for open-source models | Various | Open models, Fine-tuning |
| [Anyscale](https://www.anyscale.com/) | Ray-based platform for AI applications | Enterprise | Distributed training, Serving |
| [Google Vertex AI](https://cloud.google.com/vertex-ai) | Managed platform for building, deploying, and scaling AI models | Pay-as-you-go | Model tuning, RAG pipelines, enterprise security |

## 🤝 Contributing

We welcome contributions! This toolkit grows stronger with community input.

### How to Contribute

1. **Fork the repository**
2. **Create a feature branch** (`git checkout -b feature/amazing-tool`)
3. **Add your contribution** (new tool, template, or tutorial)
4. **Submit a pull request**

### Contribution Guidelines

- **Quality over quantity** - Focus on tools and resources that provide real value
- **Production-ready** - Include tools that work in real-world scenarios
- **Well-documented** - Provide clear descriptions and usage examples
- **Up-to-date** - Ensure tools are actively maintained

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](https://github.com/codedspaces/demo-2/blob/cursor/setup-ai-engineering-toolkit-repository-028a/LICENSE) file for details.

---

## 📧 Stay Connected

### Newsletter
Get weekly AI engineering insights, tool reviews, and exclusive demos delivered to your inbox:

**[📧 Subscribe to AI Engineering Newsletter →](https://aiengineering.beehiiv.com/subscribe)**

*Join 100,000+ engineers building better LLM applications*

### Social Media
[![X Follow](https://img.shields.io/twitter/follow/Sumanth_077?style=social&logo=x)](https://x.com/Sumanth_077)
[![LinkedIn Follow](https://img.shields.io/badge/LinkedIn-Follow-blue?style=social&logo=linkedin)](https://www.linkedin.com/company/theaiengineering/)

---


---

**Built with ❤️ for the AI Engineering community**

*Star ⭐ this repo if you find it helpful!*
