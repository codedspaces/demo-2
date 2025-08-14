# LLM Basics & Prompt Engineering

A comprehensive guide to understanding Large Language Models and mastering prompt engineering techniques.

## Table of Contents

1. [Understanding LLMs](#understanding-llms)
2. [Key Concepts](#key-concepts)
3. [Prompt Engineering Fundamentals](#prompt-engineering-fundamentals)
4. [Advanced Techniques](#advanced-techniques)
5. [Hands-on Examples](#hands-on-examples)
6. [Best Practices](#best-practices)

## Understanding LLMs

### What are Large Language Models?

Large Language Models (LLMs) are AI systems trained on vast amounts of text data to understand and generate human-like text. They work by predicting the next word in a sequence based on statistical patterns learned during training.

```
Input: "The capital of France is"
LLM Process: [Analyzes patterns] → High probability: "Paris"
Output: "Paris"
```

### How LLMs Work

1. **Tokenization**: Text is broken into tokens (words/subwords)
2. **Embedding**: Tokens are converted to numerical vectors
3. **Attention**: Model focuses on relevant parts of the input
4. **Generation**: Probability distribution over next tokens
5. **Sampling**: Select next token based on strategy

### Popular LLM Architectures

| Model Family | Examples | Strengths | Use Cases |
|--------------|----------|-----------|-----------|
| GPT | GPT-4, GPT-3.5 | General purpose, conversational | Chatbots, content generation |
| Claude | Claude-3, Claude-2 | Reasoning, safety | Analysis, research assistance |
| Llama | Llama 2, Code Llama | Open source, customizable | Local deployment, fine-tuning |
| Gemini | Gemini Pro, Ultra | Multimodal, reasoning | Complex analysis, vision tasks |

## Key Concepts

### Temperature & Sampling

**Temperature** controls randomness in output generation:

```python
# Low temperature (0.1) - More deterministic
response = llm.generate("Explain photosynthesis", temperature=0.1)
# Output: Consistent, factual explanations

# High temperature (0.9) - More creative
response = llm.generate("Write a story about a robot", temperature=0.9)
# Output: Varied, creative narratives
```

### Context Window

The maximum number of tokens an LLM can process at once:

- **GPT-4**: 8K-128K tokens
- **Claude-3**: 200K tokens
- **Llama 2**: 4K tokens

### Tokens vs Words

```python
# Example tokenization
text = "Hello, world!"
tokens = ["Hello", ",", " world", "!"]  # 4 tokens
# Rule of thumb: 1 token ≈ 0.75 words
```

## Prompt Engineering Fundamentals

### Basic Prompt Structure

```
[Context] + [Task] + [Format] + [Examples] = Effective Prompt
```

#### 1. Zero-Shot Prompts

Direct instruction without examples:

```
Classify the sentiment of this text: "I love this product!"
```

#### 2. Few-Shot Prompts

Providing examples to guide the model:

```
Classify sentiment as positive, negative, or neutral:

Text: "This movie was amazing!"
Sentiment: positive

Text: "The service was terrible."
Sentiment: negative

Text: "It was okay, nothing special."
Sentiment: neutral

Text: "Best purchase I've ever made!"
Sentiment:
```

#### 3. Chain-of-Thought

Breaking down complex reasoning:

```
Solve this step by step:
What is 23 * 47?

Step 1: Break down the multiplication
Step 2: Calculate partial products
Step 3: Add the results
```

### Prompt Templates

**Classification Template:**
```
You are an expert classifier. 
Classify the following text into one of these categories: [CATEGORIES]

Text: [INPUT]
Category:
```

**Analysis Template:**
```
You are a helpful analyst. Please analyze the following [TOPIC].

[CONTEXT]

Provide your analysis in the following format:
1. Summary
2. Key insights
3. Recommendations
```

## Advanced Techniques

### 1. Role-Based Prompting

```python
# Instead of:
"Explain machine learning"

# Use:
"You are a senior data scientist explaining machine learning to a 
business executive. Focus on practical applications and ROI."
```

### 2. Instruction Following

```python
prompt = """
Follow these instructions exactly:
1. Read the provided text
2. Extract key facts (max 5)
3. Format as numbered list
4. End with confidence score (1-10)

Text: [YOUR_TEXT_HERE]
"""
```

### 3. Constrained Generation

```python
prompt = """
Write a product description with these constraints:
- Exactly 50 words
- Include "innovative" and "sustainable"
- Target audience: tech professionals
- Tone: professional but engaging
"""
```

### 4. Multi-Step Reasoning

```python
prompt = """
Solve this problem using the following steps:

Step 1: Identify the key information
Step 2: Determine what needs to be calculated
Step 3: Show your work
Step 4: Provide the final answer
Step 5: Verify your result

Problem: [YOUR_PROBLEM]
"""
```

## Hands-on Examples

### Example 1: Content Generation

```python
def generate_blog_post(topic, audience):
    prompt = f"""
    You are a professional content writer. Create a blog post about {topic} 
    for {audience}.
    
    Requirements:
    - 300-400 words
    - Include an engaging hook
    - Use 3-4 subheadings
    - End with a call-to-action
    
    Topic: {topic}
    Audience: {audience}
    """
    return llm.generate(prompt)

# Usage
post = generate_blog_post("AI in healthcare", "medical professionals")
```

### Example 2: Data Analysis

```python
def analyze_data(data_description, question):
    prompt = f"""
    You are a data analyst. Given this data: {data_description}
    
    Please answer: {question}
    
    Provide:
    1. Direct answer
    2. Methodology used
    3. Limitations/assumptions
    4. Confidence level
    """
    return llm.generate(prompt)
```

### Example 3: Code Generation

```python
def generate_code(language, task, requirements):
    prompt = f"""
    Generate {language} code for: {task}
    
    Requirements:
    {requirements}
    
    Please provide:
    1. Clean, commented code
    2. Example usage
    3. Error handling
    4. Performance considerations
    """
    return llm.generate(prompt)
```

## Best Practices

### ✅ Do's

1. **Be Specific**: Clear instructions yield better results
   ```
   # Good: "Write a 150-word summary of machine learning for beginners"
   # Bad: "Tell me about ML"
   ```

2. **Provide Context**: Set the scene for better understanding
   ```
   "You are helping a small business owner understand AI tools..."
   ```

3. **Use Examples**: Show the desired format
   ```
   Input: "Sunny, 75°F"
   Output: "Perfect weather for outdoor activities!"
   ```

4. **Break Down Complex Tasks**: Use step-by-step approaches

5. **Iterate and Refine**: Test and improve your prompts

### ❌ Don'ts

1. **Don't Be Vague**: Ambiguous prompts lead to inconsistent results
2. **Don't Assume Knowledge**: Provide necessary context
3. **Don't Ignore Token Limits**: Keep prompts within context windows
4. **Don't Skip Testing**: Always validate outputs
5. **Don't Forget Edge Cases**: Consider unusual inputs

### Prompt Optimization Framework

```python
def optimize_prompt(base_prompt, test_inputs, success_criteria):
    """
    1. Start with base prompt
    2. Test with sample inputs
    3. Identify failure modes
    4. Refine prompt
    5. Re-test
    6. Repeat until success criteria met
    """
    pass
```

### Common Pitfalls

1. **Prompt Injection**: User inputs that manipulate behavior
   ```python
   # Vulnerable
   prompt = f"Translate: {user_input}"
   
   # Safer
   prompt = f"Translate the following text to French: '{user_input}'"
   ```

2. **Hallucination**: Model generating false information
   ```python
   # Mitigation
   prompt = "Based only on the provided context, answer: [question]"
   ```

3. **Bias Amplification**: Reinforcing societal biases
   ```python
   # Better
   prompt = "Provide a balanced perspective considering diverse viewpoints..."
   ```

## Practical Exercises

### Exercise 1: Sentiment Analysis
Create a prompt that accurately classifies sentiment with explanations.

### Exercise 2: Content Summarization
Build a prompt that creates structured summaries of long articles.

### Exercise 3: Creative Writing
Design a prompt for generating consistent character dialogues.

### Exercise 4: Technical Documentation
Create a prompt that converts code into clear documentation.

## Tools for Prompt Engineering

| Tool | Purpose | Features |
|------|---------|----------|
| [PromptBase](https://promptbase.com/) | Prompt marketplace | Ready-made prompts |
| [Prompt Perfect](https://promptperfect.jina.ai/) | Prompt optimization | Auto-improvement |
| [LangSmith](https://smith.langchain.com/) | Prompt testing | Version control, analytics |
| [OpenAI Playground](https://platform.openai.com/playground) | Interactive testing | Model comparison |

## Next Steps

1. Practice with different prompt types
2. Study successful prompts in your domain
3. Build a prompt library for common tasks
4. Learn about [Agent Development](../05-agent-development/)
5. Explore [RAG Systems](../04-rag-systems/)

## Resources

- [OpenAI Prompt Engineering Guide](https://platform.openai.com/docs/guides/prompt-engineering)
- [Anthropic Prompt Engineering](https://docs.anthropic.com/claude/docs/prompt-engineering)
- [Prompt Engineering Guide](https://www.promptingguide.ai/)
- [LangChain Prompt Templates](https://python.langchain.com/docs/modules/model_io/prompts/)