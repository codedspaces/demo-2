# Contributing to AI Engineering Toolkit

Thank you for your interest in contributing to the AI Engineering Toolkit! This repository thrives on community contributions that help AI engineers build better LLM applications.

## 🎯 Our Mission

We're building a curated, practical resource for AI engineers that focuses on:
- **Real-world utility** over theoretical concepts
- **Production-ready** tools and examples
- **Quality over quantity** in our recommendations
- **Community-driven** content and curation

## 📋 How to Contribute

### 1. Types of Contributions We Welcome

- **🛠️ Tool Recommendations**: New tools, frameworks, or libraries
- **📦 Templates & Examples**: Working code examples and templates
- **📚 Tutorials & Guides**: Step-by-step implementations
- **🐛 Bug Fixes**: Corrections to existing content
- **📖 Documentation**: Improvements to READMEs and guides
- **🔍 Reviews**: Testing and feedback on existing tools

### 2. Before You Start

1. **Search existing issues** to see if your contribution is already being discussed
2. **Check the roadmap** to understand our current priorities
3. **Read our quality standards** below
4. **Open an issue** to discuss major changes before implementing

## 🌟 Quality Standards

### Tool Recommendations

Before suggesting a new tool, ensure it meets these criteria:

- [ ] **Actively maintained** (commits within last 6 months)
- [ ] **Production-ready** (stable version available)
- [ ] **Well-documented** (clear installation and usage instructions)
- [ ] **Open source or free tier** available
- [ ] **Adds unique value** (not duplicating existing recommendations)

### Code Examples & Templates

- [ ] **Runnable code** that works out of the box
- [ ] **Clear documentation** with setup instructions
- [ ] **Error handling** for common issues
- [ ] **Dependencies listed** with versions
- [ ] **License compatibility** (MIT preferred)

### Tutorials & Guides

- [ ] **Clear learning objectives**
- [ ] **Step-by-step instructions**
- [ ] **Working examples** with expected outputs
- [ ] **Troubleshooting section**
- [ ] **Next steps** for further learning

## 📝 Submission Process

### Step 1: Fork & Clone

```bash
# Fork the repository on GitHub
git clone https://github.com/yourusername/ai-engineering-toolkit.git
cd ai-engineering-toolkit
```

### Step 2: Create a Branch

```bash
git checkout -b feature/your-contribution-name
```

### Step 3: Make Your Changes

Follow the appropriate template for your contribution type:

#### Adding a Tool

1. Add to the relevant section in `README.md`
2. Use this format:
   ```markdown
   | [Tool Name](https://github.com/tool/repo) | Brief description | Language | License | ![GitHub stars](https://img.shields.io/github/stars/tool/repo.svg) |
   ```

#### Adding a Template

1. Create a new directory under `templates/`
2. Include:
   - `README.md` with setup and usage instructions
   - Working code files
   - `requirements.txt` or equivalent
   - `.env.example` if environment variables are needed

#### Adding a Tutorial

1. Create a new directory under `tutorials/`
2. Include:
   - `README.md` with the tutorial content
   - Any code files or notebooks
   - Sample data if needed

### Step 4: Test Your Contribution

- [ ] **Code runs successfully** on a fresh environment
- [ ] **Links work** and point to correct resources
- [ ] **Documentation is clear** and free of typos
- [ ] **Examples produce expected outputs**

### Step 5: Submit Pull Request

1. **Commit your changes**:
   ```bash
   git add .
   git commit -m "Add [tool/template/tutorial]: [brief description]"
   git push origin feature/your-contribution-name
   ```

2. **Create a pull request** with:
   - Clear title describing the change
   - Description explaining the value added
   - Screenshots/examples if relevant
   - Link to any related issues

## 🔄 Review Process

### What Happens Next

1. **Automated checks** run (linting, link validation)
2. **Maintainer review** within 5-7 days
3. **Community feedback** period (3-5 days)
4. **Final review** and merge

### Review Criteria

We evaluate contributions based on:

- **Relevance**: Does it fit our mission and audience?
- **Quality**: Is it well-implemented and documented?
- **Uniqueness**: Does it add new value?
- **Maintainability**: Can we keep it up-to-date?

## 📋 Contribution Templates

### Tool Suggestion Template

```markdown
## Tool Information
- **Name**: [Tool Name]
- **Repository**: [GitHub/GitLab URL]
- **Category**: [Vector DB/RAG/Agent Framework/etc.]
- **Language**: [Python/JavaScript/etc.]
- **License**: [MIT/Apache/etc.]

## Why This Tool?
- [Unique features]
- [Production readiness]
- [Community adoption]

## Suggested Placement
- Section: [Which README section]
- Description: [Proposed description]
```

### Template Contribution Template

```markdown
## Template Information
- **Name**: [Template Name]
- **Category**: [RAG/Agent/Chatbot/etc.]
- **Complexity**: [Beginner/Intermediate/Advanced]
- **Technologies**: [LangChain, Streamlit, etc.]

## What It Demonstrates
- [Key concepts]
- [Use cases]
- [Learning objectives]

## Files Included
- [ ] README.md
- [ ] requirements.txt
- [ ] Main application files
- [ ] Example data/configs
- [ ] Tests (if applicable)
```

## 🚀 Getting Help

### Community Support

- **GitHub Discussions**: Ask questions and share ideas
- **Issues**: Report bugs or request features
- **Discord** (coming soon): Real-time community chat

### Maintainer Contact

For major contributions or questions:
- Open an issue with the `question` label
- Tag maintainers in your PR for urgent matters

## 🏆 Recognition

We value our contributors! Here's how we recognize contributions:

- **Contributors file**: All contributors listed in CONTRIBUTORS.md
- **Release notes**: Major contributions highlighted in releases
- **Newsletter mentions**: Significant contributions featured in AI Engineering newsletter
- **Maintainer invitations**: Regular contributors invited to join the maintainer team

## 📜 Code of Conduct

### Our Standards

We're committed to providing a welcoming and inclusive environment:

- **Be respectful** in all interactions
- **Be constructive** in feedback and discussions
- **Be collaborative** and help others learn
- **Be inclusive** and consider diverse perspectives

### Unacceptable Behavior

- Harassment, discrimination, or personal attacks
- Spam, promotional content, or off-topic discussions
- Sharing false or misleading information
- Violating intellectual property rights

### Enforcement

Violations will be addressed by:
1. Warning and guidance
2. Temporary restrictions
3. Permanent ban (for severe cases)

Report issues to maintainers via private message or email.

## 📊 Contribution Guidelines Summary

| Contribution Type | Review Time | Requirements |
|-------------------|-------------|--------------|
| Tool addition | 3-5 days | Meets quality criteria |
| Template/Example | 5-7 days | Working code + docs |
| Tutorial | 7-10 days | Complete guide + examples |
| Bug fix | 1-3 days | Clear problem + solution |
| Documentation | 1-3 days | Accurate + helpful |

## 🎉 Thank You!

Every contribution, no matter how small, helps make the AI Engineering Toolkit better for everyone. We appreciate your time and effort in helping the community build better LLM applications!

---

**Questions?** Open an issue or start a discussion. We're here to help! 🚀