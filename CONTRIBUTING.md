# 🤝 Contributing to SciTrans-LLMs

Thank you for your interest in contributing to our research project! This guide will help you understand how to contribute effectively.

---

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Code Style](#code-style)
- [Testing](#testing)
- [Documentation](#documentation)
- [Submitting Changes](#submitting-changes)
- [Areas for Contribution](#areas-for-contribution)

---

## 📜 Code of Conduct

We are committed to providing a welcoming and inclusive environment for all contributors. Please:

- ✅ Be respectful and constructive
- ✅ Welcome newcomers and help them get started
- ✅ Focus on what is best for the community
- ✅ Show empathy towards other community members
- ❌ Use inappropriate language or personal attacks
- ❌ Publish others' private information
- ❌ Engage in trolling or insulting behavior

---

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- Git
- Basic understanding of machine translation and LLMs
- Familiarity with Python development

### First Steps

1. **Fork the repository** on GitHub
2. **Clone your fork**:
   ```bash
   git clone https://github.com/YOUR_USERNAME/SciTrans-LLMs.git
   cd SciTrans-LLMs
   ```
3. **Set up upstream remote**:
   ```bash
   git remote add upstream https://github.com/uglydavy/SciTrans-LLMs.git
   ```

---

## 🛠️ Development Setup

### 1. Create Development Environment

```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install development dependencies
pip install -e ".[dev,full]"
```

### 2. Install Pre-commit Hooks (Optional)

```bash
pip install pre-commit
pre-commit install
```

### 3. Verify Setup

```bash
# Run tests
pytest tests/ -v

# Check code style
ruff check scitrans_llms/

# Type checking
mypy scitrans_llms/
```

---

## 📝 Code Style

We follow Python best practices and use automated tools to maintain code quality.

### General Guidelines

- **PEP 8**: Follow Python style guide
- **Line length**: Maximum 100 characters
- **Naming**: Use descriptive variable/function names
- **Comments**: Explain *why*, not *what*
- **Docstrings**: Required for all public functions and classes

### Example Function

```python
def translate_with_context(
    text: str,
    context: list[str],
    glossary: Glossary | None = None,
) -> TranslationResult:
    """
    Translate text using document-level context.
    
    This function incorporates previous translations as context to maintain
    coherence across the document. The glossary is enforced both in the
    LLM prompt and post-processing.
    
    Args:
        text: Source text to translate
        context: List of previous translations for context
        glossary: Optional terminology glossary
        
    Returns:
        TranslationResult with translated text and metadata
        
    Example:
        >>> result = translate_with_context(
        ...     "The model uses attention.",
        ...     context=["Deep learning is powerful."],
        ...     glossary=my_glossary,
        ... )
        >>> print(result.translated_text)
    """
    # Implementation here
    pass
```

### Docstring Format

We use Google-style docstrings:

```python
"""
Brief description of the function.

More detailed explanation if needed. Can span multiple lines
and include examples, implementation notes, etc.

Args:
    param1: Description of param1
    param2: Description of param2

Returns:
    Description of return value

Raises:
    ValueError: When and why this is raised
    
Example:
    >>> result = my_function(arg1, arg2)
    >>> print(result)
"""
```

### Type Hints

Use type hints for all function signatures:

```python
from typing import Optional, List, Dict, Tuple

def process_blocks(
    blocks: List[Block],
    config: PipelineConfig,
) -> Tuple[List[str], Dict[str, float]]:
    """Process blocks and return translations with stats."""
    pass
```

---

## 🧪 Testing

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_pipeline.py -v

# Run with coverage
pytest tests/ --cov=scitrans_llms --cov-report=html

# View coverage report
open htmlcov/index.html
```

### Writing Tests

Create tests in the `tests/` directory:

```python
# tests/test_my_feature.py
import pytest
from scitrans_llms.my_module import my_function

def test_my_function_basic():
    """Test basic functionality of my_function."""
    result = my_function("input")
    assert result == "expected_output"

def test_my_function_edge_case():
    """Test edge case handling."""
    with pytest.raises(ValueError):
        my_function(None)

@pytest.mark.parametrize("input,expected", [
    ("hello", "bonjour"),
    ("world", "monde"),
])
def test_my_function_examples(input, expected):
    """Test multiple examples."""
    result = my_function(input)
    assert result == expected
```

---

## 📚 Documentation

### Module Documentation

Every module should have a docstring at the top:

```python
"""
Module name and brief description.

This module provides functionality for X, Y, and Z.
It is used primarily for A and B purposes.

Classes:
    ClassName: Description
    AnotherClass: Description

Functions:
    function_name: Description
    another_function: Description

Example:
    >>> from scitrans_llms.my_module import MyClass
    >>> obj = MyClass()
    >>> obj.do_something()
"""
```

### README Updates

If you add new features, update `README.md`:

- Add to Table of Contents
- Add usage examples
- Update module documentation section
- Add to command reference if CLI command

---

## 🔄 Submitting Changes

### 1. Create a Branch

```bash
# Update your fork
git fetch upstream
git checkout main
git merge upstream/main

# Create feature branch
git checkout -b feature/my-new-feature
```

### 2. Make Changes

- Write code following our style guide
- Add tests for new functionality
- Update documentation
- Run tests locally

### 3. Commit Changes

Use clear, descriptive commit messages:

```bash
# Good commit messages
git commit -m "Add support for German language translation"
git commit -m "Fix glossary enforcement in multi-turn context"
git commit -m "Update README with new API examples"

# Bad commit messages (avoid these)
git commit -m "fix bug"
git commit -m "update"
git commit -m "changes"
```

### 4. Push and Create Pull Request

```bash
# Push to your fork
git push origin feature/my-new-feature
```

Then go to GitHub and create a Pull Request with:

- **Clear title**: Summarize the change
- **Description**: Explain what and why
- **Tests**: Mention what tests you added
- **Related issues**: Reference any related issues

### Pull Request Template

```markdown
## Description
Brief description of changes

## Motivation
Why are these changes needed?

## Changes Made
- Added X feature
- Fixed Y bug
- Updated Z documentation

## Testing
- Added test_new_feature.py
- All existing tests pass
- Manual testing performed

## Checklist
- [ ] Tests pass locally
- [ ] Code follows style guide
- [ ] Documentation updated
- [ ] Type hints added
```

---

## 🎯 Areas for Contribution

### 🐛 Bug Fixes

Found a bug? We'd love your help fixing it!

1. Check existing issues to avoid duplicates
2. Create an issue describing the bug
3. Fork, fix, test, and submit PR

### ✨ New Features

Ideas for new features:

- **Additional Language Pairs**: Support for more languages
- **New LLM Backends**: Integration with other LLM services
- **Enhanced Glossaries**: Domain-specific terminology databases
- **Better Evaluation Metrics**: New quality metrics
- **PDF Layout Improvements**: Better table/figure handling
- **GUI Enhancements**: More features in web interface

### 📖 Documentation

Documentation is always appreciated:

- Fix typos or clarify confusing sections
- Add more examples
- Translate documentation to other languages
- Create video tutorials
- Write blog posts about usage

### 🧪 Tests

Help improve test coverage:

- Add unit tests for uncovered code
- Create integration tests
- Add edge case tests
- Performance benchmarks

### 🌍 Translations

Help translate:

- User interface strings
- Documentation
- Error messages
- Example glossaries for other domains

---

## 🔬 Research Contributions

As a research project, we especially welcome:

### 📊 Experimental Results

- Run experiments with new language pairs
- Test with different domains (medical, legal, etc.)
- Compare with other systems
- Share interesting findings

### 📝 Publications

If you use this system in research:

- Cite our work
- Share your papers with us
- Contribute back improvements

### 💡 Ideas

Open an issue to discuss:

- Novel translation approaches
- New evaluation methods
- Architecture improvements
- Research questions

---

## 🐞 Reporting Bugs

### Before Reporting

1. Check existing issues
2. Try latest version
3. Verify it's reproducible

### Bug Report Template

```markdown
## Bug Description
Clear description of the bug

## Steps to Reproduce
1. Step one
2. Step two
3. See error

## Expected Behavior
What should happen

## Actual Behavior
What actually happens

## Environment
- OS: macOS 13.0
- Python: 3.9.10
- Version: 0.1.0

## Additional Context
Any other relevant information
```

---

## 💬 Communication

- **GitHub Issues**: For bugs, features, questions
- **Email**: aknk.v@pm.me for private inquiries
- **Pull Requests**: For code contributions

---

## 🙏 Recognition

Contributors will be:

- Listed in CONTRIBUTORS.md
- Mentioned in release notes
- Acknowledged in thesis (if significant contribution)
- Co-authors on papers (for research contributions)

---

## 📄 License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

## 🎓 For Students

This is a master's thesis project, making it a great learning opportunity!

### What You'll Learn

- Machine translation with LLMs
- PDF processing and layout analysis
- Research methodology
- Software engineering best practices
- Open source collaboration

### Getting Started as a Student

1. Read the thesis guide: `THESIS_GUIDE.md`
2. Run the experiments: `EXPERIMENTS.md`
3. Start with small contributions
4. Ask questions - we're here to help!

---

Thank you for contributing to SciTrans-LLMs! 🚀

**Together we're building better tools for scientific translation.**

