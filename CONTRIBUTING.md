# Contributing to PyFinQuant

Thank you for your interest in contributing to PyFinQuant! This document provides guidelines and instructions for contributing to the project.

## Table of Contents
- [Code of Conduct](#code-of-conduct)
- [How to Contribute](#how-to-contribute)
- [Development Setup](#development-setup)
- [Coding Standards](#coding-standards)
- [Testing](#testing)
- [Documentation](#documentation)
- [Pull Request Process](#pull-request-process)
- [Release Process](#release-process)

## Code of Conduct

By participating in this project, you agree to abide by our Code of Conduct. Please be respectful and considerate of others.

## How to Contribute

1. **Fork** the repository
2. **Clone** your fork:
   ```bash
   git clone https://github.com/your-username/PyFinQuant.git
   cd PyFinQuant
   ```
3. **Create a new branch** for your feature:
   ```bash
   git checkout -b feature/your-feature-name
   ```
4. **Make your changes**
5. **Test your changes**
6. **Commit your changes** with a clear commit message
7. **Push** to your fork
8. **Create a Pull Request**

## Development Setup

### Prerequisites
- Python 3.8 or higher
- Git
- Poetry or UV (package manager)

### Installation

#### Using Poetry
```bash
# Install Poetry if you haven't already
curl -sSL https://install.python-poetry.org | python3 -

# Install dependencies
poetry install
poetry shell
```

#### Using UV
```bash
# Install UV if you haven't already
pip install uv

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate  # Windows

# Install dependencies
uv pip install -e ".[dev]"
```

## Coding Standards

### Code Style
- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) guidelines
- Use type hints for all function parameters and return values
- Write docstrings following [Google style](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings)

### Tools
We use several tools to maintain code quality:
```bash
# Format code
black src/ tests/
isort src/ tests/

# Check code quality
flake8 src/
mypy src/
```

### Commit Messages
- Use present tense ("Add feature" not "Added feature")
- Use imperative mood ("Move cursor to..." not "Moves cursor to...")
- Limit the first line to 72 characters or less
- Reference issues and pull requests after the first line

## Testing

### Running Tests
```bash
# Run all tests
pytest tests/

# Run tests with coverage
pytest --cov=src/pyfinquant tests/

# Run specific test file
pytest tests/test_file.py
```

### Writing Tests
- Write tests for all new features
- Follow the Arrange-Act-Assert pattern
- Use descriptive test names
- Include edge cases and error conditions

## Documentation

### Building Documentation
```bash
cd docs
make html
```

### Documentation Standards
- Keep docstrings up to date
- Include examples in docstrings
- Document all public APIs
- Update README.md for significant changes

## Pull Request Process

1. **Update the documentation** if necessary
2. **Update the tests** if necessary
3. **Ensure all tests pass**
4. **Update the CHANGELOG.md** with your changes
5. **Submit your PR** with a clear description

### PR Checklist
- [ ] Tests added/updated
- [ ] Documentation updated
- [ ] Code follows style guidelines
- [ ] CHANGELOG.md updated
- [ ] All tests pass

## Release Process

1. **Update version** in pyproject.toml
2. **Update CHANGELOG.md**
3. **Create a release tag**
4. **Build and publish** the package

### Versioning
We follow [Semantic Versioning](https://semver.org/):
- MAJOR version for incompatible API changes
- MINOR version for backward-compatible functionality
- PATCH version for backward-compatible bug fixes

## Questions?

If you have any questions about contributing, please:
1. Check the [documentation](https://pyfinquant.readthedocs.io/)
2. Open an [issue](https://github.com/puneethgb098/PyFinQuant/issues)
3. Contact the maintainers

Thank you for contributing to PyFinQuant! 