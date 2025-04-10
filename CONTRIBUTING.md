# Contributing to PyFinQuant

Thank you for your interest in contributing to PyFinQuant! This document provides guidelines and instructions for contributing to the project.

## Code of Conduct

By participating in this project, you agree to abide by the [Code of Conduct](CODE_OF_CONDUCT.md).

## How to Contribute

1. Fork the repository
2. Create a new branch for your feature/fix
3. Make your changes
4. Run tests and linting
5. Submit a pull request

## Development Setup

1. Clone your fork:
```bash
git clone https://github.com/yourusername/pyfinquant.git
cd pyfinquant
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install development dependencies:
```bash
pip install -e .[dev]
```

## Code Style

We use several tools to maintain code quality:

- Black for code formatting
- isort for import sorting
- flake8 for linting
- mypy for type checking

Run all checks before submitting a PR:
```bash
black src/ tests/
isort src/ tests/
flake8 src/ tests/
mypy src/
```

## Testing

We use pytest for testing. Run tests with:
```bash
pytest tests/ -v
```

## Documentation

We use Sphinx for documentation. Build docs with:
```bash
cd docs
make html
```

## Pull Request Process

1. Update the README.md with details of changes if needed
2. Update the CHANGELOG.md with details of changes
3. The PR must pass all CI checks
4. The PR must be reviewed by at least one maintainer

## Questions?

Feel free to open an issue if you have any questions about contributing! 