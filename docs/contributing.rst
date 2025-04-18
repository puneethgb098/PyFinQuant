Contributing
============

We welcome contributions to PyFinQuant! Here's how you can help:

1. Fork the repository
2. Create a new branch for your feature or bugfix
3. Make your changes
4. Submit a pull request

Development Setup
----------------

1. Clone the repository:

.. code-block:: bash

    git clone https://github.com/puneethgb098/PyFinQuant.git
    cd PyFinQuant

2. Create a virtual environment:

.. code-block:: bash

    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate

3. Install development dependencies:

.. code-block:: bash

    pip install -e ".[dev]"

4. Install pre-commit hooks:

.. code-block:: bash

    pre-commit install

Coding Standards
---------------

- Follow PEP 8 style guide
- Use type hints
- Write docstrings for all public functions and classes
- Add tests for new features
- Update documentation as needed

Testing
-------

Run the test suite:

.. code-block:: bash

    pytest

Documentation
------------

Build the documentation:

.. code-block:: bash

    cd docs
    make html

Pull Request Process
------------------

1. Update the documentation
2. Add tests for new features
3. Ensure all tests pass
4. Update the changelog
5. Submit your pull request 