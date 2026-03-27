# Contributing

Contributions to Aragog are welcome. Here is how to get started.

## Setting up for development

1. Clone the repository:
   ```console
   git clone git@github.com:FormingWorlds/aragog.git
   cd aragog
   ```

2. Install in editable mode with development dependencies:
   ```console
   pip install -e ".[docs]"
   pip install pytest pytest-cov pytest-dependency
   ```

   Or with Poetry:
   ```console
   poetry install --with test --with docs
   ```

3. Verify your setup:
   ```console
   pytest tests/
   ```

## Development workflow

1. Create a feature branch from `main`:
   ```console
   git checkout -b your-name/short-description
   ```

2. Make your changes. Follow the existing code style:
   - Docstrings on all public functions and classes (NumPy style)
   - Type hints where practical
   - Inline comments for non-obvious logic

3. Lint and format before committing:
   ```console
   ruff check --fix src/ tests/
   ruff format src/ tests/
   ```

4. Run the test suite:
   ```console
   pytest tests/
   ```

5. Open a pull request against `main`.

## Reporting issues

If you find a bug or have a feature request, please open an [issue on GitHub](https://github.com/FormingWorlds/aragog/issues).

## Code of Conduct

All contributors are expected to follow our [Code of Conduct](CODE_OF_CONDUCT.md).
