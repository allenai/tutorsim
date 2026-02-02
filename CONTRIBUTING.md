# Contributing to tutor-bench

Thank you for your interest in contributing to tutor-bench! We welcome contributions from the community and are grateful for any help you can provide.

## Code of Conduct

By participating in this project, you agree to abide by our code of conduct: be respectful, inclusive, and constructive in all interactions.

## How to Contribute

### Reporting Issues

If you find a bug or have a feature request:

1. Check the [issue tracker](https://github.com/allenai/tutor-bench/issues) to see if it has already been reported
2. If not, create a new issue with:
   - A clear, descriptive title
   - Steps to reproduce (for bugs)
   - Expected vs actual behavior
   - System information (Python version, OS, etc.)

### Contributing Code

#### Setup Development Environment

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/tutor-bench.git
   cd tutor-bench
   ```

3. Create a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

4. Install development dependencies:
   ```bash
   make install-dev
   ```

5. Create a feature branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```

#### Development Workflow

1. Make your changes
2. Add or update tests as needed
3. Run the test suite:
   ```bash
   make test
   ```

4. Format and lint your code:
   ```bash
   make format  # Auto-format code
   make lint    # Check for linting issues
   ```

5. Run all checks:
   ```bash
   make run-checks
   ```

6. Update documentation if needed:
   ```bash
   make docs
   make serve-docs  # Preview at http://localhost:8000
   ```

7. Update CHANGELOG.md with your changes

#### Commit Guidelines

We follow conventional commits for clear history:

- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation changes
- `style:` Code style changes (formatting, etc.)
- `refactor:` Code refactoring
- `test:` Test additions or changes
- `chore:` Maintenance tasks
- `perf:` Performance improvements

Example:
```bash
git commit -m "feat: add new evaluation metric for student engagement"
```

#### Pull Request Process

1. Push your changes to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```

2. Create a pull request on GitHub

3. Ensure your PR:
   - Has a clear title and description
   - References any related issues
   - Passes all CI checks
   - Has updated CHANGELOG.md
   - Includes tests for new functionality
   - Has updated documentation if needed

4. Wait for review and address any feedback

### Testing Guidelines

- Write tests for all new functionality
- Ensure tests are deterministic and don't depend on external services
- Use fixtures for test data
- Mark slow tests with `@pytest.mark.slow`
- Mark GPU-requiring tests with `@pytest.mark.gpu`

Example test:
```python
def test_new_feature():
    """Test description."""
    # Arrange
    data = prepare_test_data()

    # Act
    result = new_feature(data)

    # Assert
    assert result.success
    assert result.value == expected_value
```

### Documentation Guidelines

- Use Google-style docstrings:
  ```python
  def function(arg1: str, arg2: int) -> bool:
      """Brief description.

      Longer description if needed.

      Args:
          arg1: Description of arg1
          arg2: Description of arg2

      Returns:
          Description of return value

      Raises:
          ValueError: When validation fails
      """
  ```

- Update relevant documentation when changing functionality
- Include usage examples in docstrings
- Keep README.md up to date

### Code Style Guidelines

- Follow PEP 8 with 120-character line limit
- Use type hints for all function signatures
- Prefer descriptive variable names
- Keep functions focused and small
- Add comments for complex logic
- Use f-strings for string formatting

### Performance Considerations

- Profile before optimizing
- Consider async/await for I/O operations
- Use appropriate data structures
- Document performance characteristics in docstrings
- Add benchmarks for performance-critical code

## Development Commands Reference

```bash
# Install for development
make install-dev

# Run tests
make test           # All tests
make test-fast      # Quick tests only
make test-cov       # With coverage report

# Code quality
make format         # Auto-format code
make lint          # Run linter
make typecheck     # Type checking
make run-checks    # All checks

# Documentation
make docs          # Build docs
make serve-docs    # Serve docs locally

# Cleanup
make clean         # Remove build artifacts
```

## Release Process (Maintainers)

1. Update version in `tutor_bench/version.py`
2. Update CHANGELOG.md
3. Create a git tag:
   ```bash
   git tag -a v0.2.0 -m "Release version 0.2.0"
   git push origin v0.2.0
   ```
4. GitHub Actions will automatically build and publish to PyPI

## Getting Help

- Check the [documentation](https://tutor-bench.readthedocs.io)
- Ask questions in [GitHub Discussions](https://github.com/allenai/tutor-bench/discussions)
- Contact the maintainers: kylel@allenai.org

## Recognition

Contributors will be recognized in:
- The project's AUTHORS file
- Release notes
- Project documentation

Thank you for contributing to tutor-bench!
