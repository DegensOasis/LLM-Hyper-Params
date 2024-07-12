# Contributing to LLM-Hyper-Params

First off, thank you for considering contributing to LLM-Hyper-Params! It's people like you that will make this tool a great resource for optimizing Language Model parameters. This document provides guidelines for contributing to the project.

## Code of Conduct

By participating in this project, you are humbly requested to uphold our Code of Conduct. 

## How Can I Contribute?

### Reporting Bugs

This section guides you through submitting a bug report for LLM-Hyper-Params. Following these guidelines helps maintainers and the community understand your report, reproduce the behavior, and find related reports.

- Use a clear and descriptive title for the issue to identify the problem.
- Describe the exact steps which reproduce the problem in as many details as possible.
- Provide specific examples to demonstrate the steps, including configuration files or command-line arguments used.
- Describe the behavior you observed after following the steps and point out what exactly is the problem with that behavior.
- Explain which behavior you expected to see instead and why.
- Include details about your configuration and environment.

### Suggesting Enhancements

This section guides you through submitting an enhancement suggestion for LLM-Hyper-Params, including completely new features and minor improvements to existing functionality.

- Use a clear and descriptive title for the issue to identify the suggestion.
- Provide a step-by-step description of the suggested enhancement in as many details as possible.
- Provide specific examples to demonstrate the steps or point out the part of LLM-Hyper-Params where the enhancement could be implemented.
- Explain why this enhancement would be useful to most LLM-Hyper-Params users.

### Pull Requests

- Fill in the required template
- Do not include issue numbers in the PR title
- Follow the Python style guide
- Include appropriate test cases
- Document new code based on the Documentation Styleguide
- End all files with a newline

## Styleguides

### Git Commit Messages

- Use the present tense ("Add feature" not "Added feature")
- Use the imperative mood ("Move cursor to..." not "Moves cursor to...")
- Limit the first line to 72 characters or less
- Reference issues and pull requests liberally after the first line
- Consider starting the commit message with an applicable emoji:
    * 🎨 `:art:` when improving the format/structure of the code
    * 🐎 `:racehorse:` when improving performance
    * 📝 `:memo:` when writing docs
    * 🐛 `:bug:` when fixing a bug
    * 🔥 `:fire:` when removing code or files

### Python Styleguide

All Python code must adhere to the [PEP 8 style guide](https://www.python.org/dev/peps/pep-0008/).

Additionally:
- Use type hints for function arguments and return values
- Use docstrings for all public modules, functions, classes, and methods

### Documentation Styleguide

- Use [Markdown](https://daringfireball.net/projects/markdown) for documentation.
- Reference functions and classes appropriately.
- Use code blocks for command-line examples or Python code snippets.

## Additional Notes

### Issue and Pull Request Labels

This section lists the labels we use to help us track and manage issues and pull requests in LLM-Hyper-Params.

- `bug` - Issues for bugs in the code
- `enhancement` - Issues for new features or improvements
- `documentation` - Issues related to documentation
- `good first issue` - Good for newcomers
- `optimization` - Issues related to performance optimization
- `LLM-specific` - Issues related to specific Language Models

## Getting Started

To set up LLM-Hyper-Params for local development:

1. Fork the LLM-Hyper-Params repo on GitHub.
2. Clone your fork locally:
    ```
    git clone git@github.com:your_name_here/LLM-Hyper-Params.git
    ```
3. Create a branch for local development:
    ```
    git checkout -b name-of-your-bugfix-or-feature
    ```
4. Make your changes locally.
5. Run the tests to ensure your changes haven't broken any existing functionality:
    ```
    python -m unittest discover tests
    ```
6. Commit your changes and push your branch to GitHub:
    ```
    git add .
    git commit -m "Your detailed description of your changes."
    git push origin name-of-your-bugfix-or-feature
    ```
7. Submit a pull request through the GitHub website.

Thank you again for your interest in improving LLM-Hyper-Params! We look forward to your contributions.
