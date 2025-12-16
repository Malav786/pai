# WildFace Recognition: Model Inversion Attacks and Defense Mechanisms

[![Open in Visual Studio Code](https://classroom.github.com/assets/open-in-vscode-2e0aaae1b6195c2367325f4f02e2d04e9abb55f0b24a779b69b11b9e10269abc.svg)](https://classroom.github.com/online_ide?assignment_repo_id=21953942&assignment_repo_type=AssignmentRepo)

## Purpose

This project investigates the security vulnerabilities of face recognition systems by demonstrating and analyzing **model inversion attacks**, a class of privacy attacks where adversaries can reconstruct private training data (facial images) through black-box queries to a deployed classifier. 

The project demonstrates how attackers can extract sensitive biometric information from machine learning models even without access to model parameters, using only prediction outputs. We implement **Natural Evolution Strategy (NES)** combined with an autoencoder architecture to perform model inversion attacks on a face recognition classifier trained on the Labeled Faces in the Wild (LFW) dataset.

To mitigate these threats, we design and evaluate multiple defense mechanisms:
- **Top-k probability filtering** to limit information leakage
- **Probability rounding** to reduce gradient resolution  
- **Calibrated noise injection** to introduce uncertainty
- **Behavioral anomaly detection** using Isolation Forest to monitor query patterns

Our comprehensive evaluation provides quantitative insights into the trade-offs between model utility, privacy protection, and computational overhead, offering actionable recommendations for deploying secure face recognition systems in production environments.

## Usage Instructions

### Prerequisites

- Python 3.12 or higher
- [Poetry](https://python-poetry.org/docs/#installation) for dependency management

### Clone the Project

```bash
git clone <repository-url>
cd pai
```

### Install the Project

This project uses Poetry for dependency management. Install dependencies as follows:

```bash

# Install project dependencies
poetry install

# This will install all dependencies including:
# - PyTorch for deep learning models
# - scikit-learn for machine learning utilities
# - Jupyter for notebook execution
# - Development tools (Black, isort)
```

### Code Formatting with Black

This project uses [Black](https://black.readthedocs.io/) for code formatting. Black is configured in `pyproject.toml` and supports both Python files and Jupyter notebooks.

**After updating dependencies, update the lock file:**
```bash
poetry lock
poetry install
```

**Format Python files:**
```bash
poetry run black finalproject/
```

**Format Jupyter notebooks:**
```bash
poetry run black notebooks/
```

**Format everything:**
```bash
poetry run black finalproject/ notebooks/
```

### Use the Project

#### Running the Main Project Notebook

The main project notebook (`notebooks/finalproject.ipynb`) contains the complete analysis:

```bash
# Start Jupyter notebook server
poetry run jupyter notebook
```

Then open `notebooks/finalproject.ipynb` in your browser.

#### Running Module Tests

Test individual modules using the testing notebook:

```bash
poetry run jupyter notebook
# Open notebooks/module_testing.ipynb
```

## Known Issues

1. **Notebook Execution**: Some cells in `finalproject.ipynb` may take significant time to execute (especially model training and attack optimization). Ensure you have sufficient computational resources.

2. **Dataset Download**: The LFW dataset is automatically downloaded on first use via scikit-learn. This requires an internet connection and may take several minutes depending on connection speed.

3. **Memory Requirements**: Training the autoencoder and running NES attacks can be memory-intensive. For systems with limited RAM, consider reducing batch sizes or attack population sizes.

4. **Device Compatibility**: The code defaults to CPU. For GPU acceleration, ensure CUDA-compatible PyTorch is installed and modify device settings in the notebooks.

5. **Reproducibility**: While random seeds are set, results may vary slightly across different Python/PyTorch versions or hardware configurations.

6. **Jupyter Notebook Formatting**: Black formatting of notebooks may occasionally require manual cell execution order adjustments if cells have dependencies.

## Feature Roadmap

The following features are planned for future development:

1. **Advanced Attack Techniques**
   - Adaptive population sizing for NES optimization
   - Hybrid gradient-free methods combining multiple optimization strategies
   - White-box attack variants for comparison

2. **Enhanced Defense Mechanisms**
   - Adaptive defense parameters that adjust based on threat levels
   - Differential privacy integration with formal privacy budgets
   - Federated learning compatibility for distributed settings

3. **Evaluation Improvements**
   - Additional metrics for reconstruction quality assessment
   - Real-world deployment evaluation with diverse user populations
   - Standardized evaluation benchmarks for model inversion attacks

4. **Production Features**
   - REST API for model serving with built-in defense mechanisms
   - Real-time attack detection dashboard
   - Automated defense selection based on threat intelligence

5. **Documentation and Tooling**
   - Interactive visualization tools for attack/defense analysis
   - Command-line interface for running attacks and defenses
   - Docker containerization for easy deployment

6. **Multi-modal Extensions**
   - Extension to color images and higher resolutions
   - Support for other biometric modalities (voice, gait, etc.)
   - Temporal pattern exploitation in query sequences

## Contributing

Contributions to this project are welcome! Please follow these guidelines:

1. **Fork the repository** and create a feature branch
2. **Follow code style**: All code must be formatted with Black before submission
   ```bash
   poetry run black finalproject/ notebooks/
   ```
3. **Write tests**: Add tests for new features in the `tests/` directory
4. **Update documentation**: Update README, docstrings, and relevant documentation
5. **Submit a pull request** with a clear description of changes

### Development Setup

```bash
# Install development dependencies
poetry install

# Run code formatting
poetry run black finalproject/ notebooks/

# Sort imports (optional)
poetry run isort finalproject/
```

### Code Style

- **Formatting**: Black (line length: 88, Python 3.12)
- **Type Hints**: Use type hints for function signatures
- **Docstrings**: Follow Google-style docstrings for all public functions and classes
- **Naming**: Follow PEP 8 naming conventions

## License

This project is distributed under the **MIT License**. See the [LICENSE](LICENSE) file for details.

**Copyright (c) 2024 RUC Practical AI Class**

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the conditions in the LICENSE file.

### Third-Party Components

- **LFW Dataset**: Non-commercial research and educational use only (University of Massachusetts Amherst)
- **PyTorch**: BSD-style license
- **scikit-learn**: BSD 3-Clause License
- All other dependencies: See individual package licenses

## Contact

For questions, issues, or contributions, please contact:

**Project Maintainer**: RUC Practical AI Class  
**Email**: sarthakchandervanshi@gmail.com, malav.10802@gmail.com

For technical questions about the implementation, please open an issue on the repository.

---

## Additional Resources

- **Model Card**: See [MODEL_CARD.md](MODEL_CARD.md) for detailed model information
- **Data Card**: See [DATA_CARD.md](DATA_CARD.md) for dataset documentation
- **Project Instructions**: See [INSTRUCTIONS.md](INSTRUCTIONS.md) for assignment requirements
