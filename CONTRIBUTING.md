# Contributing to Residual Masking Network (RMN)

Thank you for your interest in contributing to **Residual Masking Network**! We welcome contributions, bug fixes, feature proposals, and documentation improvements.

## Code of Conduct
Please ensure all discussions and interactions remain friendly, respectful, and constructive.

## How to Contribute

1. **Fork the Repository**: Click "Fork" on GitHub to create your own copy of the repository.
2. **Clone your Fork**:
   ```bash
   git clone https://github.com/<your-username>/ResidualMaskingNetwork-master.git
   cd ResidualMaskingNetwork-master
   ```
3. **Create a Feature Branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```
4. **Set Up Development Environment**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Or on Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   pip install flake8 pytest black isort
   ```
5. **Make Your Changes**: Write clean, readable code with descriptive comments where appropriate.
6. **Code Style**: Format your code according to standard PEP 8 conventions:
   ```bash
   black .
   isort .
   flake8 .
   ```
7. **Commit & Push**:
   ```bash
   git add .
   git commit -m "feat: describe your change concisely"
   git push origin feature/your-feature-name
   ```
8. **Submit a Pull Request**: Open a pull request against the `main` branch with a clear description of your changes.

## Reporting Issues
If you encounter any bugs, please open a GitHub Issue providing:
- Python and PyTorch version
- Operating System details
- Complete traceback / error message
- Minimal steps to reproduce
