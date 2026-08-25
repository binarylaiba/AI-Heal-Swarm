# Contributing to AutoHeal Swarm

Thank you for your interest in contributing to **AutoHeal Swarm**! We welcome contributions, bug fixes, feature requests, and ideas.

---

## 🛠️ Development Workflow

1. **Fork the Repository** on GitHub.
2. **Clone your fork**:
   ```bash
   git clone https://github.com/<your-username>/AI-Heal-Swarm.git
   cd AI-Heal-Swarm
   ```
3. **Create a virtual environment**:
   ```bash
   python -m venv .venv
   # Windows:
   .\.venv\Scripts\Activate.ps1
   # Linux/macOS:
   source .venv/bin/activate
   ```
4. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
5. **Create a feature branch**:
   ```bash
   git checkout -b feature/my-new-feature
   ```

---

## 🧪 Testing Guidelines

Before opening a pull request, ensure all unit tests pass:

```bash
python -m unittest discover -s . -p 'test*.py' -v
```

If you add new data models or contracts in `contracts.py`, add corresponding test cases in `test_contracts.py`.

---

## 🔒 Security Best Practices

- **Never commit `.env` or API keys.**
- If you modify `sandbox.py`, ensure AST security rules remain strict and forbid dangerous dynamic evaluation (`eval`, `exec`, `__import__`, `subprocess.*`, `socket.*`).

---

## 📬 Submitting a Pull Request

1. Push your branch to GitHub:
   ```bash
   git push origin feature/my-new-feature
   ```
2. Open a **Pull Request** targeting the `main` branch.
3. Provide a clear description of the changes, bug fixes, or enhancements made.
