# 📦 Installation Guide for SciTrans-LLMs

Complete step-by-step installation guide for setting up the SciTrans-LLMs system.

---

## 🖥️ System Requirements

### Minimum Requirements

- **Operating System**: Linux, macOS, or Windows 10/11
- **Python**: 3.9 or higher (we use Python 3.9)
- **RAM**: 4 GB minimum, 8 GB recommended
- **Disk Space**: 2 GB for installation + models
- **Internet**: Required for initial setup and API usage

### Recommended Requirements

- **RAM**: 16 GB (for processing large PDFs)
- **GPU**: NVIDIA GPU with CUDA support (optional, for YOLO layout detection)
- **Disk Space**: 5 GB (for models and cache)

---

## 📥 Step 1: Install Python 3.9+

### macOS

```bash
# Using Homebrew
brew install python@3.9

# Verify installation
python3 --version
```

### Linux (Ubuntu/Debian)

```bash
# Update package list
sudo apt update

# Install Python 3.9
sudo apt install python3.9 python3.9-venv python3.9-dev

# Verify installation
python3 --version
```

### Windows

1. Download Python 3.9+ from [python.org](https://www.python.org/downloads/)
2. Run installer and **check "Add Python to PATH"**
3. Verify in Command Prompt:
   ```cmd
   python --version
   ```

---

## 📁 Step 2: Clone the Repository

```bash
# Clone from GitHub
git clone https://github.com/uglydavy/SciTrans-LLMs.git

# Navigate to project directory
cd SciTrans-LLMs

# Verify you're in the right place
ls -la  # Should see README.md, pyproject.toml, etc.
```

---

## 🐍 Step 3: Create Virtual Environment

**Why virtual environment?** Keeps project dependencies isolated from system Python.

### On Linux/macOS

```bash
# Create virtual environment
python3 -m venv .venv

# Activate it
source .venv/bin/activate

# Your prompt should now show (.venv)
```

### On Windows (Command Prompt)

```cmd
# Create virtual environment
python -m venv .venv

# Activate it
.venv\Scripts\activate.bat
```

### On Windows (PowerShell)

```powershell
# Create virtual environment
python -m venv .venv

# Activate it
.venv\Scripts\Activate.ps1
```

**Note**: Keep this terminal window open with the virtual environment activated for all following steps!

---

## 📦 Step 4: Install Dependencies

### Option A: Full Installation (Recommended)

Installs everything including GUI, YOLO, and all LLM backends:

```bash
# Upgrade pip first
pip install --upgrade pip

# Install with all features
pip install -e ".[full]"
```

### Option B: Minimal Installation

Core features only (no GUI, no YOLO):

```bash
pip install -e .
```

### Option C: Custom Installation

Choose what you need:

```bash
# Core + OpenAI only
pip install -e ".[openai]"

# Core + all LLM backends
pip install -e ".[all-llm]"

# Core + YOLO layout detection
pip install -e ".[layout]"

# Development tools
pip install -e ".[dev]"

# Combine multiple extras
pip install -e ".[openai,layout]"
```

**Installation time**: 5-10 minutes depending on your internet speed.

---

## ✅ Step 5: Verify Installation

```bash
# Check if scitrans command is available
scitrans --version

# Should output: SciTrans-LLMs v0.1.0
```

If you get "command not found":

```bash
# Use module form instead
python3 -m scitrans_llms --version
```

### Run System Info Check

```bash
# Check available backends and dependencies
scitrans info
```

You should see:
- ✓ Available backends (at least free and dictionary)
- ✓ or ✗ for optional dependencies (PyMuPDF, SacreBLEU, etc.)
- API key status

### Run Demo

```bash
# Quick test with offline dictionary translator
scitrans demo
```

This should:
1. Create a sample document
2. Apply masking
3. Translate (using dummy backend)
4. Show statistics

If this works, your installation is successful! 🎉

---

## 🔑 Step 6: Configure API Keys

To use real LLM backends (OpenAI, DeepSeek, Anthropic), you need API keys.

### Method 1: Using CLI Key Manager (Recommended)

```bash
# Set OpenAI key
scitrans keys set openai
# You'll be prompted to enter your key securely

# Verify it's set
scitrans keys list
```

### Method 2: Environment Variables

Add to your shell profile (`~/.bashrc`, `~/.zshrc`, or `~/.bash_profile`):

```bash
export OPENAI_API_KEY="sk-your-key-here"
export DEEPSEEK_API_KEY="sk-your-key-here"
export ANTHROPIC_API_KEY="sk-ant-your-key-here"
```

Then reload:

```bash
source ~/.bashrc  # or ~/.zshrc
```

### Method 3: Interactive Setup Script

```bash
python3 scripts/setup_keys.py
```

### Where to Get API Keys

| Service | Website | Notes |
|---------|---------|-------|
| OpenAI | [platform.openai.com](https://platform.openai.com) | GPT-4, GPT-4o |
| DeepSeek | [platform.deepseek.com](https://platform.deepseek.com) | Cheaper alternative |
| Anthropic | [console.anthropic.com](https://console.anthropic.com) | Claude models |

---

## 🧪 Step 7: Test Everything

### Test 1: Text Translation

```bash
scitrans translate --text "Hello world" --backend dummy
```

Expected: Should show translated text and statistics.

### Test 2: API Key Test (if configured)

```bash
scitrans translate --text "Machine learning is powerful" --backend openai
```

Expected: Should translate using GPT model.

### Test 3: PDF Translation (if PyMuPDF installed)

```bash
# Get a sample PDF first
# Then translate it
scitrans translate -i sample.pdf -o output.pdf --backend dummy
```

### Test 4: GUI Launch

```bash
scitrans gui
```

Expected: Browser window opens with Gradio interface.

---

## 🐛 Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'scitrans_llms'"

**Solution:**
```bash
# Make sure you're in the project directory
cd /path/to/SciTrans-LLMs

# Make sure virtual environment is activated
source .venv/bin/activate  # Linux/macOS
# or
.venv\Scripts\activate  # Windows

# Reinstall
pip install -e .
```

### Issue: "scitrans: command not found"

**Solution:**
```bash
# Use module form
python3 -m scitrans_llms --help

# Or add to PATH
export PATH="$PATH:$(pwd)/.venv/bin"
```

### Issue: "ImportError: numpy._core..."

**Solution:**
```bash
# Reinstall with correct NumPy version
pip install "numpy>=1.25.0,<2.0.0" --force-reinstall
```

### Issue: "API key not found"

**Solution:**
```bash
# Check key status
scitrans keys list

# Set the key
scitrans keys set openai

# Or export environment variable
export OPENAI_API_KEY="sk-..."
```

### Issue: GUI doesn't start

**Solution:**
```bash
# Install GUI dependencies
pip install gradio>=4.0.0

# Or install full package
pip install -e ".[full]"
```

### Issue: PDF parsing fails

**Solution:**
```bash
# Install PDF dependencies
pip install PyMuPDF>=1.23.0

# Verify
python3 -c "import fitz; print('PyMuPDF OK')"
```

### Issue: "Permission denied" errors on macOS/Linux

**Solution:**
```bash
# Fix permissions
chmod +x scripts/*.py

# Or run with python3
python3 scripts/setup_keys.py
```

---

## 🔄 Updating the System

### Pull Latest Changes

```bash
# Make sure you're in the project directory
cd /path/to/SciTrans-LLMs

# Pull updates from GitHub
git pull origin main

# Reinstall (in case dependencies changed)
pip install -e ".[full]" --upgrade
```

---

## 🗑️ Uninstallation

### Remove Package

```bash
# Uninstall
pip uninstall scitrans_llms

# Remove virtual environment
rm -rf .venv

# Remove cached data (optional)
rm -rf ~/.scitrans
```

---

## 📚 Next Steps

After successful installation:

1. **Read the README**: `README.md` for full documentation
2. **Set up API keys**: See "API Key Setup" section in README
3. **Try examples**: Run `scitrans demo`
4. **Explore CLI**: Run `scitrans --help`
5. **Launch GUI**: Run `scitrans gui`
6. **Run experiments**: See `EXPERIMENTS.md` for research workflows

---

## 🆘 Getting Help

If you encounter issues not covered here:

1. **Check system info**: `scitrans info`
2. **Check documentation**: `README.md`, `EXPERIMENTS.md`, `THESIS_GUIDE.md`
3. **Check GitHub issues**: https://github.com/uglydavy/SciTrans-LLMs/issues
4. **Contact**: aknk.v@pm.me

---

## 📋 Installation Checklist

- [ ] Python 3.9+ installed
- [ ] Repository cloned
- [ ] Virtual environment created and activated
- [ ] Dependencies installed (`pip install -e ".[full]"`)
- [ ] `scitrans --version` works
- [ ] `scitrans info` shows available backends
- [ ] `scitrans demo` runs successfully
- [ ] API keys configured (for LLM backends)
- [ ] GUI launches successfully

---

**Installation Complete!** 🎉

You're now ready to use SciTrans-LLMs for scientific document translation.

