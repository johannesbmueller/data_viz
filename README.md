# example code and instructions for module 6 python for data visualization course


# Python Setup for Data Visualization

A beginner-friendly guide for setting up Python with Jupyter notebooks for data visualization work. This guide uses **Miniforge** (free, open-source, no licensing restrictions) instead of Anaconda.

---

## Prerequisites

Create a working directory for your projects:

**Location:**
- **Windows:** `C:\Users\<YourUsername>\Documents\data_viz_seminar`
- **macOS/Linux:** `~/Documents/data_viz_seminar`

---

## Windows Setup

### 1. Install Miniforge

Miniforge provides `mamba` (a faster `conda`) and is completely free for all use cases.

1. Download from [Miniforge Releases](https://github.com/conda-forge/miniforge/releases)
2. Get **Miniforge3-Windows-x86_64.exe**
3. Run installer:
   - Select "Just Me"
   - ⚠️ **Do NOT add to PATH** (use Miniforge Prompt instead)

### 2. Install VS Code

1. Download from [code.visualstudio.com](https://code.visualstudio.com/)
2. During installation, check:
   - ✅ Add to PATH
   - ✅ Register as code editor

### 3. Create Environment

Open **Miniforge Prompt** (from Start Menu):

```bash
mamba create -n viz_env python=3.12 -y
mamba activate viz_env
mamba install jupyter pandas matplotlib seaborn numpy scipy -y
```

### 4. Setup VS Code

1. Open VS Code → **File → Open Folder** → Select `data_viz_seminar`
2. Press `Ctrl+Shift+P` → Type "Python: Select Interpreter" → Choose `viz_env`
3. Create new file: `first_notebook.ipynb`
4. Install Jupyter extension (if prompted)
5. **Select kernel:** Click kernel selector (top-right) → Choose `viz_env`

### 5. Verify Setup

Run in first cell:

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Verify setup
print("✓ Setup successful!")
print(f"Pandas: {pd.__version__}")
print(f"Matplotlib: {plt.matplotlib.__version__}")
print(f"Seaborn: {sns.__version__}")
```

---

## macOS Setup

### 1. Install Miniforge

**Check your chip:**
- Apple menu → About This Mac
- **M1/M2/M3/M4** = Apple Silicon
- **Intel** = Intel chip

**Install:**

1. Download from [Miniforge Releases](https://github.com/conda-forge/miniforge/releases)
   - **Apple Silicon:** `Miniforge3-MacOSX-arm64.sh`
   - **Intel:** `Miniforge3-MacOSX-x86_64.sh`

2. Open Terminal and run:

```bash
cd ~/Downloads
bash Miniforge3-MacOSX-*.sh
```

3. Follow prompts:
   - Press Enter to review license
   - Type `yes` to accept
   - Press Enter for default location
   - Type `yes` to initialize
   - Close and reopen Terminal

### 2. Install VS Code

1. Download from [code.visualstudio.com](https://code.visualstudio.com/)
2. Drag to Applications folder
3. (Optional) Add to PATH: `Cmd+Shift+P` → "Shell Command: Install 'code' command in PATH"

### 3. Create Environment

Open Terminal:

```bash
mamba create -n viz_env python=3.12 -y
mamba activate viz_env
mamba install jupyter pandas matplotlib seaborn numpy scipy -y
```

### 4. Setup VS Code

1. Open VS Code → **File → Open** → Select `data_viz_seminar`
2. Press `Cmd+Shift+P` → Type "Python: Select Interpreter" → Choose `viz_env`
3. Create new file: `first_notebook.ipynb`
4. Install Jupyter extension (if prompted)
5. **Select kernel:** Click kernel selector (top-right) → Choose `viz_env`

### 5. Verify Setup

Run in first cell:

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Verify setup
print("✓ Setup successful!")
print(f"Pandas: {pd.__version__}")
print(f"Matplotlib: {plt.matplotlib.__version__}")
print(f"Seaborn: {sns.__version__}")
```

---

## Troubleshooting

### Kernel not appearing in VS Code
**Solution:** Restart VS Code after creating the environment

### Import errors despite installed packages
**Solution:** Verify correct kernel is selected (one with `viz_env`)
```python
import sys
print(sys.executable)  # Should show viz_env path
```

### Jupyter extension won't install
**Solution:** Install manually from Extensions marketplace (`Ctrl/Cmd+Shift+X`)

### "mamba: command not found"
- **Windows:** Use **Miniforge Prompt**, not Command Prompt
- **macOS:** Close and reopen Terminal after installation

### Activating environment later
- **Windows:** Open Miniforge Prompt → `mamba activate viz_env`
- **macOS:** Open Terminal → `mamba activate viz_env`

---

## Why Miniforge?

| Feature | Miniforge | Anaconda |
|---------|-----------|----------|
| **License** | ✅ Free (all uses) | ⚠️ Requires license for commercial/institutional use |
| **Speed** | ✅ Fast (mamba) | ⚠️ Slower (conda) |
| **Packages** | ✅ conda-forge | ⚠️ Limited free channels |
| **Academic use** | ✅ Recommended | ⚠️ Licensing issues |

---

## Additional Packages

Install more packages anytime:

```bash
mamba activate viz_env
mamba install plotly scikit-learn statsmodels -y
```

---

## Quick Reference

**Activate environment:**
```bash
mamba activate viz_env
```

**Deactivate environment:**
```bash
mamba deactivate
```

**List installed packages:**
```bash
mamba list
```

**Update package:**
```bash
mamba update pandas
```

**Remove environment:**
```bash
mamba env remove -n viz_env
```

---

## Resources

- [Miniforge GitHub](https://github.com/conda-forge/miniforge)
- [Mamba Documentation](https://mamba.readthedocs.io/)
- [VS Code Python Tutorial](https://code.visualstudio.com/docs/python/python-tutorial)
- [Jupyter Documentation](https://jupyter.org/documentation)

---

## License

This guide is provided as-is for educational purposes. Feel free to adapt and share.
