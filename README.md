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

### 0. Remove Existing Anaconda/Miniconda (if installed)

**Check if you have Anaconda/Miniconda:** Open Terminal and look at your prompt. If you see `(base)` before your username, you have conda installed.

**To remove it:**

1. Remove conda initialization:
```bash
conda init --reverse
```

2. Close and reopen Terminal

3. Remove the installation directory (check which one exists):
```bash
# Check which directory exists:
ls -la ~ | grep -E 'anaconda|miniconda|miniforge'

# Remove the one you find (usually one of these):
rm -rf ~/anaconda3
# OR
rm -rf ~/miniconda3
# OR
rm -rf ~/opt/anaconda3
```

4. Clean up your shell config file (removes conda references):
```bash
# For macOS default (zsh):
nano ~/.zshrc
# Look for lines with 'conda' and delete them, then save (Ctrl+O, Enter, Ctrl+X)

# Or use this automated cleanup:
sed -i.bak '/conda/d' ~/.zshrc
```

5. Close and reopen Terminal - `(base)` should be gone

**If you're not sure what to remove,** you can also just proceed with Miniforge installation, but having multiple conda installations can cause conflicts.

---

### 1. Install Miniforge

**Check your chip:**
- Apple menu → About This Mac
- **M1/M2/M3/M4** = Apple Silicon
- **Intel** = Intel chip

**Install:**

1. Download from [Miniforge Releases](https://github.com/conda-forge/miniforge/releases)
   - **Apple Silicon:** `Miniforge3-MacOSX-arm64.sh`
   - **Intel:** `Miniforge3-MacOSX-x86_64.sh`

2. **Install Miniforge:**

   Open Terminal and run:
   
   ```bash
   cp ~/Downloads/Miniforge3-MacOSX-arm64.sh ~
   cd ~
   bash Miniforge3-MacOSX-arm64.sh
   ```
   
   **Note:** Replace `arm64` with `x86_64` if you have an Intel Mac
   
   **Why copy it?** macOS has strict security on the Downloads folder. Moving it to your home directory avoids permission issues.

3. Follow prompts:
   - Press Enter to review license
   - Type `yes` to accept
   - Press Enter for default location
   - **Important:** When asked "Do you wish to update your shell profile to automatically initialize conda?" → Type `yes`
     - This allows you to use `mamba activate` commands in any Terminal window
     - Note: Miniforge uses mamba for commands but conda for initialization - this is normal
   
4. **CRITICAL: Restart Terminal completely**
   
   ⚠️ **You MUST do this or mamba won't work!**
   
   - Press **Cmd+Q** to **completely quit Terminal** (don't just close the window!)
   - Wait 2 seconds
   - Open Terminal again from Applications
   - You should now see `(base)` at the start of your prompt
   
   **Test it worked:**
   ```bash
   mamba --version
   ```
   
   If you see a version number, you're good! If you see "command not found", run:
   ```bash
   ~/miniforge3/bin/conda init zsh
   ```
   Then quit and reopen Terminal again (Cmd+Q).
   
5. **(Optional) Disable auto-activation of base environment:**
   
   If you don't want to see `(base)` in your prompt every time:
   ```bash
   conda config --set auto_activate_base false
   ```
   Then quit Terminal (Cmd+Q) and reopen. You'll only see environment names when you explicitly activate them.

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