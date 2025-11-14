# example code and instructions for module 6 python for data visualization course
# Python Setup for Data Visualization

## What This Guide Does

This guide walks you through setting up Python for data visualization work using **Miniforge/Mamba**. 

### Why Mamba and Not Anaconda?

**Mamba** is a package manager (like an app store for Python packages) that's built on top of conda/Anaconda infrastructure. Until recently, everyone used Anaconda, but Anaconda now requires commercial licenses for institutions and companies. **Miniforge with Mamba is completely free and open-source** with no licensing restrictions.

**What you'll install:**
1. **Miniforge** - Provides the mamba package manager
2. **VS Code** - Your coding environment
3. **Python environment** - Python 3.12 with data visualization libraries
4. **Jupyter notebooks** - Interactive coding interface

**Time needed:** ~15-20 minutes

---

## Prerequisites

Create a working directory for your projects:

**Windows:** `C:\Users\<YourUsername>\Documents\data_viz_seminar`  
**macOS:** `~/Documents/data_viz_seminar`

---

## Windows Setup

### Step 1: Install Miniforge

1. Go to [Miniforge Releases](https://github.com/conda-forge/miniforge/releases)
2. Download **Miniforge3-Windows-x86_64.exe**
3. Run the installer:
   - Select **"Just Me"**
   - ⚠️ **Important:** Do NOT check "Add to PATH"
   - Click through to complete installation

### Step 2: Install VS Code

1. Go to [code.visualstudio.com](https://code.visualstudio.com/)
2. Download Windows version and run installer
3. During installation, check:
   - ✅ **"Add to PATH"**
   - ✅ **"Register as code editor"**

### Step 3: Configure Mamba to Use Only Conda-Forge

⚠️ **Critical step to avoid Anaconda licensing issues**

1. Open **Miniforge Prompt** from Start Menu (search "Miniforge")
2. Run these commands one by one:

```bash
conda config --remove channels defaults
conda config --add channels conda-forge
conda config --set channel_priority strict
```

### Step 4: Create Your Python Environment

In **Miniforge Prompt**, run:

```bash
mamba create -n viz_env python=3.12 -y
mamba activate viz_env
mamba install jupyter pandas matplotlib seaborn numpy scipy -y
```

**If you get SSL certificate errors:** Run `conda config --set ssl_verify false` then try again

### Step 5: Setup VS Code

1. Open **VS Code**
2. **File → Open Folder** → Select `Documents\data_viz_seminar`
3. Press **Ctrl+Shift+P** → Type "Python: Select Interpreter" → Choose the one with **`viz_env`**
4. Create a new file: Click **"New File"** → Save as `test_notebook.ipynb`
5. If prompted, click **"Install"** for the Jupyter extension
6. **Click the kernel selector** (top-right corner) → Choose **`viz_env`**

### Step 6: Test Your Setup

In the first cell of your notebook, paste and run (Shift+Enter):

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

print("✓ Setup successful!")
print(f"Pandas: {pd.__version__}")
print(f"Matplotlib: {plt.matplotlib.__version__}")
print(f"Seaborn: {sns.__version__}")
```

If you see version numbers, you're done! 🎉

---

## macOS Setup

### Step 1: Download Miniforge

1. Go to [Miniforge Releases](https://github.com/conda-forge/miniforge/releases)
2. Check your Mac chip first:
   - Click **Apple menu** → **About This Mac**
   - **M1/M2/M3/M4** → Download **Miniforge3-MacOSX-arm64.sh**
   - **Intel** → Download **Miniforge3-MacOSX-x86_64.sh**

### Step 2: Install Miniforge

1. Open **Terminal** (Applications → Utilities → Terminal)
2. Run these commands:

```bash
cp ~/Downloads/Miniforge3-MacOSX-arm64.sh ~
cd ~
bash Miniforge3-MacOSX-arm64.sh
```

**Note:** Change `arm64` to `x86_64` if you have Intel Mac

3. Follow the prompts:
   - Press **Enter** to review license
   - Type **`yes`** to accept
   - Press **Enter** for default location
   - ⚠️ **Important:** When asked about initializing conda, type **`yes`**

4. **CRITICAL: Restart Terminal**
   - Press **Cmd+Q** to completely quit Terminal
   - Wait 2 seconds
   - Open Terminal again
   - You should see **(base)** at the start of your prompt

5. **Test that mamba works:**

```bash
mamba --version
```

If you see a version number, continue. If not, run:

```bash
~/miniforge3/bin/conda init zsh
```

Then quit Terminal (Cmd+Q) and reopen.

### Step 3: Install VS Code

1. Download from [code.visualstudio.com](https://code.visualstudio.com/)
2. Open the download and drag VS Code to **Applications** folder

### Step 4: Configure Mamba to Use Only Conda-Forge

⚠️ **Critical step to avoid Anaconda licensing issues**

Open Terminal and run these commands one by one:

```bash
conda config --remove channels defaults
conda config --add channels conda-forge
conda config --set channel_priority strict
```

### Step 5: Create Your Python Environment

In Terminal, run:

```bash
mamba create -n viz_env python=3.12 -y
mamba activate viz_env
mamba install jupyter pandas matplotlib seaborn numpy scipy -y
```

**If you get SSL certificate errors:** Run `conda config --set ssl_verify false` then try again

### Step 6: Setup VS Code

1. Open **VS Code**
2. **File → Open** → Select `Documents/data_viz_seminar`
3. Press **Cmd+Shift+P** → Type "Python: Select Interpreter" → Choose the one with **`viz_env`**
4. Create a new file: Click **"New File"** → Save as `test_notebook.ipynb`
5. If prompted, click **"Install"** for the Jupyter extension
6. **Click the kernel selector** (top-right corner) → Choose **`viz_env`**

### Step 7: Test Your Setup

In the first cell of your notebook, paste and run (Shift+Enter):

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

print("✓ Setup successful!")
print(f"Pandas: {pd.__version__}")
print(f"Matplotlib: {plt.matplotlib.__version__}")
print(f"Seaborn: {sns.__version__}")
```

If you see version numbers, you're done! 🎉

### Step 8 (Optional): Hide the (base) Prompt

If you don't want to see **(base)** every time you open Terminal:

```bash
conda config --set auto_activate_base false
```

Then quit Terminal (Cmd+Q) and reopen. You'll only see environment names when you activate them.

---

## Common Issues

### "Kernel not found" in VS Code
- **Solution:** Restart VS Code after creating the environment

### "Import error" even though packages installed
- **Solution:** Make sure you selected the correct kernel (the one with `viz_env`)
- Check by running: `import sys; print(sys.executable)` - should show path with `viz_env`

### "mamba: command not found" (macOS)
- **Solution:** You didn't restart Terminal. Press Cmd+Q to quit completely, then reopen

### "mamba: command not found" (Windows)
- **Solution:** You're in regular Command Prompt. Use **Miniforge Prompt** from Start Menu

### SSL certificate errors during installation
- **Solution:** Run `conda config --set ssl_verify false` then try again
- This is common on institutional networks with firewalls

---

## Using Your Environment Later

**Activate environment:**
- **Windows:** Open Miniforge Prompt → `mamba activate viz_env`
- **macOS:** Open Terminal → `mamba activate viz_env`

**Install more packages:**
```bash
mamba activate viz_env
mamba install plotly scikit-learn -y
```

**List installed packages:**
```bash
mamba list
```

---

## Why Miniforge Instead of Anaconda?

| Feature | Miniforge | Anaconda |
|---------|-----------|----------|
| **License** | ✅ Free (all uses) | ⚠️ Requires commercial license for institutions |
| **Speed** | ✅ Fast (mamba) | ⚠️ Slower (conda) |
| **Packages** | ✅ conda-forge | ⚠️ Limited free channels |
| **Recommended** | ✅ By universities | ⚠️ Licensing restrictions since 2024 |

---

## Quick Reference Commands

```bash
# Activate environment
mamba activate viz_env

# Deactivate environment
mamba deactivate

# Install new package
mamba install package-name -y

# Update package
mamba update package-name

# List all environments
mamba env list

# Remove environment
mamba env remove -n viz_env
```

---

## Resources

- [Miniforge GitHub](https://github.com/conda-forge/miniforge)
- [Mamba Documentation](https://mamba.readthedocs.io/)
- [VS Code Python Tutorial](https://code.visualstudio.com/docs/python/python-tutorial)

---

## Getting Help

If you run into issues:
1. Check the **Common Issues** section above
2. Make sure you followed every step exactly
3. Try restarting VS Code or Terminal
4. Ask your instructor for help

---
