# Python for Data Visualization — Setup Guide

Example code and instructions for **Module 6: Python for Data Visualization**.

This guide walks you through installing everything you need, step by step.
It is written for people who are **new to Python and programming** — just follow
the section for your operating system from top to bottom.

> **Time needed:** about 15–20 minutes (most of it is waiting for downloads).

---

## What you will install

1. **Miniforge** — a free package manager that installs Python and the libraries we need.
2. **VS Code** — the program you will write and run code in.
3. **The course environment (`viz_env`)** — Python 3.12 plus all required libraries, created in one command.

### Why Miniforge and not Anaconda?

You may have heard of **Anaconda**. We do **not** use it, because Anaconda now
requires a **paid commercial license** for universities and research institutes.
**Miniforge is free for everyone**, open-source, and installs the exact same
libraries. It comes with a fast installer called **`mamba`** that we use below.

> You do **not** need to understand the difference to follow this guide. Just
> install Miniforge as described.

---

## Step 0 — Download the course materials (everyone)

1. Go to the course repository: **https://github.com/johannesbmueller/data_viz**
2. Click the green **`Code`** button → **Download ZIP**.
3. **Extract (unzip) the file** into your Documents folder. You will get a folder
   named `data_viz-main`.
   - **Windows:** `C:\Users\<YourName>\Documents\data_viz-main`
   - **macOS / Linux:** `~/Documents/data_viz-main`
4. *(Optional)* Rename the folder to just `data_viz`.

**Remember where this folder is — you will open it in VS Code later.**

---

## 🪟 Windows

### 1. Install Miniforge
1. Download the installer: **https://conda-forge.org/download/**
   → choose **Windows**, file name `Miniforge3-Windows-x86_64.exe`.
2. Run the downloaded `.exe`. In the installer:
   - Choose **"Just Me"**.
   - Leave the install location at the default.
   - On the options screen, the defaults are fine. (Leaving *"Add to PATH"*
     **unchecked** is recommended — you will use the Miniforge Prompt instead.)
   - Click **Install** and wait until it finishes.

### 2. Install VS Code
1. Download from **https://code.visualstudio.com/**
2. Run the installer. When asked, tick:
   - ✅ **Add to PATH**
   - ✅ **Register Code as an editor for supported file types**

### 3. Create the course environment
1. Open the **Miniforge Prompt** from the Start Menu (type "Miniforge" to find it).
   A black window opens with `(base)` at the start of the line.
2. Go into the course folder (adjust the path if you renamed it):
   ```bat
   cd %USERPROFILE%\Documents\data_viz-main
   ```
3. Create the environment from the included file (this downloads everything —
   it can take a few minutes):
   ```bat
   mamba env create -f environment.yml
   ```

✅ When it finishes you have an environment called **`viz_env`** with every
library the course needs. **Continue to "Open the course in VS Code" below.**

---

## 🍎 macOS

### 1. Install Miniforge
1. Find out your chip: **Apple menu → About This Mac**.
   - **Apple M1/M2/M3/M4** → you have an *Apple Silicon* (arm64) Mac.
   - **Intel** → you have an *Intel* (x86_64) Mac.
2. Open **Terminal** (Applications → Utilities → Terminal) and run **one** of the
   following blocks, depending on your chip. Copy the whole block, paste it, press Enter.

   **Apple Silicon (M1/M2/M3/M4):**
   ```bash
   curl -L -o ~/Miniforge3.sh https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-arm64.sh
   bash ~/Miniforge3.sh
   ```

   **Intel Mac:**
   ```bash
   curl -L -o ~/Miniforge3.sh https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-x86_64.sh
   bash ~/Miniforge3.sh
   ```
3. Follow the prompts in Terminal:
   - Press **Enter** / **Space** to scroll through the license, then type **`yes`**.
   - Press **Enter** to accept the default install location.
   - When asked **"Do you wish to update your shell profile…?"**, type **`yes`**.
4. **Close Terminal completely** (**Cmd+Q**), wait a moment, then **open it again**.
   You should now see **`(base)`** at the start of the prompt.

   > If you do **not** see `(base)`, run this once, then quit and reopen Terminal:
   > ```bash
   > ~/miniforge3/bin/conda init zsh
   > ```

### 2. Install VS Code
1. Download from **https://code.visualstudio.com/**
2. Open the downloaded file and drag **Visual Studio Code** into your **Applications** folder.

### 3. Create the course environment
In Terminal:
```bash
cd ~/Documents/data_viz-main
mamba env create -f environment.yml
```
(This downloads everything and can take a few minutes.)

✅ When it finishes you have an environment called **`viz_env`**.
**Continue to "Open the course in VS Code" below.**

---

## 🐧 Linux

### 1. Install Miniforge
Open a **terminal** and run (works on most 64-bit PCs):
```bash
curl -L -o ~/Miniforge3.sh https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash ~/Miniforge3.sh
```
> On an ARM machine (e.g. Raspberry Pi, some servers), replace `x86_64` with `aarch64`.

Follow the prompts:
- Press **Enter** to scroll the license, type **`yes`** to accept.
- Press **Enter** for the default location.
- When asked to update your shell profile, type **`yes`**.

Then **close and reopen the terminal**. You should see **`(base)`** at the start of the line.

### 2. Install VS Code
- **Debian/Ubuntu:** download the `.deb` from **https://code.visualstudio.com/** and
  double-click it, or run `sudo apt install ./<downloaded-file>.deb`.
- **Fedora/RHEL:** download the `.rpm` and run `sudo dnf install ./<downloaded-file>.rpm`.
- Or install via your distro's software center / snap: `sudo snap install code --classic`.

### 3. Create the course environment
```bash
cd ~/Documents/data_viz-main
mamba env create -f environment.yml
```

✅ When it finishes you have an environment called **`viz_env`**.
**Continue below.**

---

## Open the course in VS Code (all systems)

1. Open **VS Code**.
2. **File → Open Folder…** and select your course folder
   (`Documents/data_viz-main`).
3. When VS Code asks, install the recommended **Python** and **Jupyter**
   extensions (a notification appears in the bottom-right; click **Install**).
4. Open one of the notebooks in the `notebooks` folder (a file ending in `.ipynb`).
5. In the **top-right** of the notebook, click **Select Kernel** →
   **Python Environments…** → choose **`viz_env`**.

### Test that everything works
Create a new file `notebooks/test_notebook.ipynb`, paste this into the first cell,
and press **Shift + Enter** to run it:

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

print("✓ Setup successful!")
print(f"Pandas:     {pd.__version__}")
print(f"Matplotlib: {plt.matplotlib.__version__}")
print(f"Seaborn:    {sns.__version__}")
```

If you see version numbers printed, **you are done!** 🎉

---

## Troubleshooting

**`mamba: command not found` (macOS / Linux)**
You didn't reopen the terminal after installing. Fully quit (macOS: **Cmd+Q**) and
reopen it. Still nothing? Run `~/miniforge3/bin/conda init zsh` (macOS) or
`~/miniforge3/bin/conda init bash` (Linux), then reopen the terminal.

**`mamba: command not found` (Windows)**
You are in the regular Command Prompt/PowerShell. Open the **Miniforge Prompt**
from the Start Menu instead.

**VS Code can't find `viz_env` / "Select Kernel" doesn't list it**
Close and reopen VS Code after creating the environment, then try **Select Kernel**
again. Make sure the Python and Jupyter extensions are installed.

**"No module named …" when running a notebook**
You are using the wrong kernel. Click **Select Kernel** (top-right) and pick
**`viz_env`**. To confirm which Python you're on, run in a cell:
```python
import sys; print(sys.executable)   # the path should contain "viz_env"
```

**SSL / certificate errors during install (common on institute networks)**
Run this once, then retry the `mamba env create` command:
```bash
conda config --set ssl_verify false
```

**Hide the `(base)` prefix in your terminal (macOS / Linux, optional)**
```bash
conda config --set auto_activate_base false
```
Reopen the terminal afterwards.

---

## Using your environment later

```bash
# Activate it (do this before installing packages from the terminal)
mamba activate viz_env

# Install an extra package
mamba install <package-name>

# List installed packages
mamba list

# List all environments
mamba env list

# Start over: delete and recreate the environment
mamba env remove -n viz_env
mamba env create -f environment.yml
```

---

## Resources

- Miniforge download page: https://conda-forge.org/download/
- Miniforge on GitHub: https://github.com/conda-forge/miniforge
- VS Code Python tutorial: https://code.visualstudio.com/docs/python/python-tutorial

## Getting help

1. Re-read the **Troubleshooting** section above.
2. Make sure you followed every step for *your* operating system, in order.
3. Try restarting VS Code or your terminal.
4. Ask your instructor — and say which step failed and the exact error message.

---

*Last updated: June 2026*
