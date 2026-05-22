# ============================================================
# CELL 1 — Mount Google Drive
# ============================================================
from google.colab import drive
drive.mount('/content/drive')

# ============================================================
# CELL 2 — Create project folder structure
# ============================================================
import os

# Base directory on Drive
DRIVE_BASE = "/content/drive/MyDrive/imdb_peft_project"

DIRS = {
    "root":         DRIVE_BASE,
    "checkpoints":  f"{DRIVE_BASE}/checkpoints/roberta_lora",
    "checkpoints2": f"{DRIVE_BASE}/checkpoints/deberta_lora",
    "oof":          f"{DRIVE_BASE}/oof_predictions",
    "results":      f"{DRIVE_BASE}/results",
    "notebooks":    f"{DRIVE_BASE}/notebooks",
    "code":         f"{DRIVE_BASE}/code",
}

for path in DIRS.values():
    os.makedirs(path, exist_ok=True)
    print(f"✓ {path}")

print("\nFolder structure ready.")

# ============================================================
# CELL 3 — Clone or pull GitHub repository
# ============================================================
# FIRST USE: clone the repository
# SUBSEQUENT USES: pull latest changes

import subprocess

GITHUB_USERNAME = "DenizhanOngun"
GITHUB_REPO     = "lora-imdb-classifier"
GITHUB_EMAIL    = "denizhan.eser@hotmail.com"

import getpass
GITHUB_TOKEN = getpass.getpass("Enter GitHub Token: ")

REPO_URL  = f"https://{GITHUB_USERNAME}:{GITHUB_TOKEN}@github.com/{GITHUB_USERNAME}/{GITHUB_REPO}.git"
REPO_PATH = f"{DRIVE_BASE}/code/{GITHUB_REPO}"

# Configure git identity
subprocess.run(f'git config --global user.email "{GITHUB_EMAIL}"', shell=True)
subprocess.run(f'git config --global user.name "{GITHUB_USERNAME}"', shell=True)

if not os.path.exists(REPO_PATH):
    print("Repository not found, cloning...")
    subprocess.run(f"git clone {REPO_URL} {REPO_PATH}", shell=True)
    print("✓ Clone complete.")
else:
    print("Repository exists, pulling latest changes...")
    subprocess.run(f"cd {REPO_PATH} && git remote set-url origin {REPO_URL}", shell=True)
    subprocess.run(f"cd {REPO_PATH} && git pull", shell=True)
    print("✓ Pull complete.")

print(f"\nRepository path: {REPO_PATH}")

# ============================================================
# CELL 4 — Push to GitHub
# (call after each important step)
# ============================================================
def push_to_github(commit_message: str):
    """
    Stages all changes and pushes to GitHub.
    Usage: push_to_github("fold 2 complete")
    """
    import subprocess

    commands = [
        f"cd {REPO_PATH} && git add .",
        f"cd {REPO_PATH} && git commit -m '{commit_message}'",
        f"cd {REPO_PATH} && git push",
    ]

    for cmd in commands:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            if "nothing to commit" in result.stdout:
                print("ℹ No changes to commit, push skipped.")
                return
            print(f"⚠ Error: {result.stderr}")
            return

    print(f"✓ Pushed to GitHub: '{commit_message}'")

# ============================================================
# CELL 5 — Copy a code file to the local repository
# (call after creating each new .py file)
# ============================================================
import shutil

def save_code_to_repo(source_path: str, filename: str = None):
    """
    Copies a .py file into the local repository folder.
    Usage: save_code_to_repo("/content/01_data_preprocessing.py")
    """
    if filename is None:
        filename = os.path.basename(source_path)

    dest_path = os.path.join(REPO_PATH, filename)
    shutil.copy2(source_path, dest_path)
    print(f"✓ {filename} → copied to repository.")

# ============================================================
# CELL 6 — Drive helper functions
# (for model weights and OOF predictions)
# ============================================================
import numpy as np
import json

def save_oof(predictions: np.ndarray, model_name: str, fold: int):
    """Saves OOF predictions to Drive."""
    path = f"{DIRS['oof']}/{model_name}_fold{fold}.npy"
    np.save(path, predictions)
    print(f"✓ OOF saved: {path}")

def load_oof(model_name: str, n_folds: int = 5) -> np.ndarray:
    """Loads and concatenates saved OOF predictions."""
    all_preds = []
    for fold in range(n_folds):
        path = f"{DIRS['oof']}/{model_name}_fold{fold}.npy"
        if os.path.exists(path):
            all_preds.append(np.load(path))
            print(f"✓ Loaded: fold {fold}")
        else:
            print(f"⚠ Not found: fold {fold} — training required.")
    return np.concatenate(all_preds) if all_preds else None

def save_results(metrics: dict, filename: str = "results.json"):
    """Saves evaluation metrics to Drive as JSON."""
    path = f"{DIRS['results']}/{filename}"
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"✓ Results saved: {path}")

# ============================================================
# CELL 7 — Setup summary
# ============================================================
print("=" * 50)
print("SETUP COMPLETE")
print("=" * 50)
print(f"Drive folder  : {DRIVE_BASE}")
print(f"GitHub repo   : {REPO_PATH}")
print()
print("Available functions:")
print("  push_to_github('message')        → push code to GitHub")
print("  save_code_to_repo('file.py')     → copy .py file to repository")
print("  save_oof(preds, 'roberta', 0)    → save OOF predictions")
print("  load_oof('roberta')              → load OOF predictions")
print("  save_results(metrics)            → save evaluation results")

# ============================================================
# TYPICAL WORKFLOW
# ============================================================
# 1. Run this file (mount Drive, prepare repository)
# 2. Work in other cells / notebooks
# 3. After each important step:
#      save_code_to_repo("/content/01_data_preprocessing.py")
#      push_to_github("data preprocessing complete")
# 4. After each OOF fold:
#      save_oof(preds, "roberta", fold_number)
# 5. If Colab disconnects:
#      load_oof("roberta") to resume from last saved fold
