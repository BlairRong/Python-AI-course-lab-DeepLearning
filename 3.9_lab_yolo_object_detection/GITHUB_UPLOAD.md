# 📤 GitHub Upload Guide - Complete Instructions

This guide will walk you through uploading your YOLO project to GitHub step-by-step.

---

## Step-by-Step Instructions

### Step 1: Create a GitHub Account (if needed)
1. Go to [github.com](https://github.com)
2. Click **"Sign up"**
3. Enter email, password, and username
4. Verify your email address
5. **✓ Done!**

---

### Step 2: Create a New Repository on GitHub

1. Log in to GitHub
2. Click the **"+"** icon in the top right corner
3. Select **"New repository"**
4. Fill in the details:
   - **Repository name**: `YOLO-Object-Detection` (or any name)
   - **Description**: `Real-time object detection for red cups, blue bottles, and phones using YOLOv8`
   - **Visibility**: Choose **"Public"** (to share) or **"Private"** (only you can see)
   - **Initialize with**: Leave unchecked (we'll push existing code)
5. Click **"Create repository"**
6. **✓ Repository created!**

---

### Step 3: Install Git (if needed)

**Check if Git is installed:**
```bash
git --version
```

If not installed:
- **macOS**: `brew install git`
- **Windows**: Download from [git-scm.com](https://git-scm.com/download/win)
- **Linux**: `sudo apt-get install git`

---

### Step 4: Configure Git (First Time Only)

```bash
# Set your name
git config --global user.name "Your Name"

# Set your email (same as GitHub account)
git config --global user.email "your.email@example.com"

# Check configuration
git config --global user.name
git config --global user.email
```

Or for the current project only (without `--global`):
```bash
cd /Users/ron/Desktop/Deeplearning/3.9_lab_yolo_object_detection
git config user.name "Your Name"
git config user.email "your.email@example.com"
```

---

### Step 5: Initialize Git Repository

Navigate to your project directory:
```bash
cd /Users/ron/Desktop/Deeplearning/3.9_lab_yolo_object_detection
```

Initialize git:
```bash
git init
```

**✓ Git is now tracking this directory**

---

### Step 6: Create `.gitignore` File

Before uploading, create a `.gitignore` file to exclude unnecessary files:

```bash
cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
ENV/
.venv

# Jupyter
.ipynb_checkpoints
*.ipynb_checkpoints

# IDE
.vscode/
.idea/
*.swp
*.swo

# Large files
*.pt          # Model weights
*.pth
runs/         # Training outputs (OPTIONAL - remove if you want to upload these)
*.mp4         # Videos
*.avi

# OS
.DS_Store
Thumbs.db

# Optional: Exclude large directories
downloaded_images/
EOF
```

**View .gitignore:**
```bash
cat .gitignore
```

---

### Step 7: Add Files to Git

Add all project files:
```bash
git add .
```

Or add specific files:
```bash
git add README.md lab_report.md CONFIGURATION.md yolo.ipynb requirements.txt
```

Check what will be uploaded:
```bash
git status
```

---

### Step 8: Create First Commit

```bash
git commit -m "Initial commit: YOLO object detection project with training code and documentation"
```

**Commit message tips:**
- Use clear, descriptive messages
- Explain WHAT changed and WHY
- Good examples:
  - `"Add training notebook with YOLOv8n model"`
  - `"Add lab report and README documentation"`
  - `"Fix numpy import issue in training cell"`

---

### Step 9: Add Remote Repository

Connect your local repo to GitHub (replace URL):

```bash
git remote add origin https://github.com/YOUR_USERNAME/YOLO-Object-Detection.git
```

**Example:**
```bash
git remote add origin https://github.com/BlairRong/YOLO-Object-Detection.git
```

Verify:
```bash
git remote -v
```

---

### Step 10: Rename Main Branch (if needed)

GitHub uses `main` as default, but Git uses `master`. Rename if needed:

```bash
git branch -M main
```

---

### Step 11: Push to GitHub

**Push your code:**
```bash
git push -u origin main
```

When prompted:
- **Username**: Your GitHub username
- **Password**: Your GitHub **personal access token** (not password)

### 🔐 Create Personal Access Token (Recommended)

Instead of password, GitHub requires a token:

1. Go to GitHub → Settings → Developer settings → **Personal access tokens**
2. Click **"Generate new token"**
3. Select scopes: Check **"repo"**
4. Click **"Generate token"**
5. **Copy the token** (you won't see it again!)
6. Use this token as password when pushing

**First push:**
```bash
git push -u origin main
```

**Future pushes:**
```bash
git push
```

---

### Step 12: Verify Upload

1. Go to your GitHub repository page
2. Refresh the page
3. You should see all your files!
4. **✓ Upload complete!**

---

## Common Git Commands

### View Status
```bash
git status
```

### View Commits
```bash
git log --oneline
```

### Make Changes and Update

After making changes:
```bash
# 1. Check what changed
git status

# 2. Add changes
git add .

# 3. Commit with message
git commit -m "Description of changes"

# 4. Push to GitHub
git push
```

### View Recent Commits
```bash
git log --oneline -10
```

### Revert Last Commit
```bash
# Before pushing:
git reset --soft HEAD~1

# Or create a new commit that undoes changes:
git revert HEAD
git push
```

---

## Project Structure on GitHub

After uploading, your repository will look like:
```
YOLO-Object-Detection/
├── 📄 README.md                    ← Project overview
├── 📊 lab_report.md               ← Detailed lab report
├── ⚙️ CONFIGURATION.md            ← Setup & troubleshooting
├── 📋 GITHUB_UPLOAD.md            ← This file!
├── 📦 requirements.txt            ← Dependencies
├── 🐍 setup.py                    ← Setup script
├── 🔧 setup.sh                    ← Bash setup
├── 📔 yolo.ipynb                  ← Main notebook
├── 📝 data.yaml                   ← Dataset config
│
├── 📁 images/                     ← Original images
├── 📁 labels/                     ← Original labels
├── 📁 train/                      ← Training data
├── 📁 val/                        ← Validation data
│
└── .gitignore                     ← Files to exclude
```

---

## What Gets Uploaded / Ignored

### ✅ WILL BE UPLOADED
- `.py`, `.ipynb` files (code)
- `.md` files (documentation)
- `.txt` files (requirements, labels)
- `.yaml` files (config)
- `.sh`, Python scripts

### ❌ WON'T BE UPLOADED (via .gitignore)
- `__pycache__/` (Python cache)
- `.ipynb_checkpoints/` (Jupyter temp files)
- `*.pt`, `*.pth` (model weights - too large)
- `runs/` (training outputs - optional)
- `.DS_Store` (macOS files)
- `.venv/` (virtual environment)

**To include large files (optional):**
- Remove from `.gitignore`
- Or use [Git LFS](https://git-lfs.github.com/) for large files

---

## Troubleshooting

### Issue: "fatal: not a git repository"
```bash
# Solution: Initialize git in project directory
cd /Users/ron/Desktop/Deeplearning/3.9_lab_yolo_object_detection
git init
```

### Issue: "Authentication failed"
```bash
# Solution 1: Use personal access token instead of password
# (See Step 11: Create Personal Access Token)

# Solution 2: Clear cached credentials and retry
git credential-osxkeychain erase  # on macOS
git push  # Will prompt for credentials again
```

### Issue: "remote already exists"
```bash
# Solution: Remove and re-add remote
git remote remove origin
git remote add origin https://github.com/YOUR_USERNAME/REPO_NAME.git
```

### Issue: Large files error
```bash
# Remove from git tracking:
git rm --cached *.pt
git rm --cached runs/

# Add to .gitignore
echo "*.pt" >> .gitignore
echo "runs/" >> .gitignore

# Commit changes
git add .
git commit -m "Ignore large files"
git push
```

### Issue: Wrong branch name
```bash
# Check current branch
git branch

# Rename branch
git branch -M main

# Push to correct branch
git push -u origin main
```

---

## Next Steps After Upload

### 1. Add README Badge (Optional)
In GitHub, go to repository → About section → Edit
- Add description
- Add topics: `yolo`, `object-detection`, `deep-learning`
- Add website (if you have one)

### 2. Enable GitHub Pages (Optional - for documentation)
Settings → Pages → Choose `main` branch → Save
Your README will be available at: `https://YOUR_USERNAME.github.io/REPO_NAME/`

### 3. Share Your Project
```
Direct link: https://github.com/YOUR_USERNAME/YOLO-Object-Detection
Share on social media, CV, portfolio, etc.
```

### 4. Collaborate
- Invite team members: Settings → Collaborators
- Create pull requests for changes
- Use issues for bug tracking

---

## Git Workflow Summary

**One-time setup:**
```bash
git config --global user.name "Your Name"
git config --global user.email "your@email.com"
cd /path/to/project
git init
git remote add origin https://github.com/USERNAME/REPO.git
git branch -M main
```

**First upload:**
```bash
git add .
git commit -m "Initial commit: YOLO object detection project"
git push -u origin main
```

**Future uploads:**
```bash
git add .
git commit -m "Description of your changes"
git push
```

---

## Useful Resources

- **Git Documentation**: https://git-scm.com/doc
- **GitHub Documentation**: https://docs.github.com
- **GitHub Desktop** (GUI alternative): https://desktop.github.com/
- **VS Code Git Extension**: Built-in, no installation needed
- **Interactive Git Tutorial**: https://learngitbranching.js.org/

---

## Quick Command Reference

```bash
# Setup
git config --global user.name "Name"
git config --global user.email "email@example.com"
git init
git remote add origin https://github.com/user/repo.git
git branch -M main

# First time
git add .
git commit -m "Initial commit"
git push -u origin main

# Updates
git status                    # Check what changed
git add .                     # Stage all changes
git commit -m "Message"       # Commit with message
git push                      # Push to GitHub
git log --oneline -5          # View last 5 commits
git pull                      # Get latest from GitHub
```

---

**Congratulations! 🎉 Your project is now on GitHub!**

Share the link with others and keep updating as your project evolves.

Contact: BlairRong/Python-AI-course-lab-DeepLearning
