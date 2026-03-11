# ⚡ Quick Reference - Cheat Sheet

## 🎯 Quick Answers to Common Questions

---

## Question 1: How Do I Quit the Webcam?

### ✅ FASTEST WAY: Press 'q' Key
```
1. Click on the "YOLO Real-Time Detection" window
2. Press the 'q' key on your keyboard
3. The window closes → You're done!
```

### Alternative Escape Methods:
```
Method A: Press Ctrl+C (or Cmd+C on Mac) in the terminal
Method B: Click Kernel → Interrupt Kernel (in Jupyter menu)
Method C: Close the window and restart kernel
```

**Remember**: The webcam line is COMMENTED OUT by default
```python
# run_detector_on_video(model, video_source=0)  # Commented out
```

---

## Question 2: Why Is the Webcam Running Automatically?

**Answer**: You must have uncommented the line yourself. 

**To fix**: Comment it out again:
```python
# run_detector_on_video(model, video_source=0, confidence_threshold=0.5)
#     ↑ Add # at the beginning to comment out
```

Or in Jupyter: Select the line and press **Ctrl+/**

---

## Question 3: How Do I Run the Webcam Now?

### When You're Ready:
```python
# Uncomment the line first:
run_detector_on_video(model, video_source=0, confidence_threshold=0.5)

# Then press Shift+Enter to run the cell
# To exit: Press 'q' in the detection window
```

---

## Question 4: How Do I Upload to GitHub?

### 🚀 Super Quick Version (5 Steps):

```bash
# Step 1: Go to your project folder
cd /Users/ron/Desktop/Deeplearning/3.9_lab_yolo_object_detection

# Step 2: Initialize git
git init

# Step 3: Configure git (first time only)
git config --global user.name "Your Name"
git config --global user.email "your@email.com"

# Step 4: Add files and commit
git add .
git commit -m "Initial commit: YOLO object detection project"

# Step 5: Push to GitHub
git remote add origin https://github.com/YOUR_USERNAME/REPO_NAME.git
git branch -M main
git push -u origin main
```

### 📋 Detailed Version:
See **GITHUB_UPLOAD.md** in the project folder for complete instructions

---

## Quick Command Quick Sheet

### Git Setup (One Time)
```bash
git config --global user.name "Your Name"
git config --global user.email "you@example.com"
```

### First Upload
```bash
cd /your/project/directory
git init
git add .
git commit -m "Initial commit message"
git remote add origin https://github.com/username/repo.git
git branch -M main
git push -u origin main
```

### Future Uploads (After Making Changes)
```bash
git add .
git commit -m "What you changed"
git push
```

### Check Status
```bash
git status      # What files changed
git log         # View commit history
```

---

## Real-Time Detection Options

### Option 1: Webcam
```python
run_detector_on_video(model, video_source=0, confidence_threshold=0.5)
# Exit: Press 'q'
```

### Option 2: Video File
```python
video_file = '/path/to/your/video.mp4'
output_video = os.path.join(base_dir, 'detection_output.mp4')
run_detector_on_video(model, video_source=video_file, output_path=output_video)
# Exit: Press 'q'
```

### Option 3: Single Image
```python
image_path = os.path.join(base_dir, 'val', 'images', 'your_image.jpg')
detect_on_image(model, image_path)
# Shows image with detections - just close the window
```

---

## File Descriptions

| File | Purpose |
|------|---------|
| `yolo.ipynb` | Main Jupyter notebook with all code |
| `README.md` | Project overview & setup instructions |
| `lab_report.md` | Detailed lab report with results |
| `CONFIGURATION.md` | Troubleshooting & detailed setup |
| `GITHUB_UPLOAD.md` | Step-by-step GitHub upload guide |
| `requirements.txt` | Python dependencies (for `pip install -r requirements.txt`) |
| `setup.py` / `setup.sh` | Automated setup scripts |
| `data.yaml` | YOLO dataset configuration file |
| `.gitignore` | (Create before upload) Files to exclude from GitHub |

---

## Troubleshooting Quick Fixes

### Stuck on Other Issues?

| Issue | Quick Fix |
|-------|-----------|
| **Numpy error** | Restart kernel (Kernel → Restart) |
| **Model not found** | Run training cells first (Cell 6) |
| **Webcam not working** | Check if camera permissions enabled on macOS |
| **Git authentication error** | Use personal access token instead of password |
| **Large file error** | Add to `.gitignore` before pushing |

---

## Next Steps Checklist

- [ ] ✅ Fix notebook (issues resolved above)
- [ ] ✅ Understand how to quit webcam (press 'q')
- [ ] ⬜ Read GITHUB_UPLOAD.md fully
- [ ] ⬜ Create GitHub account (if needed)
- [ ] ⬜ Create GitHub repository
- [ ] ⬜ Configure git locally
- [ ] ⬜ Upload your project (git push)
- [ ] ⬜ Verify it's on GitHub
- [ ] ⬜ Share the link with others!

---

## Still Stuck?

1. **For notebook issues**: Check `CONFIGURATION.md`
2. **For GitHub issues**: Check `GITHUB_UPLOAD.md`
3. **For detection issues**: Check `README.md`
4. **For general questions**: See `lab_report.md`

---

**Remember**: 
- All files are documented ✓
- Your notebook is fixed ✓
- GitHub instructions are ready ✓
- You're good to go! 🚀
