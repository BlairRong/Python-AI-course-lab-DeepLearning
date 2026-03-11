#!/usr/bin/env python3
"""
Environment setup script for YOLO Object Detection project
Automatically installs dependencies and verifies setup
"""

import subprocess
import sys
import os

def run_command(cmd, description=""):
    """Run a shell command and return status"""
    try:
        if description:
            print(f"✓ {description}...")
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            return True, result.stdout
        else:
            return False, result.stderr
    except Exception as e:
        return False, str(e)

def main():
    print("=" * 60)
    print("YOLO Object Detection Environment Setup")
    print("=" * 60)
    print()
    
    # Check Python version
    print(f"Python version: {sys.version}")
    print()
    
    # Check if requirements.txt exists
    if not os.path.exists("requirements.txt"):
        print("❌ requirements.txt not found!")
        print("Please ensure you're in the project directory")
        sys.exit(1)
    
    # Install dependencies
    print("Installing dependencies from requirements.txt...")
    print("-" * 60)
    
    success, output = run_command(
        f"{sys.executable} -m pip install -r requirements.txt",
        "Installing packages"
    )
    
    if success:
        print("✓ Dependencies installed successfully!")
    else:
        print("❌ Failed to install dependencies")
        print(output)
        sys.exit(1)
    
    print()
    print("=" * 60)
    print("Verification")
    print("=" * 60)
    
    # Verify key packages
    packages = [
        ("ultralytics", "YOLO Framework"),
        ("cv2", "OpenCV"),
        ("torch", "PyTorch"),
        ("numpy", "NumPy"),
        ("sklearn", "Scikit-learn"),
    ]
    
    all_ok = True
    for import_name, display_name in packages:
        try:
            __import__(import_name)
            print(f"✓ {display_name} is installed")
        except ImportError:
            print(f"❌ {display_name} is NOT installed")
            all_ok = False
    
    print()
    print("=" * 60)
    print("Directory Structure Check")
    print("=" * 60)
    
    required_dirs = ["images", "labels"]
    for dir_name in required_dirs:
        if os.path.exists(dir_name):
            file_count = len(os.listdir(dir_name))
            print(f"✓ {dir_name}/ exists ({file_count} files)")
        else:
            print(f"⚠ {dir_name}/ not found (will be created during training)")
    
    print()
    if all_ok:
        print("=" * 60)
        print("✅ Setup Complete! Ready to use.")
        print("=" * 60)
        print()
        print("Next steps:")
        print("1. Run 'jupyter notebook yolo.ipynb'")
        print("2. Execute cells in order (Cell 1 → Cell 6+)")
        print("3. For real-time detection: run_detector_on_video(model, video_source=0)")
        print()
    else:
        print("=" * 60)
        print("⚠ Some packages are missing. Please check the errors above.")
        print("=" * 60)
        sys.exit(1)

if __name__ == "__main__":
    main()
