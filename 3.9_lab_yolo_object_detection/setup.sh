#!/bin/bash
# Setup script for YOLO Object Detection Project
# Run this script to set up your environment

echo "=================================================="
echo "YOLO Object Detection Environment Setup"
echo "=================================================="
echo ""

# Check Python version
echo "✓ Checking Python version..."
python --version

# Create conda environment (optional)
echo ""
echo "✓ Setting up environment..."
echo ""
echo "Two options:"
echo "1. Using pip (quick):"
echo "   pip install -r requirements.txt"
echo ""
echo "2. Using conda (recommended):"
echo "   conda create -n yolo python=3.9"
echo "   conda activate yolo"
echo "   pip install -r requirements.txt"
echo ""

# Install requirements
echo "Installing dependencies..."
pip install -r requirements.txt

echo ""
echo "=================================================="
echo "Environment setup complete!"
echo "=================================================="
echo ""
echo "Next steps:"
echo "1. Verify dataset exists in images/ and labels/ directories"
echo "2. Run: jupyter notebook yolo.ipynb"
echo "3. Execute cells in order:"
echo "   - Cell 1: Set working directory"
echo "   - Cell 2: Check dataset"
echo "   - Cell 3: Split train/val"
echo "   - Cell 4: Create config file"
echo "   - Cell 5: Train model"
echo "   - Cells 6+: Evaluation and inference"
echo ""
echo "For real-time detection:"
echo "   run_detector_on_video(model, video_source=0)"
echo ""
