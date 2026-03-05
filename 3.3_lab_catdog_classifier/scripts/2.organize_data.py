"""
organize_data.py - Organize images into cat and dog folders
"""

import os
import shutil
import pathlib

# Define paths
data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
pet_images_dir = os.path.join(data_dir, 'PetImages')

# Check if the extracted folder exists
if not os.path.exists(pet_images_dir):
    print("PetImages folder not found. Make sure the dataset was extracted correctly.")
    exit(1)

# Create organized directories
organized_dir = os.path.join(data_dir, 'organized')
cats_dir = os.path.join(organized_dir, 'cats')
dogs_dir = os.path.join(organized_dir, 'dogs')

os.makedirs(cats_dir, exist_ok=True)
os.makedirs(dogs_dir, exist_ok=True)

# Source directories
source_cats = os.path.join(pet_images_dir, 'Cat')
source_dogs = os.path.join(pet_images_dir, 'Dog')

# Function to copy valid images
def copy_valid_images(source_dir, target_dir, class_name):
    print(f"Processing {class_name} images...")
    
    # Get all files in source directory
    files = list(pathlib.Path(source_dir).iterdir())
    
    valid_count = 0
    invalid_count = 0
    
    for file_path in files:
        if file_path.is_file() and file_path.suffix.lower() == '.jpg':
            # Check if file has content (not zero bytes)
            if file_path.stat().st_size > 0:
                # Copy to target directory
                shutil.copy2(file_path, os.path.join(target_dir, file_path.name))
                valid_count += 1
            else:
                invalid_count += 1
                print(f"  Skipping zero-byte file: {file_path.name}")
    
    print(f"  Copied {valid_count} valid {class_name} images")
    print(f"  Skipped {invalid_count} invalid {class_name} images")

# Copy valid images
copy_valid_images(source_cats, cats_dir, 'cat')
copy_valid_images(source_dogs, dogs_dir, 'dog')

print(f"\nData organized successfully!")
print(f"Cats: {len(os.listdir(cats_dir))} images")
print(f"Dogs: {len(os.listdir(dogs_dir))} images")


#run the script: 
#VS Code - View - Terminal - Newterminal - python scripts/organize_data.py