"""
Data Inspection

inspect_data.py - Inspect dataset characteristics
"""

import os
import matplotlib.pyplot as plt
from PIL import Image
import random

# Define paths
data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'organized')
cats_dir = os.path.join(data_dir, 'cats')
dogs_dir = os.path.join(data_dir, 'dogs')

# Get list of images
cat_images = os.listdir(cats_dir)[:5]  # First 5 cat images
dog_images = os.listdir(dogs_dir)[:5]  # First 5 dog images

# Function to display image information
def inspect_images(image_list, class_name, source_dir):
    print(f"\n--- {class_name} Images ---")
    for img_name in image_list:
        img_path = os.path.join(source_dir, img_name)
        with Image.open(img_path) as img:
            print(f"File: {img_name}")
            print(f"  Format: {img.format}")
            print(f"  Size: {img.size} (width x height)")
            print(f"  Mode: {img.mode}")
            
            # Check background complexity (simple heuristic - image entropy would be better)
            print(f"  Background: {'Complex' if img.size[0] > 300 else 'Simple'}")

# Inspect cats
inspect_images(cat_images, 'CAT', cats_dir)

# Inspect dogs
inspect_images(dog_images, 'DOG', dogs_dir)

# Display sample images
def display_samples(cat_images, dog_images):
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    
    for i, img_name in enumerate(cat_images[:5]):
        img_path = os.path.join(cats_dir, img_name)
        img = Image.open(img_path)
        axes[0, i].imshow(img)
        axes[0, i].set_title(f'Cat {i+1}')
        axes[0, i].axis('off')
    
    for i, img_name in enumerate(dog_images[:5]):
        img_path = os.path.join(dogs_dir, img_name)
        img = Image.open(img_path)
        axes[1, i].imshow(img)
        axes[1, i].set_title(f'Dog {i+1}')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'sample_images.png'))
    plt.show()

display_samples(cat_images, dog_images)
print("\nSample images saved to data/sample_images.png")

