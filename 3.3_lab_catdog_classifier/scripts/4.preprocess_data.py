"""
Data Preprocessing 数据预处理
preprocess_data.py - Preprocess images and create train/validation splits
"""

import os
import shutil
import random
import math
from PIL import Image
import numpy as np

# Define paths
base_dir = os.path.dirname(os.path.dirname(__file__))
organized_dir = os.path.join(base_dir, 'data', 'organized')
processed_dir = os.path.join(base_dir, 'data', 'processed')

# Create processed directory structure
train_dir = os.path.join(processed_dir, 'train')
val_dir = os.path.join(processed_dir, 'validation')

train_cats = os.path.join(train_dir, 'cats')
train_dogs = os.path.join(train_dir, 'dogs')
val_cats = os.path.join(val_dir, 'cats')
val_dogs = os.path.join(val_dir, 'dogs')

# Create all directories
for dir_path in [train_cats, train_dogs, val_cats, val_dogs]:
    os.makedirs(dir_path, exist_ok=True)

# Function to split and copy images
def split_and_copy(source_dir, train_target, val_target, split_ratio=0.8):
    """
    Split images into training and validation sets
    
    Args:
        source_dir: Source directory containing images
        train_target: Target directory for training images
        val_target: Target directory for validation images
        split_ratio: Proportion of images for training (default 0.8)
    """
    # Get all image files
    images = [f for f in os.listdir(source_dir) 
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    # Shuffle images randomly
    random.shuffle(images)
    
    # Calculate split point
    split_point = int(len(images) * split_ratio)
    
    # Split images
    train_images = images[:split_point]
    val_images = images[split_point:]
    
    # Copy training images
    for img in train_images:
        src = os.path.join(source_dir, img)
        dst = os.path.join(train_target, img)
        shutil.copy2(src, dst)
    
    # Copy validation images
    for img in val_images:
        src = os.path.join(source_dir, img)
        dst = os.path.join(val_target, img)
        shutil.copy2(src, dst)
    
    print(f"  {len(train_images)} training, {len(val_images)} validation")

print("Splitting dataset into training (80%) and validation (20%)...")
print("Cats:")
split_and_copy(os.path.join(organized_dir, 'cats'), train_cats, val_cats)
print("Dogs:")
split_and_copy(os.path.join(organized_dir, 'dogs'), train_dogs, val_dogs)

# Function to verify images can be loaded and resized
def verify_images(directory, target_size=(150, 150)):
    """Verify that all images can be loaded and resized"""
    problematic = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(root, file)
                try:
                    with Image.open(img_path) as img:
                        # Try to resize to target size
                        img = img.resize(target_size)
                        # Convert to numpy array to verify
                        img_array = np.array(img)
                except Exception as e:
                    problematic.append((img_path, str(e)))
    
    return problematic

print("\nVerifying images can be loaded and resized...")
problematic_images = verify_images(processed_dir)

if problematic_images:
    print(f"Found {len(problematic_images)} problematic images:")
    for img_path, error in problematic_images[:5]:  # Show first 5
        print(f"  {img_path}: {error}")
else:
    print("All images verified successfully!")

print(f"\nDataset prepared successfully!")
print(f"Training cats: {len(os.listdir(train_cats))}")
print(f"Training dogs: {len(os.listdir(train_dogs))}")
print(f"Validation cats: {len(os.listdir(val_cats))}")
print(f"Validation dogs: {len(os.listdir(val_dogs))}")





"""
Data Preprocessing

Why Normalization?
Normalization scales pixel values from the original range [0, 255] to [0, 1] or [-1, 1]. This is necessary because:

- Neural networks train faster when inputs are on a similar scale
- Prevents large pixel values from dominating the gradient updates
- Improves numerical stability during training

Why Train/Validation Split?
Splitting data into training and validation sets (80/20 ratio) is essential because:

- **Training set**: Used to update model weights through gradient descent
- **Validation set**: Used to monitor how well the model generalizes to unseen data
- This helps detect overfitting - when validation accuracy stops improving while training accuracy continues to rise
"""


"""
在深度学习项目中先分类数据不调整尺寸的做法非常常见:
将原始图像从 organized/cats 和 organized/dogs 中按 80/20 比例复制到 processed/train 和 processed/validation 目录中。
在这个过程中，没有对图像进行 resize 或归一化，只是简单复制文件。
然后，在训练脚本 train_model.py 中，通过 ImageDataGenerator 的 rescale=1./255 和 target_size=(150,150) 在加载图像时动态地进行了 resize 和归一化。
也就是说，物理磁盘上的图像保留了原始尺寸和像素值范围，仅在内存中处理时被调整和归一化。

因为：
# 节省存储空间：如果每个图像都保存为调整大小后的版本，会占用额外磁盘空间（尤其是多份副本）。
# 灵活性高：训练时可以随时更改目标尺寸，而不必重新生成整个数据集。
# 数据增强：如果后续做数据增强（旋转、缩放等），也必须在加载时实时应用，提前固定尺寸反而限制增强效果。

老师的要求可以理解成在送入模型前确保图像尺寸一致且归一化
"""