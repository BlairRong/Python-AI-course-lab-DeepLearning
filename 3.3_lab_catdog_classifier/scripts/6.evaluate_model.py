"""
Evaluate Model and Inspect Predictions 评估模型并检查预测结果
evaluate_model.py - Evaluate model and inspect predictions
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
import random

# Configuration 配置
IMG_HEIGHT = 150
IMG_WIDTH = 150
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'processed')
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'best_model.h5')

validation_dir = os.path.join(DATA_DIR, 'validation')

# Load the trained model 加载已训练的模型
print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded successfully!")

# Prepare validation data generator 准备验证数据生成器
validation_datagen = ImageDataGenerator(rescale=1./255)
validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=32,
    class_mode='binary',
    shuffle=False  # Keep order for consistent evaluation
)

# Get class indices 获取类索引
class_indices = validation_generator.class_indices
print(f"Class mapping: {class_indices}")


# ============================================
# OVERALL EVALUATION 总体评价
# ============================================
print("MODEL EVALUATION")

# Reset generator 重置生成器
validation_generator.reset()

# Evaluate 评估
loss, accuracy = model.evaluate(validation_generator, verbose=1)
print(f"\nValidation Loss: {loss:.4f}")
print(f"Validation Accuracy: {accuracy:.4f}")




# ============================================
# INSPECT PREDICTIONS 检查预测结果
# ============================================
print("INSPECTING INDIVIDUAL PREDICTIONS")

def predict_single_image(image_path):
    """Predict class for a single image"""
    img = load_img(image_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
    img_array = img_to_array(img)
    img_array = img_array / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension 添加批次纬度
    
    prediction = model.predict(img_array, verbose=0)[0][0]
    predicted_class = "DOG" if prediction > 0.5 else "CAT"
    confidence = prediction if prediction > 0.5 else 1 - prediction
    
    return predicted_class, confidence, img

# Get sample images from validation set 从验证集中获取示例图像
def get_sample_images(class_name, num_samples=5):
    class_dir = os.path.join(validation_dir, class_name)
    images = os.listdir(class_dir)
    return [os.path.join(class_dir, img) for img in random.sample(images, min(num_samples, len(images)))]

# Get correct and incorrect predictions 获取正确和错误的预测
print("\nAnalyzing predictions on validation set...")

# Reset generator to get all predictions 重置生成器以获取所有预测结果
validation_generator.reset()
predictions = model.predict(validation_generator, verbose=0)
predicted_classes = (predictions > 0.5).astype(int).flatten()
true_classes = validation_generator.classes

# Find indices of correct and incorrect predictions 找出正确预测和错误预测的指标
correct_indices = np.where(predicted_classes == true_classes)[0]
incorrect_indices = np.where(predicted_classes != true_classes)[0]

print(f"Total validation samples: {len(true_classes)}")
print(f"Correct predictions: {len(correct_indices)}")
print(f"Incorrect predictions: {len(incorrect_indices)}")

# Get filenames 获取文件名
filenames = validation_generator.filenames

# Display some correct predictions 展示一些正确的预测结果
print("CORRECT PREDICTIONS (Sample)")

fig, axes = plt.subplots(2, 5, figsize=(15, 6))

for i, idx in enumerate(correct_indices[:5]):
    # Get image info
    filename = filenames[idx]
    true_class = "DOG" if true_classes[idx] == 1 else "CAT"
    pred_class = "DOG" if predicted_classes[idx] == 1 else "CAT"
    confidence = predictions[idx][0] if true_classes[idx] == 1 else 1 - predictions[idx][0]
    
    # Load and display image
    img_path = os.path.join(validation_dir, filename)
    img = load_img(img_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
    
    axes[0, i].imshow(img)
    axes[0, i].set_title(f"True: {true_class}\nPred: {pred_class}\nConf: {confidence:.2f}")
    axes[0, i].axis('off')


# Display some incorrect predictions 展示一些错误的预测结果
for i, idx in enumerate(incorrect_indices[:5]):
    # Get image info
    filename = filenames[idx]
    true_class = "DOG" if true_classes[idx] == 1 else "CAT"
    pred_class = "DOG" if predicted_classes[idx] == 1 else "CAT"
    confidence = predictions[idx][0] if predicted_classes[idx] == 1 else 1 - predictions[idx][0]
    
    # Load and display image
    img_path = os.path.join(validation_dir, filename)
    img = load_img(img_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
    
    axes[1, i].imshow(img)
    axes[1, i].set_title(f"True: {true_class}\nPred: {pred_class}\nConf: {confidence:.2f}")
    axes[1, i].axis('off')

plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'predictions_analysis.png'))
plt.show()





# ============================================
# ANALYZE MISCLASSIFICATIONS 分析错误分类
# ============================================
print("MISCLASSIFICATION ANALYSIS")

# Get a few misclassified images for detailed analysis 获取一些分类错误的图像以进行详细分析
misclassified_examples = []
for idx in incorrect_indices[:5]:
    filename = filenames[idx]
    img_path = os.path.join(validation_dir, filename)
    
    # Load original image (not resized) to analyze 加载原始图像（未调整大小）进行分析
    original_img = load_img(img_path)
    
    misclassified_examples.append({
        'filename': filename,
        'true_class': 'DOG' if true_classes[idx] == 1 else 'CAT',
        'pred_class': 'DOG' if predicted_classes[idx] == 1 else 'CAT',
        'confidence': predictions[idx][0] if predicted_classes[idx] == 1 else 1 - predictions[idx][0],
        'image': original_img
    })

print("\nAnalyzing why these images were misclassified:")

for i, example in enumerate(misclassified_examples):
    print(f"\nExample {i+1}: {example['filename']}")
    print(f"  True: {example['true_class']}, Predicted: {example['pred_class']}")
    print(f"  Confidence: {example['confidence']:.2f}")
    







# ============================================
# REFLECTION ON DATASET DIFFICULTY 对数据集难度的反思
# ============================================

"""
Dataset difficulty is driven by multiple factors:

1. Visual Ambiguity:
- Some dog breeds look cat-like (small, pointy ears)
- Some cat breeds look dog-like (large, floppy ears)
- Young animals of both species can look similar

2. Background Noise:
- Cluttered backgrounds distract from the main subject
- Objects that look like animals (stuffed toys, shadows)
- Textures that resemble fur in the background

3. Intra-class Variation:
- Dogs: Chihuahua vs. Great Dane (extremely different)
- Cats: Sphynx vs. Persian (completely different appearance)
- Different poses, angles, and scales

4. Image Quality:
- Low resolution loses important details
- Motion blur makes features indistinct
- Poor lighting hides distinguishing characteristics

These factors combine to make this "simple" binary classification
task surprisingly challenging for machine learning models.




数据集的难度由多种因素造成：

1. 视觉歧义：
- 某些犬种外形酷似猫（耳朵小而尖）
- 某些猫种外形酷似狗（耳朵大而下垂）
- 幼年时期的犬猫幼崽外形相似

2. 背景噪声：
- 杂乱的背景会分散注意力，影响对主要对象的识别
- 看起来像动物的物体（毛绒玩具、阴影）
- 背景中类似毛发的纹理

3. 类内差异：
- 犬类：吉娃娃 vs. 大丹犬（差异极大）
- 猫类：斯芬克斯猫 vs. 波斯猫（外形完全不同）
- 不同的姿势、角度和体型

4. 图像质量：
- 低分辨率会丢失重要细节
- 运动模糊会使特征模糊不清
- 光照不足会掩盖显著特征

这些因素共同作用，使得这个看似简单的二元分类任务, 对机器学习模型而言却异常具有挑战性。
"""


#run the evaluation: python scripts/evaluate_model.py