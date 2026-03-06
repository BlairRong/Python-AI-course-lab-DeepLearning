"""
Build and Train CNN Model (Convolutional Neural Network)
train_model.py - Build and train CNN for cat vs dog classification
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Set random seeds for reproducibility 设置随机种子以确保可复现性
tf.random.set_seed(42)
np.random.seed(42)


# ============================================
# CONFIGURATION 配置 （可更改为200*200）
# ============================================
IMG_HEIGHT = 150
IMG_WIDTH = 150
BATCH_SIZE = 32
EPOCHS = 20
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'processed')

train_dir = os.path.join(DATA_DIR, 'train')
validation_dir = os.path.join(DATA_DIR, 'validation')

print("CAT VS DOG CLASSIFIER")
print(f"Image size: {IMG_HEIGHT} x {IMG_WIDTH}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Training epochs: {EPOCHS}")
print(f"Training data: {train_dir}")
print(f"Validation data: {validation_dir}")

# ============================================
# DATA PREPROCESSING & AUGMENTATION 数据预处理与增强(pixel normalization & resize)
# ============================================
print("\n[1/5] Preparing data generators...")

# Training data generator生成器 (with augmentation增强 for later use)
train_datagen = ImageDataGenerator(
    rescale=1./255,  # Normalize pixel values to [0,1]
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Validation data generator (only rescaling, no augmentation仅进行缩放不进行增强)
validation_datagen = ImageDataGenerator(rescale=1./255)

# Create generators that flow images from directories 创建生成器，从目录中导入图像
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='binary',  # binary classification
    shuffle=True
)

validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False
)

print(f"Class mapping: {train_generator.class_indices}")










# ============================================
# CNN ARCHITECTURE DESIGN 卷积神经网络算法建筑设计
# ============================================
print("\n[2/5] Building CNN architecture...")

model = models.Sequential([
    # First convolutional block 第一个卷积块
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    layers.MaxPooling2D(2, 2),
    
    # Second convolutional block - more filters 第二个卷积块 - 更多滤波器
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    
    # Third convolutional block - even more filters 第三个卷积块 - 更多滤波器
    layers.Conv2D(128, (3, 3), activation='relu'), #128可更改为64
    layers.MaxPooling2D(2, 2),
    
    # Fourth convolutional block 第四个卷积块 
    layers.Conv2D(128, (3, 3), activation='relu'), #128可更改为64
    layers.MaxPooling2D(2, 2),
    
    # Transition to classifier # 过渡到分类器
    layers.Flatten(),
    layers.Dropout(0.5),  # Regularization: randomly drop 50% of neurons 正则化：随机丢弃 50% 的神经元
    
    # Fully connected layer 全连接层
    layers.Dense(512, activation='relu'),
    
    # Output layer - single neuron with sigmoid for binary classification用于二元分类的单个神经元，采用 sigmoid 函数
    layers.Dense(1, activation='sigmoid')
])

# Display model architecture
model.summary()




# ============================================
# COMPILE MODEL 编译模型
# ============================================
print("\n[3/5] Compiling model...")

model.compile(
    optimizer='adam',  # Adaptive learning rate optimizer 自适应学习率优化器
    loss='binary_crossentropy',  # Suitable for binary classification 适用于二元分类（0-1）    
    metrics=['accuracy']
)

print("Optimizer: Adam")
print("Loss function: Binary Crossentropy")
print("Metrics: Accuracy")

# ============================================
# TRAINING 训练多个epoch
# ============================================
print("\n[4/5] Training model...")

# Callbacks for better training 回调以进行更好的培训
callbacks = [
    EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),
    ModelCheckpoint(
        filepath=os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'best_model.h5'),
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
]

# Create models directory 创建模型目录
os.makedirs(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models'), exist_ok=True)

# Train the model 训练模型
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // BATCH_SIZE,
    callbacks=callbacks,
    verbose=1
)


# ============================================
# EVALUATION & VISUALIZATION 评估与可视化
# ============================================
print("\n[5/5] Evaluating and visualizing results...")

# Plot training history
def plot_training_history(history):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot accuracy
    axes[0].plot(history.history['accuracy'], label='Training Accuracy')
    axes[0].plot(history.history['val_accuracy'], label='Validation Accuracy')
    axes[0].set_title('Model Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True)
    
    # Plot loss
    axes[1].plot(history.history['loss'], label='Training Loss')
    axes[1].plot(history.history['val_loss'], label='Validation Loss')
    axes[1].set_title('Model Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'training_history.png'))
    plt.show()
    
    print("\nTraining history saved to 'training_history.png'")

plot_training_history(history)

# Final evaluation 最终评估
val_loss, val_acc = model.evaluate(validation_generator, verbose=0)
print(f"\nFinal Validation Accuracy: {val_acc:.4f}")
print(f"Final Validation Loss: {val_loss:.4f}")

# ============================================
# ANALYSIS: UNDERFITTING vs OVERFITTING 分析：欠拟合与过拟合
# ============================================

print("MODEL ANALYSIS")

# Calculate final metrics 计算最终指标
train_acc = history.history['accuracy'][-1]
val_acc = history.history['val_accuracy'][-1]
train_loss = history.history['loss'][-1]
val_loss = history.history['val_loss'][-1]

print(f"Final Training Accuracy: {train_acc:.4f}")
print(f"Final Validation Accuracy: {val_acc:.4f}")
print(f"Final Training Loss: {train_loss:.4f}")
print(f"Final Validation Loss: {val_loss:.4f}")

# Check for underfitting
if train_acc < 0.7:
    print("\n⚠️  Possible UNDERFITTING detected:")
    print("   - Training accuracy is low (<70%)")
    print("   - Model may need more capacity (more layers/filters)")
    print("   - Consider training for more epochs")
elif val_acc < 0.7:
    print("\n⚠️  Possible ISSUE detected:")
    print("   - Validation accuracy is low despite good training accuracy")
    print("   - Check for data leakage or mismatch between train/val distributions")

# Check for overfitting
if train_acc - val_acc > 0.15:
    print("\n⚠️  Possible OVERFITTING detected:")
    print(f"   - Gap between training and validation accuracy: {train_acc - val_acc:.4f}")
    print("   - Model is memorizing training data rather than generalizing")
    print("   - Solutions: Increase dropout, add more regularization, use data augmentation")
elif train_acc - val_acc > 0.05:
    print("\n📊 Mild overfitting detected:")
    print(f"   - Gap: {train_acc - val_acc:.4f}")
    print("   - Model is generalizing reasonably well")
else:
    print("\n✅ Good generalization:")
    print("   - Training and validation accuracies are close")
    print("   - Model is learning well without overfitting")

# Save the final model
model.save(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'final_model.h5'))
print("\nModel saved to 'models/final_model.h5'")



