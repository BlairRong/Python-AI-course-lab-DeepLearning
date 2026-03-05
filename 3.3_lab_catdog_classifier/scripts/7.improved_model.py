"""
Improve the Model
improved_model.py - Improved model with data augmentation and transfer learning
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from tensorflow.keras.applications import VGG16
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# Configuration
IMG_HEIGHT = 150
IMG_WIDTH = 150
BATCH_SIZE = 32
EPOCHS = 30
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'processed')

train_dir = os.path.join(DATA_DIR, 'train')
validation_dir = os.path.join(DATA_DIR, 'validation')

print("IMPROVED CAT VS DOG CLASSIFIER")


# ============================================
# IMPROVEMENT 1: ENHANCED DATA AUGMENTATION 增强型数据增强
# ============================================
print("\n[1/4] Setting up enhanced data augmentation...")

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,           # Increased rotation 增加旋转
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.3,              # Increased zoom range
    horizontal_flip=True,
    brightness_range=[0.8, 1.2], # New: vary brightness 新增：可调节亮度
    fill_mode='nearest'
)

validation_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=True
)

validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False
)





# ============================================
# IMPROVEMENT 2: TWO ARCHITECTURE OPTIONS 两种建筑方案
# ============================================
print("\n[2/4] Choose architecture:")
print("1. Custom CNN (from scratch)")
print("2. Transfer Learning with VGG16")

choice = input("\nEnter your choice (1 or 2): ")

if choice == '2':
    print("\nBuilding Transfer Learning model with VGG16...")
    
    # Load pre-trained VGG16 without top layers
    base_model = VGG16(
        weights='imagenet',
        include_top=False,
        input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)
    )
    
    # Freeze base model layers
    base_model.trainable = False
    
    # Build new classifier on top
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.5),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(1, activation='sigmoid')
    ])
    
    print("VGG16 base frozen. Training only new classifier layers.")
    
else:
    print("\nBuilding improved custom CNN...")
    
    model = models.Sequential([
        # First convolutional block
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
        layers.BatchNormalization(),  # New: batch normalization
        layers.MaxPooling2D(2, 2),
        
        # Second convolutional block
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, 2),
        
        # Third convolutional block
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, 2),
        
        # Fourth convolutional block
        layers.Conv2D(256, (3, 3), activation='relu'),  # Increased filters
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, 2),
        
        # Classifier
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(1, activation='sigmoid')
    ])

model.summary()





# ============================================
# IMPROVEMENT 3: ADVANCED OPTIMIZATION 高级优化
# ============================================
print("\n[3/4] Compiling with advanced optimizer...") #使用高级优化器编译

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),  # Lower learning rate
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
)




# ============================================
# IMPROVEMENT 4: SMART CALLBACKS 智能回调
# ============================================
print("\n[4/4] Setting up advanced callbacks...")

callbacks = [
    EarlyStopping(
        monitor='val_loss',
        patience=7,
        restore_best_weights=True,
        verbose=1
    ),
    ModelCheckpoint(
        filepath=os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'improved_model.h5'),
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-7,
        verbose=1
    )
]






# ============================================
# TRAINING
# ============================================
print("\nStarting improved model training...")

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
# RESULTS COMPARISON
# ============================================
print("RESULTS COMPARISON")

# Load original model if exists
original_model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'best_model.h5')
if os.path.exists(original_model_path):
    original_model = tf.keras.models.load_model(original_model_path)
    
    # Evaluate original model
    validation_generator.reset()
    original_loss, original_acc = original_model.evaluate(validation_generator, verbose=0)[:2]
    
    # Evaluate improved model
    validation_generator.reset()
    improved_loss, improved_acc = model.evaluate(validation_generator, verbose=0)[:2]
    
    print("\nPerformance Comparison:")
    print(f"Original Model - Accuracy: {original_acc:.4f}, Loss: {original_loss:.4f}")
    print(f"Improved Model - Accuracy: {improved_acc:.4f}, Loss: {improved_loss:.4f}")
    print(f"Improvement: +{(improved_acc - original_acc)*100:.2f}% accuracy")
    
    # Plot comparison 图表比较
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Load original history if available
    # For simplicity, just show improved model training curves 为简单起见，仅展示改进后的模型训练曲线。
    axes[0].plot(history.history['accuracy'], label='Training Accuracy')
    axes[0].plot(history.history['val_accuracy'], label='Validation Accuracy')
    axes[0].set_title('Improved Model Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True)
    
    axes[1].plot(history.history['loss'], label='Training Loss')
    axes[1].plot(history.history['val_loss'], label='Validation Loss')
    axes[1].set_title('Improved Model Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'improved_training.png'))
    plt.show()
else:
    print("Original model not found. Showing improved model results only.")

# Save improved model
model.save(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'improved_model.h5'))
print("\nImproved model saved to 'models/improved_model.h5'")


# Run the improved model:
# python scripts/improved_model.py

