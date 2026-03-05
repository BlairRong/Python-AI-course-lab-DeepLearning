# Lab Report: Cat vs Dog Classification

## 1. Data Inspection Observations

After inspecting the dataset, I observed several factors that make this classification task challenging despite having only two classes:

- **Resolution Differences**: Images vary significantly in size – some are high-resolution close-ups while others are low-resolution or distant shots.

- **Background Complexity**: Backgrounds range from simple solid colors to complex outdoor scenes with vegetation, furniture, or other objects that can confuse the model.

- **Lighting Conditions**: Images have varying lighting – some are well-lit studio shots, others are dark indoor photos or have harsh outdoor lighting creating shadows.

- **Breed and Pose Variation**: Dogs and cats come in vastly different shapes, sizes, and colors. Animals appear in different poses – sitting, standing, running, sleeping – making it hard to learn consistent features.

- **Partial Occlusion**: Some images show only parts of animals (face only, body partially hidden) rather than complete animals.

These variations mean the model must learn to recognize the essential features of cats and dogs while ignoring irrelevant variations in background, lighting, and pose.

## 2. Data Preprocessing: Normalization & Train/Validation Split

### Why Normalization?

- Normalization scales pixel values from the original range [0, 255] to [0, 1] or [-1, 1]. This is necessary because:

- Neural networks train faster when inputs are on a similar scale.

- It prevents large pixel values from dominating the gradient updates.

- It improves numerical stability during training.

### Why Train/Validation Split?

Splitting data into training and validation sets (80/20 ratio) is essential because:

- **Training set**: Used to update model weights through gradient descent.

- **Validation set**: Used to monitor how well the model generalizes to unseen data.

This helps detect overfitting – when validation accuracy stops improving while training accuracy continues to rise.

In this lab, images were resized to 150×150 pixels and normalized on-the-fly using ImageDataGenerator(rescale=1./255). The dataset was split into 80% training (19,998 images) and 20% validation (5,000 images).

## 3. Architecture Explanation

The CNN model was built using Keras Sequential API with the following layers:

**1.Convolutional layers: Learn spatial features from images.**

- First layer (32 filters, 3×3): Detects edges, colors, simple textures.

- Second layer (64 filters, 3×3): Combines edges into shapes (ears, eyes).

- Third layer (128 filters, 3×3): Detects object parts (face, body).

- Fourth layer (128 filters, 3×3): Learns complete object representations.

**2.MaxPooling layers (2×2): Reduce spatial dimensions and add translation invariance.**

**3.Dropout (0.5)**: Randomly disables 50% of neurons to prevent overfitting.

**4.Flatten layer**: Converts 2D feature maps into a 1D vector.

**5.Dense layer (512 units, ReLU)**: Combines features for final decision.

**6.Sigmoid output**: Outputs a probability between 0 (cat) and 1 (dog).

Total trainable parameters: ~1.5 million.

## 4. Model Training and Analysis

### Learning Curves Analysis 分析学习曲线 (see training_history.png)

After training the CNN model for 20 epochs, I observed the following:

- **Training accuracy** started at approximately 50% in the first epoch and gradually increased to 87.29% by the final epoch.

- **Validation accuracy** followed a similar upward trend, reaching 90.20% on the best model (saved at epoch 19) and 88.40% at the final epoch.

- **Training loss** decreased steadily from around 0.63 at epoch 3 to 0.2926 at epoch 20.

- **Validation loss** decreased initially from 0.54 (epoch 3) to a low of 0.2345 at epoch 19, then slightly increased to 0.2682 at epoch 20, suggesting minor fluctuations but no severe degradation.

## Overfitting Assessment 模型是否存在欠拟合和过度拟合的情况

Based on the learning curves:

- The model shows good generalization overall. Training and validation accuracies remain close throughout training, with the final gap being only 1.5 percentage points (87.29% vs 88.40%).

- The validation loss reached its minimum at epoch 19 and then rose slightly, which could hint at the **beginning of overfitting**, but the effect is marginal微小 and the model continues to perform well on unseen data.

- The early stopping callback (patience=5)停止回调函数 was not triggered触发, indicating that validation performance did not deteriorate significantly for five consecutive epochs.

- Conclusion: The model is learning meaningful patterns and is not suffering from underfitting or severe overfitting.

## 5. Misclassification Analysis 错误分类分析(see predictions_analysis.png)

After evaluating the model on the validation set (5,000 images) and inspecting the misclassified examples, I identified these common patterns:

1. **Unusual backgrounds**: Images with complex or distracting backgrounds were often misclassified. For instance, a cat lying on a patterned carpet might be mistaken for a dog.

2. **Extreme poses**: Animals in unusual positions (e.g., upside down, from behind, or partially hidden) confused the model.

3. **Poor lighting**: Very dark or overexposed images reduced classification confidence and led to errors.

4. **Small objects**: Images where the animal was small in the frame or taken from a distance were challenging, as the model had fewer pixels to work with.

5. **Breed ambiguity**: Some dog breeds (e.g., Shih Tzu, Poodle) have features resembling cats (pointed ears, small size), and vice versa, causing confusion.

Interestingly, **all five misclassified examples** shown in the evaluation output were cats predicted as **dogs**, with confidence values ranging from 0.56 to 0.93. This suggests that the model may be slightly biased toward predicting "dog" in ambiguous cases, possibly because the training set contained more dog images with similar backgrounds or poses.

## 6. Reflection on Dataset Difficulty 思考数据集的难度

Dataset difficulty is driven by multiple factors:

### 1.Visual Ambiguity 视觉歧义

- Some dog breeds look cat-like (small, pointy ears).

- Some cat breeds look dog-like (large, floppy ears).

- Young animals of both species can look similar.

### 2.Background Noise

- Cluttered backgrounds distract from the main subject.

- Objects that look like animals (stuffed toys, shadows).

- Textures that resemble fur in the background.

### 3.Intra-class Variation 类内变异

- Dogs: Chihuahua vs. Great Dane (extremely different). 吉娃娃 & 大丹犬

- Cats: Sphynx vs. Persian (completely different appearance). 斯芬克斯猫 vs. 波斯猫

- Different poses, angles, and scales.

### 4.Image Quality

- Low resolution loses important details.

- Motion blur makes features indistinct.

- Poor lighting hides distinguishing characteristics.

These factors combine to make this "simple" binary classification task surprisingly challenging for machine learning models.

## 7. Proposed Improvement and Comparison 改进建议及比较

### Improvement Implemented

I chose to implement transfer learning using the VGG16 architecture pre-trained on ImageNet. The original fully connected top layers were removed, and a new classifier was added consisting of:

- Global average pooling

- Dropout (0.5)

- A dense layer with 512 units and ReLU activation

- Another dropout (0.3)

- A final sigmoid output for binary classification.

The convolutional base of VGG16 was frozen, so only the newly added layers were trained. Data augmentation was also enhanced with additional brightness variations and increased zoom range.

### Results Comparison (see improved_trainning.png)

**Model**                       **Validation Accuracy**                 **Validation Loss**
Baseline (custom CNN)           90.20% (final eval)                      0.2343
Improved (VGG16 transfer)       86.40% (final) / 86.74% (best epoch)     0.3173

Surprisingly, the improved model did not outperform the baseline. The best validation accuracy achieved during transfer learning was 86.74%, which is lower than the baseline's 90.20%.

### Possible Reasons

**1.Dataset size vs. model capacity**: VGG16 is a very deep network with over 14 million parameters in its convolutional base. Even though the base was frozen, the new classifier might still require more data or more careful fine-tuning to adapt to the cat/dog domain.

**2.Limited training time**: Only 13 epochs were run due to early stopping. Perhaps unfreezing some of the top convolutional layers and training with a very low learning rate could yield better results, but that would increase the risk of overfitting.

**3.Baseline already strong**: The custom CNN achieved 90% accuracy, leaving little room for improvement. The dataset may not be complex enough to benefit from a much larger pre-trained model.

**4.Learning rate and optimization**: The initial learning rate (1e-4) might have been suboptimal for the new layers; a smaller rate or layer-wise fine-tuning could help.

### Reflection on What I Learned

This experiment reinforced several key concepts about CNNs and image classification:

**1.Architecture design matters**: The custom CNN with four convolutional blocks and dropout was able to learn meaningful features from scratch and generalize well (no overfitting). Its simple structure was sufficient for this task.

**2.Transfer learning is powerful but not always superior**: While pre-trained models like VGG16 are excellent starting points, they are not a silver bullet. For relatively simple datasets or when the baseline is already strong, the gains may be marginal. The frozen features from ImageNet might not align perfectly with the specific visual patterns of cats and dogs (e.g., fur textures, ear shapes), and the classifier head needs enough data to adapt.

**3.Data augmentation helps**: The baseline already used augmentation, which contributed to its good generalization. The improved model used even stronger augmentation, but this alone did not boost accuracy.

**4.Monitoring training curves is essential**: The baseline showed steady improvement and close train/val accuracy, indicating a good fit. The transfer learning model plateaued quickly, suggesting that either the capacity was insufficient or the training setup needed adjustment.

**5.Evaluation must go beyond a single number**: Although the improved model's final accuracy was lower, its precision/recall values (not shown) might differ. Inspecting misclassifications (as done in the evaluation script) reveals that some cat images were confidently misclassified as dogs, highlighting the difficulty of intra-class variation and background clutter.

In future work, I would experiment with fine-tuning the top convolutional layers of VGG16 after initial training, using a very low learning rate, and possibly adding more fully connected layers to better capture domain-specific features.
