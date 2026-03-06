# Cat vs Dog Image Classifier with CNN

This repository contains a complete deep learning project that classifies images of cats and dogs using a custom Convolutional Neural Network (CNN). The model achieves **90.2% validation accuracy** on the Kaggle Dogs vs Cats dataset, demonstrating effective feature learning from scratch without relying on pre‑trained weights for the final submission.

## 📌 Project Overview

- **Goal**: Build a binary image classifier to distinguish between cats and dogs.
- **Dataset**: [Kaggle Dogs vs Cats](https://www.microsoft.com/en-us/download/details.aspx?id=54765) (25,000 images, resized to 150×150).
- **Approach**:
  - Custom CNN with 4 convolutional blocks + dense layers.
  - Data augmentation (rotation, zoom, brightness) to improve generalization.
  - Early stopping and model checkpointing to prevent overfitting.
  - Transfer learning experiment using VGG16 for comparison.
- **Key Achievements**:
  - **90.2% validation accuracy** (custom CNN).
  - Thorough analysis of misclassifications and dataset challenges.
  - Insights into when transfer learning helps (or doesn’t) for this task.

## 🛠️ Technologies Used

- **Python 3.9**
- **TensorFlow / Keras** – model building, training, and evaluation.
- **Matplotlib / NumPy** – data visualization and numerical operations.
- **PIL / scikit‑learn** – image preprocessing and data splitting.
- **Git / GitHub** – version control and project sharing.

## 🧠 Model Architecture (Custom CNN)

input image size: (150, 150, 3)

| Layer Type             | Details                                                                       | Output Shape  |  Param # |
|------------------------|-------------------------------------------------------------------------------|---------------|----------|
| Conv2D + ReLU          | 32 filters, size 3×3x3 → (3×3×3 + 1 bias) × 32 = 28×32 = 896                  | (148, 148, 32)| 896      |
| MaxPooling2D           | 2×2 → Downsamples spatial dimensions by factor 2                              | (74, 74, 32)  | 0        |
| Conv2D + ReLU          | 64 filters, 3×3x32 → (3×3×32 + 1) × 64 = 289×64 = 18,496                      | (72, 72, 64)  | 18,496   |
| MaxPooling2D           | 2×2                                                                           | (36, 36, 64)  | 0        |
| Conv2D + ReLU          | 128 filters, 3×3×64 → (3×3×64 + 1) × 128 = 577×128 = 73,856                   | (34, 34, 128) | 73,856   |
| MaxPooling2D           | 2×2                                                                           | (17, 17, 128) | 0        |
| Conv2D + ReLU          | 128 filters, 3×3×128 → (3×3×128 + 1) × 128 = 1153×128 = 147,584               | (15, 15, 128) |147,584   |
| MaxPooling2D           | 2×2                                                                           | (7, 7, 128)   | 0        |
| Flatten                |  → 7×7×128 = 6272 neurons                                                     | (6272)        | 0        |
| Dropout                | 0.5 → Regularization – randomly disables 50% of neurons during training.      | (6272)        | 0        |
| Dense + ReLU           | 512 units → 6272×512 + 512 = 3,211,776                                        | (512)         | 3,211,776|
| Dense + Sigmoid        | 1 unit (binary output) → 512×1 + 1 = 513                                      | (1)           | 513      |
| Total trainable params | ~3.45 million                                                                 |               | 3,453,121|

- **Loss**: Binary Crossentropy  
- **Optimizer**: Adam (learning rate = 0.001)  
- **Regularization**: Dropout (0.5)  

## 📊 Results

| Model                   | Validation Accuracy | Validation Loss |
|-------------------------|---------------------|-----------------|
| Custom CNN (20 epochs)  | **90.2%**           | 0.2343          |
| VGG16 Transfer Learning | 86.7% (best epoch)  | 0.3173          |

**Learning curves** show steady improvement and no severe overfitting – training and validation accuracies stay close.

## 🚀 How to Run

1. **Clone the repository**  

   ```bash
   git clone https://github.com/BlairRong/Python-AI-course-lab-DeepLearning.git
   cd Python-AI-course-lab-DeepLearning/3.2_lab_catdog_classifier
   ```

2. **Set up the environment** (conda recommended)  

   ```bash
   conda create -n catdog python=3.9
   conda activate catdog
   pip install -r requirements.txt
   ```

3. **Download and prepare the dataset**  

   ```bash
   python scripts/1.download_data.py
   python scripts/2.organize_data.py
   python scripts/3.inspect_data.py
   python scripts/4.preprocess_data.py
   ```

4. **Train the baseline model**  

   ```bash
   python scripts/5.train_model.py
   ```

5. **Evaluate and visualize predictions**  

   ```bash
   python scripts/6.evaluate_model.py
   ```

6. **(Optional) Run transfer learning**  

   ```bash
   python scripts/7.improved_model.py
   ```

All key outputs (training history, best model, evaluation plots) are saved in the `models/` and `data/` folders.

## 📁 Repository Structure

```
3.2_lab_catdog_classifier/
├── data/                  # Dataset (ignored by git)
├── models/                # Saved model weights
├── scripts/               # Python scripts for each step
│   ├── 1.download_data.py
│   ├── 2.organize_data.py
│   ├── 3.inspect_data.py
│   ├── 4.preprocess_data.py
│   ├── 5.train_model.py
│   ├── 6.evaluate_model.py
│   └── 7.improved_model.py
├── requirements.txt       # Dependencies
└── README.md              # This file
```

## 💡 What I Learned

- How to design a CNN from scratch for image classification.
- The importance of data preprocessing and augmentation.
- Monitoring training curves to detect overfitting/underfitting.
- The trade‑offs between custom models and transfer learning.
- Real‑world challenges: long training times, misclassifications, and dataset biases.

## 📬 Contact

If you have any questions or would like to discuss this project further, feel free to reach out!

- **Email**: blair.rongsiying@163.com  
- **LinkedIn**: [Siying Rong](www.linkedin.com/in/siying-rong-83b732a4)
- **GitHub**: [BlairRong](https://github.com/BlairRong)

---

**Note**: The dataset is not included in this repository due to its large size. Please download it separately from the link above and place it in the `data/` folder before running the scripts.
