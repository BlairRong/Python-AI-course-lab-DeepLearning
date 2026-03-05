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

| Layer Type          | Details                              |
|---------------------|--------------------------------------|
| Conv2D + ReLU       | 32 filters, 3×3, input (150,150,3)   |
| MaxPooling2D        | 2×2                                  |
| Conv2D + ReLU       | 64 filters, 3×3                      |
| MaxPooling2D        | 2×2                                  |
| Conv2D + ReLU       | 128 filters, 3×3                     |
| MaxPooling2D        | 2×2                                  |
| Conv2D + ReLU       | 128 filters, 3×3                     |
| MaxPooling2D        | 2×2                                  |
| Flatten             |                                      |
| Dropout             | 0.5                                  |
| Dense + ReLU        | 512 units                            |
| Dense + Sigmoid     | 1 unit (binary output)               |

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
   python scripts/download_data.py
   python scripts/organize_data.py
   ```

4. **Train the baseline model**  
   ```bash
   python scripts/train_model.py
   ```

5. **Evaluate and visualize predictions**  
   ```bash
   python scripts/evaluate_model.py
   ```

6. **(Optional) Run transfer learning**  
   ```bash
   python scripts/improved_model.py
   ```

All key outputs (training history, best model, evaluation plots) are saved in the `models/` and `data/` folders.

## 📁 Repository Structure

```
3.2_lab_catdog_classifier/
├── data/                  # Dataset (ignored by git)
├── models/                # Saved model weights
├── scripts/               # Python scripts for each step
│   ├── download_data.py
│   ├── organize_data.py
│   ├── inspect_data.py
│   ├── preprocess_data.py
│   ├── train_model.py
│   ├── evaluate_model.py
│   └── improved_model.py
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
