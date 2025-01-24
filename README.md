# Handwritten Digit Recognition

This project focuses on recognizing handwritten digits using various machine learning models and techniques. The MNIST dataset was utilized to train and evaluate the models.

---

## Introduction

The ability to recognize handwritten digits is a fundamental task in computer vision with practical applications such as postal code recognition, bank check processing, and form digitization. This project leveraged the MNIST dataset, a widely used benchmark, to achieve high accuracy in digit recognition.

---

## Dataset Description

- **Dataset**: MNIST
  - 70,000 grayscale images of handwritten digits
  - Image size: 28x28 pixels
  - Training set: 60,000 images
  - Testing set: 10,000 images
- **Challenges**:
  - Data imbalance in some digits.
  - Poorly written or ambiguous digits.
  - High dimensionality (784 features per image).
  - Risk of overfitting in complex models like CNNs.

---

## Techniques Used

1. **Data Normalization**: Scaling pixel values to a range of 0 to 1 for better model convergence and stability.
2. **Dropout Regularization**: Applied in the CNN architecture to prevent overfitting by randomly deactivating neurons during training.
3. **RBF Kernel**: Used in the SVM model to handle non-linear data effectively by transforming feature space.

---

## Systematic Methods

1. Import libraries.
2. Load data.
3. Data preparation.
4. Normalization.
5. Reshaping.
6. Model creation.
7. Preprocessing.

---

## Models and Evaluation

### 1. Convolutional Neural Network (CNN)
- **Architecture**:
  - Two convolutional layers with ReLU activation and max pooling.
  - Fully connected dense layers with dropout for regularization.
  - Output layer with softmax activation.
- **Performance**:
  - Training accuracy: 99.43%
  - Test accuracy: 98.83%
  - Training loss: 1.67%
  - Test loss: 4.61%
  - Moderate training time over 10 epochs.

### 2. K-Nearest Neighbors (KNN)
- **Details**:
  - Uses Euclidean distance with an optimal number of neighbors set to 5.
- **Performance**:
  - Training accuracy: 97.9%
  - Test accuracy: 96.7%
  - Slow inference on large datasets.

### 3. Support Vector Machine (SVM) with RBF Kernel
- **Details**:
  - RBF kernel effectively handles non-linear data.
- **Performance**:
  - Validation accuracy: 97.75%
  - Test accuracy: 97.7%
  - High training speed.

### 4. Decision Tree
- **Details**:
  - Splits data based on feature thresholds.
- **Performance**:
  - Training accuracy: 100%
  - Test accuracy: 87.5%
  - Fast training and inference speed.

---

## Comparison of Models

| Model              | Accuracy (%) | Training Time | Inference Speed | Scalability |
|--------------------|--------------|---------------|-----------------|-------------|
| **CNN**           | 99.4         | Moderate      | Fast            | High        |
| KNN               | 97.9         | Fast          | Slow            | Low         |
| SVM (RBF Kernel)  | 97.7         | High          | Moderate        | Medium      |
| Decision Tree      | 86.7         | Fast          | Fast            | Medium      |

---

## Results and Conclusion

- **Best Model**: CNN
  - Accuracy: 99.4%
  - Scalable and fast inference.
- SVM performs well but is less scalable.
- KNN lacks efficiency for large datasets.
- Decision Tree is fast but less accurate.

**Recommendation**: For production deployment, the CNN model is ideal due to its high accuracy, scalability, and efficient inference. Further optimizations such as enhancing the CNN architecture or integrating transfer learning could further improve performance.

---

