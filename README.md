# CIFAR-10 Image Classification: ANN vs. CNN

This project demonstrates the implementation of deep learning models using TensorFlow to classify the CIFAR-10 dataset. It covers the transition from manual gradient descent in Artificial Neural Networks (ANN) to the more efficient feature extraction of Convolutional Neural Networks (CNN).

---

## 1. Loss Function
The model utilizes **Sparse Categorical Cross-Entropy** as the objective function. This is used because the labels are provided as integers (0-9) rather than one-hot encoded vectors. 

### Mathematical Formula
For a single prediction, the loss ($L$) is defined as:

$$L = -\sum_{i=1}^{C} y_i \log(\hat{y}_i)$$

In this specific implementation, we use `from_logits=True`, meaning the loss function internally applies the Softmax activation to the raw output scores ($s$):

$$\hat{y}_i = \frac{e^{s_i}}{\sum_{j=1}^{C} e^{s_j}}$$

---

## 2. Gradient Descent Update Rule
In Part 2 of the notebook, the training loop manually updates the model parameters using the **Stochastic Gradient Descent (SGD)** rule. The `tf.GradientTape` records the operations to compute the derivative of the loss with respect to each weight.

### The Update Formula
For each trainable parameter ($W$), the update is performed as follows:

$$W = W - \eta \cdot \nabla_W L$$

Where:
* $W$: The weights or biases of the model.
* $\eta$ (Learning Rate): A scalar determining the step size toward the minimum (set to $0.01$).
* $\nabla_W L$: The gradient of the loss with respect to the parameter.

---

## 3. Comparison of Architectures
The following table summarizes the performance of the different models tested on the CIFAR-10 dataset.

| Model Type | Key Features | Training Accuracy | Test Accuracy |
| :--- | :--- | :--- | :--- |
| **Simple ANN** | Standard Feed-Forward, 5 Epochs | ~50.2% | **49.19%** |
| **ANN w/ Dropout** | 20% Dropout layers to reduce overfitting | 53.86% | **49.57%** |
| **Simple CNN** | Conv2D and MaxPooling layers | 79.66% | **71.30%** |



---

## 4. Key Findings
* **Flattening:** We flatten the 32x32x3 images into a 3072-dimensional vector for the ANN because standard dense layers require 1D input arrays to perform matrix multiplication ($X \cdot W + B$).
* **Dropout:** Implementing Dropout layers helped the model generalize better. While it may slow down the training accuracy progression, it typically results in a smaller gap between training and validation performance.
* **CNN Performance:** The CNN outperformed the ANN by over 20%. This is because CNNs use kernels (filters) to detect spatial features like edges and textures, whereas ANNs treat every pixel as an independent feature.