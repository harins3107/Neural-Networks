# 🧠 Handwritten Digit Classification using Neural Networks (From Scratch)

## 📌 Problem Statement

The objective of this project is to implement an image classification model from scratch using only NumPy, Pandas, and core Python libraries — without using high-level deep learning frameworks such as TensorFlow, Keras, or PyTorch.

The model is trained to classify handwritten digits (0–9) from grayscale images and is integrated with a Streamlit-based interface that allows users to upload an image and view the predicted output.

---

## 📂 Dataset Description

This project uses the MNIST Handwritten Digits dataset.

- Total Classes: 10 (Digits 0–9)
- Image Size: 28 × 28 pixels
- Color Format: Grayscale
- Total Features per Image: 784 (28×28 flattened)
- Format Used: CSV (train.csv)

Each row in the dataset contains:
- First column → Label (digit 0–9)
- Remaining 784 columns → Pixel intensity values (0–255)

### Data Preprocessing Steps:
- Dataset shuffled before splitting
- Pixel values normalized by dividing by 255
- Split into:
  - Training Set
  - Validation (Dev) Set

---

## 🏗 Model Architecture Explanation

This implementation builds a fully connected Neural Network from scratch using NumPy.

### Architecture:

Input Layer:
- 784 neurons (flattened 28×28 image)

Hidden Layer:
- 64 neurons
- Activation Function: ReLU

Output Layer:
- 10 neurons (representing digits 0–9)
- Activation Function: Softmax

---

### 🔁 Forward Propagation:

1. Z1 = W1X + b1  
2. A1 = ReLU(Z1)  
3. Z2 = W2A1 + b2  
4. A2 = Softmax(Z2)

---

### 🔄 Backward Propagation:

Gradients are manually computed using:

- Cross-Entropy Loss derivative
- ReLU derivative
- Weight gradient calculation
- Bias gradient calculation

Weights are updated using Gradient Descent.

---

### 🧮 Weight Initialization:

He Initialization is used to improve training stability for ReLU activations.

---

## ⚙️ Implementation Details

Programming Language: Python

Libraries Used:
- NumPy
- Pandas
- Matplotlib
- Streamlit
- Pillow
- Pickle (for saving and loading model)

---

### Training Configuration:

- Learning Rate: 0.1
- Number of Iterations: 1000
- Optimization Method: Gradient Descent
- Loss Function: Cross-Entropy Loss

---

### Model Saving:

After training, model parameters are saved as:

mnist_model.pkl

The saved parameters include:
- W1
- b1
- W2
- b2

The model is saved using Python's pickle module and later loaded inside the Streamlit application.

---

## 📊 Evaluation Metrics Used

### ✔ Accuracy

Accuracy = (Correct Predictions / Total Predictions)

Results Achieved:

- Training Accuracy: ~93.7%
- Validation Accuracy: ~93.7%

This indicates strong generalization performance.

Optional evaluation tools:
- Confusion Matrix
- Softmax probability confidence scores

---

## 🌐 Streamlit Interface Integration

The trained model is integrated into a Streamlit-based web interface.

### Features:

- Upload handwritten digit image (PNG/JPG/JPEG)
- Automatic resizing to 28×28 pixels
- Grayscale conversion
- Pixel normalization
- Prediction display
- Confidence percentage display

The interface allows real-time digit classification.

---


---

## 🚀 Instructions to Run the Project

### Step 1: Install Required Dependencies

Open terminal inside project folder and run:

pip install -r requirements.txt

---

### Step 2: Train the Model (If Not Already Trained)

Open Jupyter Notebook:

jupyter notebook

Run all cells in:

main.ipynb

This will generate:

mnist_model.pkl

---

### Step 3: Run the Streamlit Application

In the terminal:

streamlit run app.py

Open browser at:

http://localhost:8501

Upload an image and view prediction.

---

## 📈 Sample Output

After uploading a digit image:

Prediction: 7  
Confidence: 98.34%

---

## 📌 Key Learning Outcomes

- Understanding neural network forward propagation
- Manual implementation of backpropagation
- ReLU and Softmax activation functions
- Gradient descent optimization
- Model parameter saving/loading
- Deployment using Streamlit

---

## ⚠️ Restrictions Followed

- No TensorFlow
- No Keras
- No PyTorch
- Implemented fully using NumPy and core Python

---

## 📸 Streamlit Interface Screenshots
(\images\Screenshot 2026-02-23 224609.png)
(\images\Screenshot 2026-02-23 224544.png)


## 👨‍💻 Author

[Harini VK]


AIML Activity Submission
