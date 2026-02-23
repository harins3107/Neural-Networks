# 🧠 Task 2: Memory-Based Neural Network for Time-Series Prediction

## 📌 Problem Statement

The goal of this project is to study and implement a **memory-based neural network** for a **time-series prediction task**.

Time-series data contains values recorded over time (like monthly milk production).  
To predict future values, we use a model that can remember past values.

In this project, we implemented a:

- Recurrent Neural Network (RNN)

The model was built **from scratch using only NumPy, Pandas, and core Python libraries**.  
No TensorFlow, Keras, or PyTorch were used.

The trained model was deployed using a **Streamlit interface**.

---

## 📊 Dataset Description

**Dataset Used:** Monthly Milk Production Dataset

This dataset contains:

- `Date` column (monthly data)
- `Production` column (milk production values)

### Dataset Features:

- Monthly time-series data
- Single variable (univariate)
- Supervised regression problem
- Clear seasonal pattern

We used the dataset to predict future monthly milk production.

---

## 🧠 Model Architecture Explanation

We implemented a **Simple Recurrent Neural Network (RNN)**.

### Why RNN?

Unlike normal neural networks, RNNs:

- Remember past information
- Maintain a hidden state
- Are suitable for sequential/time-series data

---

### 🔹 RNN Working Concept

At each time step:

Hidden State:

```

h_t = tanh(Wxh * x_t + Whh * h_(t-1) + bh)

```

Output:

```

y_t = Why * h_t + by

```

Where:

- Wxh = Input-to-hidden weights  
- Whh = Hidden-to-hidden weights  
- Why = Hidden-to-output weights  
- bh = Hidden bias  
- by = Output bias  

The hidden state carries information from previous months.  
This allows the model to "remember" past data.

---

## ⚙️ Implementation Details

### 1️⃣ Data Preprocessing

- Converted Date column to datetime
- Sorted data by date
- Split dataset:
  - Training set → All except last 12 months
  - Testing set → Last 12 months
- Applied MinMax scaling (implemented manually using NumPy)

Scaling formula:

```

scaled = (x - min) / (max - min)

```

---

### 2️⃣ Sequence Creation

We used a sliding window approach.

If window size = 12:

Input:
```

[Month1, Month2, ..., Month12]

```

Target:
```

Month13

```

This converts time-series data into supervised learning format.

---

### 3️⃣ Model Training

- Forward pass implemented manually
- Backpropagation Through Time (BPTT) implemented manually
- Mean Squared Error (MSE) used as loss function
- Gradient descent used to update weights

All computations were done using NumPy only.

---

### 4️⃣ Model Evaluation

The model was evaluated using regression metrics:

- **MAE (Mean Absolute Error)**
- **RMSE (Root Mean Squared Error)**
- **R² Score**

Lower MAE and RMSE indicate better performance.  
R² closer to 1 indicates better fit.

---

## 🚀 Deployment

The trained model was deployed using **Streamlit**.

The Streamlit interface allows:

- Viewing dataset preview
- Adjusting prediction horizon (number of months ahead)
- Viewing actual vs predicted graph
- Checking evaluation metrics
- Making manual predictions

The trained model is saved as:

```

rnn_milk_model.npz

```

---

## 📈 Visualization

The Streamlit interface displays:

- Actual milk production values
- Test predictions
- Future forecast values
- Performance metrics (MAE, RMSE, R²)

Screenshots of the Streamlit interface are included in this repository.

---

## 📂 Project Structure

```

├── RNN_numpy_only.ipynb        # Model implementation and training
├── streamlit.py                # Streamlit deployment file
├── monthly_milk_production.csv # Dataset
├── rnn_milk_model.npz          # Saved trained model
├── requirements.txt            # Required libraries
└── README.md                   # Project documentation

```

---

## ▶️ Instructions to Run the Project

### 1️⃣ Install Dependencies

```

pip install -r requirements.txt

```

Or manually install:

```

pip install numpy pandas matplotlib streamlit

```

---

### 2️⃣ Train the Model (Optional)

Run the notebook:

```

RNN_numpy_only.ipynb

```

This will train the model and generate:

```

rnn_milk_model.npz

```

---

### 3️⃣ Run the Streamlit App

Navigate to project folder and run:

```

streamlit run streamlit.py

```

The app will open in your browser.

---

## ✅ Conclusion

In this project:

- A memory-based neural network (RNN) was implemented from scratch.
- Time-series data was processed and scaled manually.
- The model was trained using historical data.
- Performance was evaluated using regression metrics.
- The trained model was deployed using Streamlit.
- Predictions were visualized through an interactive interface.

This project demonstrates understanding of:

- Time-series forecasting
- Memory-based neural networks
- Manual neural network implementation
- Model deployment using Streamlit
```

