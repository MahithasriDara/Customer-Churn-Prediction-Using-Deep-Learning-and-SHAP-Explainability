# Customer-Churn-Prediction-Using-Deep-Learning-and-SHAP-Explainability

# Customer Churn Prediction Using Deep Learning (Telco Dataset)

This project applies exploratory data analysis (EDA) and a deep learning Artificial Neural Network (ANN) model to predict customer churn using the Telco Customer Churn dataset. The model is built using TensorFlow/Keras and enhanced with SHAP for model interpretability.

---

## 📌 Dataset

- **Source:** [Kaggle - Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- This dataset contains customer data from a fictional telecom company and indicates whether or not a customer has left the company.

---

## 🧠 Objectives

- Understand customer churn trends through EDA
- Preprocess and engineer features for modeling
- Build a deep learning model (ANN) for binary classification
- Evaluate the model using standard metrics (Accuracy, Precision, Recall, AUC)
- Interpret model predictions using SHAP

---

## 🛠️ Tech Stack

- Python
- Pandas, NumPy
- Matplotlib, Seaborn
- TensorFlow / Keras
- Scikit-learn
- SHAP (SHapley Additive exPlanations)

---

## 📊 Exploratory Data Analysis (EDA)

- Distribution of churned vs. retained customers
- Feature correlations (tenure, contract type, charges, etc.)
- Churn trends across demographics and service usage
- Handled missing values and outliers

---

## 🔧 Preprocessing Steps

- Converted categorical variables using One-Hot Encoding
- Normalized continuous numerical features
- Handled missing values (`TotalCharges`)
- Encoded the target variable (`Churn`) into binary labels

---

## 🤖 Model Architecture

- Input → Dense(64) + ReLU → Dropout → Dense(32) + ReLU → Dense(1) + Sigmoid
- Loss: Binary Crossentropy
- Optimizer: Adam
- Early stopping applied to prevent overfitting

---

## 📈 Model Evaluation

- Achieved ~85% test accuracy
- Precision, Recall, F1 Score, and AUC used for deeper insights
- Confusion matrix and ROC curve plotted

---

## 🧠 SHAP Analysis

- Used SHAP to explain feature contributions to predictions
- Plotted summary and individual force plots
- Key churn drivers: Contract type, Tenure, Monthly Charges, Internet Service

---

## 📁 Repository Structure

Customer-Churn-Prediction-Using-Deep-Learning-and-SHAP-Explainability/
├── model/
│   ├── saved_model (1).pb
│   ├── keras_metadata.pb
│   └── variables/
│       ├── variables.data-00000-of-00001
│       └── variables.index
├── Telco-Customer-Churn.csv              # Dataset file
├── EDA_and_ANN_Telco_Customer_Churn_Predictions.ipynb  # Jupyter Notebook
├── Churn Prediction.pptx                 # Presentation (optional for showcase)
├── README.md  
