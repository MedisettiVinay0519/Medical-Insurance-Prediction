# 🏥 Medical Insurance Premium Prediction (ML & ANN)

This project tackles the classic machine learning problem of **regression** by focusing on **predicting individual medical insurance premiums** based on health and demographic data. It provides a complete data science workflow, from in-depth exploratory data analysis (EDA) to implementing and evaluating advanced models, including both a traditional **Machine Learning (ML)** model and an **Artificial Neural Network (ANN)**.

---

## 🎯 Goal

The primary goal is to build an accurate predictive model that can estimate the `expenses` (insurance premium) for an individual. This is a critical task for insurance providers for risk assessment and for policyholders for cost estimation.

## 🚀 Key Features

* **Comprehensive Data Exploration:** Thorough analysis of feature distributions and categorical variables.
* **Feature Engineering:** Preprocessing steps including one-hot encoding for categorical data.
* **Advanced Modeling:** Implementation of both a traditional Machine Learning model and an **Artificial Neural Network (ANN)**.
* **Performance Evaluation:** Rigorous model assessment using key regression metrics: $R^2$ Score, MSE, RMSE, and MAE.

---

## 📈 Data Visualization & Analysis

Visual analysis is crucial for understanding the dataset and model performance.

### Correlation Matrix & Feature Importance

The **Correlation Matrix** reveals linear relationships between all features. Notably, the strong positive correlation between the **Smoker** status and **Expenses** highlights its significance as a primary cost driver. The **Feature Importance Plot** confirms which variables have the largest impact on the final premium prediction.

| Correlation Matrix | Feature Importance Plot |
| :---: | :---: |
| ![Correlation Heatmap](images/Screenshot%202025-10-21%20143119.png) | ![Feature Importance](image_db2e5d.png) |

---

## 🧠 Model Implementation

This project evaluates the predictive power of two distinct approaches:

### 1. Traditional Machine Learning Model

* **Model Used:** A **traditional Machine Learning Regression Model** (such as Linear Regression, Ridge, or Lasso) is used as an interpretable baseline to establish performance metrics.

### 2. Artificial Neural Network (ANN)

* **Architecture:** An **Artificial Neural Network (ANN)** is developed using deep learning libraries like **TensorFlow/Keras**. This model is highly effective at capturing the complex, non-linear relationships often present in insurance data.
* **Structure:** The network includes:
    * An **Input Layer** accepting the preprocessed features.
    * Multiple **Hidden Layers** (typically with ReLU activation) for deep pattern recognition.
    * A single-neuron **Output Layer** (linear activation) for continuous value prediction.

---

## 📊 Dataset Overview

The project uses the **Insurance Dataset**.

| Column | Description | Type |
| :--- | :--- | :--- |
| `age` | Age of the primary beneficiary. | `Numerical` |
| `sex` | Gender of the policyholder. | `Categorical` |
| `bmi` | Body Mass Index (a measure of body fat). | `Numerical` |
| `children` | Number of children covered by health insurance. | `Numerical` |
| `smoker` | Indicates whether the beneficiary smokes. | `Categorical` |
| `region` | The beneficiary's residential area. | `Categorical` |
| `expenses` | **Target Variable:** Individual medical costs billed by health insurance. | `Numerical` |

---

## ✅ Results & Performance

The final, best-performing model achieved the following results (as confirmed in your notebook):

| Metric | Value |
| :--- | :--- |
| **$R^2$ Score** | **0.8281** |
| **Mean Squared Error (MSE)** | **40,476,451.43** |
| **Root Mean Squared Error (RMSE)** | **6,362.11** |
| **Mean Absolute Error (MAE)** | **5,558.55** |

---

## 🛠️ Installation and Setup


1.  **Install Dependencies:**
    The project relies on standard Python data science libraries:
    ```bash
    pip install pandas numpy matplotlib seaborn scikit-learn tensorflow keras
    ```

2.  **Run the Analysis:**
    Open and run the `medical-permium-insurance-using-ann and ml.ipynb` notebook in a Jupyter environment to execute the complete workflow.

---

