# Adult Income Prediction using Machine Learning

A machine learning classification project that predicts whether an individual's annual income is **≤50K or >50K** based on demographic, educational, occupational, and financial attributes.

## 🚀 Project Overview

This project uses the Adult/Census Income dataset to build a binary classification model.

The complete workflow includes:

- Data cleaning
- Missing value treatment
- Duplicate removal
- Exploratory Data Analysis
- Categorical feature encoding
- Feature scaling
- SMOTE for class balancing
- Machine learning model comparison
- Random Forest classification
- Hyperparameter tuning
- Streamlit deployment

## 📊 Dataset

The dataset contains information such as:

- Age
- Workclass
- Education
- Education Number
- Marital Status
- Occupation
- Relationship
- Race
- Sex
- Capital Gain
- Capital Loss
- Hours per Week
- Native Country

### Target

The target variable is:

- `<=50K`
- `>50K`

## 🧹 Data Preprocessing

The following preprocessing steps were performed:

1. Missing values represented by `?` were converted to NaN.
2. Missing categorical values were filled using the mode.
3. Duplicate records were removed.
4. The target variable was encoded as:
   - `0` → `<=50K`
   - `1` → `>50K`
5. Categorical variables were one-hot encoded.
6. Numerical features were standardized.
7. SMOTE was applied only to the training data to handle class imbalance.

## 🤖 Machine Learning Models

The following classification algorithms were evaluated:

- Logistic Regression
- K-Nearest Neighbors
- Random Forest
- Gradient Boosting

## 🏆 Model Performance

The baseline Random Forest model achieved approximately:

**Accuracy: 84.86%**

The Random Forest model was selected as the final accuracy-focused model.

Hyperparameter tuning using GridSearchCV achieved a cross-validation accuracy of approximately:

**88.79%**

The tuned model achieved approximately:

**84.31% test accuracy**

## 🌐 Streamlit Deployment

The trained Random Forest model was deployed using Streamlit.

The application allows users to enter personal, educational, occupational, and financial information and receive an income prediction.

### Prediction Output

The application predicts:

- `≤50K`
- `>50K`

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Imbalanced-learn
- Joblib
- Streamlit
- Jupyter Notebook
- VS Code

## 📁 Project Structure

```text
Adult-Income-Prediction/
│
├── adult.data
├── app.py
├── train_model.py
├── adult_income_prediction.ipynb
├── preprocessor.pkl
├── scaler.pkl
├── random_forest_model.pkl
├── requirements.txt
├── .gitignore
└── README.md


## 🌐 Run the Streamlit Application

You can run this project locally by either cloning or forking the repository.

### Option 1: Fork the Repository

1. Open this GitHub repository.
2. Click the **Fork** button in the top-right corner.
3. Select your GitHub account.
4. Open your forked repository.
5. Click **Code → Copy HTTPS URL**.

### Option 2: Clone the Repository

Open your terminal and run:

```bash
git clone https://github.com/nafiya0/adult-income-prediction.git 

git clone https://github.com/nafiya0/adult-income-prediction.git
cd adult-income-prediction
pip install -r requirements.txt
streamlit run app.py