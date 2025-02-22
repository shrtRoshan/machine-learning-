# Parkinson's Disease Progression Prediction

This repository contains machine learning models to predict Parkinson's disease progression using the `motor_UPDRS` score. The project includes exploratory data analysis (EDA), model training, and a deployed Streamlit web app.

## 📌 Project Overview

- **Dataset**: Parkinson's disease dataset containing motor-UPDRS scores.
- **EDA**: Conducted in `main.ipynb`.
- **Model Training**: Various machine learning models trained and stored in the `models/` directory.
- **Streamlit App**: Deployed at [Parkinson Predictor](https://parkinson-predictor.streamlit.app/), using XGBoost for predictions.

## 📂 Repository Structure

```
│   .gitignore
│   main.ipynb              # Exploratory Data Analysis (EDA)
│   README.md               # Project Documentation
│   requirements.txt        # Required Dependencies
│   streamlit_app.py        # Streamlit Web Application
│
├───data                    # Dataset Files
│       parkinsons_updrs.data
│       parkinsons_updrs.names
│       test.csv
│       train.csv
│       val.csv
│
├───models                  # Trained Machine Learning Models
│       best_adaboost_model.pkl
│       best_catboost_model.pkl
│       best_gbdt_model.pkl
│       linear_regression_model.pkl
│       random_forest_model.pkl
│       xgboost_best_model.pkl
│
└───model_train             # Model Training Notebooks
        ada-boost.ipynb
        cat-boost.ipynb
        db-scan.ipynb
        gradient-boosting.ipynb
        linear-regression.ipynb
        random-forest-regressor.ipynb
        xg-boost.ipynb
```

## 🚀 Getting Started

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/shrtRoshan/machine-learning-.git
cd machine-learning-
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Run the Streamlit App

```bash
streamlit run streamlit_app.py
```

## 📊 Model Training

The models were trained using different algorithms. Training notebooks can be found in the `model_train/` directory.

- Stratified K-Fold Cross-Validation was used.
- Hyperparameter tuning performed with Optuna.
- Feature importance analyzed using SHAP.

## 🌐 Deployment

The Streamlit app is deployed [here](https://parkinson-predictor.streamlit.app/)

## 🤝 Contributions

Feel free to fork, raise issues, or contribute by submitting a pull request!

---

🔗 **GitHub Repository**: [machine-learning-](https://github.com/shrtRoshan/machine-learning-.git)

