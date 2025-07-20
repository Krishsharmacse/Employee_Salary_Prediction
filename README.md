# 👨‍💼 Employee Salary Prediction Application

A machine learning-powered web application built with Streamlit that predicts whether an employee earns above or below \$50K annually based on various demographic and employment features. The model uses an XGBoost classifier, selected after comprehensive model comparison for optimal accuracy.

## 🌐 Live Demo

Try it now: **Employee Salary Prediction App** (Deployed on Streamlit Community Cloud)

---

## 🚀 Features

* **Interactive Web Interface**: User-friendly Streamlit UI with sidebar inputs
* **Single Prediction**: Input individual employee details for salary prediction
* **Batch Prediction**: Upload CSV files for bulk prediction
* **Data Visualization**: Display inputs and prediction results
* **Download Results**: Export batch predictions as CSV

---

## 🤖 Model Details

* **Algorithm**: XGBoost Classifier
* **Model Type**: Gradient Boosting Classifier
* **Key Hyperparameters**:

  * n\_estimators: 100
  * learning\_rate: 0.1
  * max\_depth: 6
  * eval\_metric: 'logloss'
* **Model Selection**: Chosen after comparing multiple algorithms

### Data Preprocessing Pipeline

* **Outlier Removal**:

  * Age: 17-65 years
  * Education: 5-16 years
* **Feature Engineering**:

  * Added 'Experience' = Age - Educational Years - 6
* **Data Cleaning**:

  * Replaced missing values ('?') with 'Others'
  * Removed non-impactful categories (e.g., Without-pay)
  * Dropped redundant features (e.g., education, fnlwgt)
* **Encoding**:

  * Label encoding for all categorical variables
* **Scaling**:

  * MinMaxScaler for numerical features

---

## 📋 Prerequisites

* **Dataset**: Adult Census Income
* **Target**: Binary classification (≤50K vs >50K)
* **Python**: 3.8 or higher
* **Libraries**: See `requirements.txt`

### Input Features

| Feature        | Type        | Description                |
| -------------- | ----------- | -------------------------- |
| Age            | Numerical   | Age of individual (17-65)  |
| Experience     | Numerical   | Years of experience (0-50) |
| Workclass      | Categorical | Employment type            |
| Education      | Categorical | Education level            |
| Marital Status | Categorical | Marital status             |
| Occupation     | Categorical | Job type                   |
| Relationship   | Categorical | Household role             |
| Race           | Categorical | Racial group               |
| Gender         | Categorical | Male/Female                |
| Capital Gain   | Numerical   | Investment income          |
| Capital Loss   | Numerical   | Investment loss            |
| Hours per Week | Numerical   | Weekly work hours (1-100)  |
| Native Country | Categorical | Country of origin          |

---

## 🛠️ Installation

```bash
git clone <your-repository-url>
cd employee-salary-prediction
pip install -r requirements.txt
```

### Required Files

* `model.pkl` - Trained XGBoost classifier
* `scaler.pkl` - MinMaxScaler for features
* `label_encoders.pkl` - Label encoders
* `adult 3.csv` - Training dataset
* `Salary_prediction_model.ipynb` - Jupyter notebook for training

---

## 🏃‍♂️ Running the Application

```bash
streamlit run app.py
```

Opens in browser: [http://localhost:8501](http://localhost:8501)

---

## 📊 How to Use

### Single Prediction

1. Use sidebar to enter inputs
2. Click "Predict Salary"
3. View prediction output (<=50K or >50K)

### Batch Prediction

1. Upload CSV file with following columns:

```csv
age,experience,workclass,educational-num,marital-status,occupation,relationship,race,gender,capital-gain,capital-loss,hours-per-week,native-country
```

2. View batch predictions in table
3. Download as CSV

---

## 🔬 Model Output

* **0**: Salary <= \$50K
* **1**: Salary > \$50K

---

## 📂 Project Structure

```
employee-salary-prediction/
├── app.py                          # Streamlit frontend
├── Salary_prediction_model.ipynb   # Training pipeline
├── model.pkl                       # Final trained model
├── scaler.pkl                      # Fitted MinMaxScaler
├── label_encoders.pkl              # Fitted encoders
├── adult 3.csv                     # Training dataset
├── requirements.txt                # Python dependencies
└── README.md                       # Documentation
```

---

## 🔧 Technical Details

* **Framework**: Streamlit
* **Algorithm**: XGBoost
* **Preprocessing**: pandas, sklearn
* **Visualization**: matplotlib
* **Model Saving**: joblib, pickle
* **Notebook**: Model comparison, analysis, training pipeline

---

## 🔬 Model Training & Development

To retrain or inspect the model:

```bash
jupyter notebook Salary_prediction_model.ipynb
```

### Training Steps

* Load and explore data
* Remove outliers
* Feature engineering
* Encode & scale
* Model comparison
* Train XGBoost with best params
* Save model and encoders

---

## 📅 CSV Format for Batch Prediction

**Required Columns**:

```csv
age,experience,workclass,educational-num,marital-status,occupation,relationship,race,gender,capital-gain,capital-loss,hours-per-week,native-country
```

**Sample Row**:

```csv
39,5,Private,13,Never-married,Adm-clerical,Not-in-family,White,Male,2174,0,40,United-States
```

---

## 📄 License

MIT License - see LICENSE file

---

## 🚗 Troubleshooting

* **Missing Files**: Ensure `.pkl` files exist
* **CSV Errors**: Check column names match format
* **Env Issues**: Use virtual environment if needed

---

## 📢 Support

Feel free to reach out via GitHub Issues or connect with the author d
