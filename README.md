#👨‍💼 Employee Salary Prediction Application
Production Link GitHub Repo

A machine learning-powered web application project built with Streamlit that predicts whether an employee earns above or below $50K annually based on various demographic and employment features. The model uses XGBoost classifier, selected after comprehensive model comparison for optimal accuracy.

🌐 Live Demo
Try the app now: Employee Salary Prediction App

The app is deployed on Streamlit Community Cloud and accessible to everyone!

🚀 Features
Interactive Web Interface: User-friendly Streamlit interface with sidebar input controls
Single Prediction: Input individual employee details and get instant salary predictions
Batch Prediction: Upload CSV files for bulk predictions
Data Visualization: Display input data and prediction results in tabular format
Download Results: Export batch predictions as CSV files
🤖 Model Details
Algorithm: XGBoost Classifier
Model Type: Gradient Boosting Classifier
Key Hyperparameters:
n_estimators: 100
learning_rate: 0.1
max_depth: 6
eval_metric: 'logloss'
Model Selection: XGBoost was chosen after comparing multiple algorithms based on accuracy performance
Data Preprocessing Pipeline
Outlier Removal: Age filtered to 17-65 years, education levels 5-16 years
Feature Engineering: Added 'Experience' feature (Age - Educational Years - 6)
Data Cleaning:
Replaced missing values ('?') with 'Others' category
Removed non-impactful categories (Without-pay, Never-worked, etc.)
Dropped redundant features (education, fnlwgt)
Encoding: Label encoding for all categorical variables
Scaling: MinMaxScaler for feature normalization
📋 Prerequisites
Source: Adult Census Income dataset

Features: 13 input features after preprocessing

Target: Binary classification (≤50K vs >50K)

Train/Test Split: 80/20 with stratification

Python 3.8 or higher

Required Python packages (see requirements.txt)

🛠️ Installation
Clone the repository (or download the files):

git clone <your-repository-url>
cd employee-salary-prediction
Install required packages:

pip install -r requirements.txt
Ensure model files are present: Make sure you have the following files in your project directory:

model.pkl - Trained XGBoost classifier model
scaler.pkl - Fitted MinMaxScaler for feature scaling
label_encoders.pkl - Label encoders for categorical variables
adult 3.csv - Training dataset (optional, for retraining)
Salary_prediction_model.ipynb - Model training notebook
🏃‍♂️ Running the Application
Execute the following command in your terminal:

streamlit run app.py
The application will open in your default web browser at http://localhost:8501

📊 Input Features
The model uses the following 13 features for prediction:

Feature	Type	Description
Age	Numerical	Age of the individual (17-65 years)
Experience	Numerical	Years of work experience (0-50 years)
Workclass	Categorical	Type of employment (Private, Government, Self-employed, etc.)
Education	Categorical	Highest education level achieved
Marital Status	Categorical	Current marital status
Occupation	Categorical	Type of occupation/job
Relationship	Categorical	Household relationship status
Race	Categorical	Racial background
Gender	Categorical	Gender (Male/Female)
Capital Gain	Numerical	Investment income (0-100,000)
Capital Loss	Numerical	Investment losses (0-5,000)
Hours per Week	Numerical	Average working hours per week (1-100)
Native Country	Categorical	Country of origin
💡 How to Use
Single Prediction
Use the sidebar to input employee details
Adjust sliders and select dropdown options
Click "Predict Salary" button
View the prediction result
Batch Prediction
Prepare a CSV file with the required columns:
age, experience, workclass, educational-num, marital-status
occupation, relationship, race, gender, capital-gain
capital-loss, hours-per-week, native-country
Upload the CSV file using the file uploader
View predictions in the results table
Download the results as a CSV file
📈 Model Output
The model predicts one of two salary categories:

Class 0: Employee earns ≤$50K annually
Class 1: Employee earns >$50K annually
🗂️ Project Structure
employee-salary-prediction/
│
├── app.py                          # Main Streamlit application
├── Salary_prediction_model.ipynb   # Model training and analysis notebook
├── model.pkl                       # Trained XGBoost model
├── scaler.pkl                      # Feature scaler (MinMaxScaler)
├── label_encoders.pkl              # Categorical variable encoders
├── adult 3.csv                     # Training dataset
├── requirements.txt                # Python dependencies
└── README.md                       # Project documentation
🔧 Technical Details
Framework: Streamlit for web interface
ML Algorithm: XGBoost Classifier (selected for optimal performance)
ML Libraries: scikit-learn for preprocessing, xgboost for modeling
Data Processing: pandas and numpy for data manipulation
Visualization: matplotlib for model analysis and feature importance
Model Persistence: joblib and pickle for saving/loading models
Feature Engineering: Custom experience calculation and comprehensive preprocessing pipeline
🔬 Model Training & Development
If you want to retrain the model or understand the development process:

Open the Jupyter notebook:

jupyter notebook Salary_prediction_model.ipynb
Training Process Overview:

Data loading and exploration
Outlier detection and removal
Feature engineering (experience calculation)
Data preprocessing and cleaning
Label encoding and scaling
Model comparison and selection
XGBoost hyperparameter tuning
Model evaluation and feature importance analysis
Key Features of the Training Pipeline:

Comprehensive data cleaning and preprocessing
Feature engineering with domain knowledge
Multiple model evaluation (XGBoost selected for best performance)
Feature importance analysis using XGBoost built-in methods
Proper train/test splitting with stratification
📝 CSV Format for Batch Prediction
Your CSV file should contain exactly these column names:

age,experience,workclass,educational-num,marital-status,occupation,relationship,race,gender,capital-gain,capital-loss,hours-per-week,native-country
39,5,Private,13,Never-married,Adm-clerical,Not-in-family,White,Male,2174,0,40,United-States
📄 License
This project is licensed under the MIT License - see the LICENSE file for details.

🚨 Troubleshooting
Common Issues:

Missing model files: Ensure model.pkl, scaler.pkl, and label_encoders.pkl are in the project directory
CSV upload errors: Check that your CSV has all required columns with exact names
Package conflicts: Try creating a virtual environment and installing fresh dependencies
📞 Support
If you encounter any issues or have questions, please:

Check the troubleshooting section above
Review the input data format requirements
Ensure all model files are present and accessible
You are free to connect with me 🙌.
