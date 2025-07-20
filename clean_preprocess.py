import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import joblib
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier, plot_importance

# Load dataset
df = pd.read_csv("adult 3.csv")

# Remove outliers
df = df[(df['age'] <= 65) & (df['age'] >= 17)]
df = df[(df['educational-num'] <= 16) & (df['educational-num'] >= 5)]

# Add experience feature
df['experience'] = df['age'] - df['educational-num'] - 6

# Rearrange columns
cols = df.columns.tolist()
cols.insert(1, cols.pop(cols.index('experience')))
df = df[cols]

# Handle missing values
for col in ['workclass', 'native-country', 'occupation']:
    df[col].replace({'?': 'Others'}, inplace=True)

# Remove non-impacting categories
df = df[~df['workclass'].isin(['Without-pay', 'Never-worked'])]
df = df[~df['education'].isin(['Preschool', '1st-4th', '5th-6th'])]

# Drop unnecessary columns
df.drop(columns=['education', 'fnlwgt'], inplace=True)

# Save cleaned data
df.to_csv("Employee_details.csv", index=False)

# Apply Label Encoding
categorical_cols = ['workclass', 'marital-status', 'occupation', 'relationship',
                    'race', 'gender', 'native-country', 'income']
label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Save label encoders
with open("label_encoders.pkl", "wb") as f:
    pickle.dump(label_encoders, f)

# Feature matrix and target
x = df.drop(columns=['income'])
y = df['income']

# MinMax Scaling
scaler = MinMaxScaler()
x = scaler.fit_transform(x)

# Save scaler
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

# Train/test split
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=23, stratify=y)

# Define models for comparison
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "XGBoost": XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=6, eval_metric='logloss', use_label_encoder=False, random_state=42)
}

# Train and evaluate each model
results = {}

for name, model in models.items():
    model.fit(xtrain, ytrain)
    ypred = model.predict(xtest)
    acc = accuracy_score(ytest, ypred)
    results[name] = acc
    print(f"\n=== {name} ===")
    print("Accuracy:", acc)
    print(classification_report(ytest, ypred))

# Plot accuracy comparison
plt.figure(figsize=(10, 6))
plt.bar(results.keys(), results.values(), color='teal')
plt.title("Model Comparison: Accuracy")
plt.ylabel("Accuracy")
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# Train final XGBoost model and save
final_model = models["XGBoost"]
joblib.dump(final_model, "model.pkl")

# Plot XGBoost feature importance
final_model.get_booster().feature_names = df.columns[:-1].tolist()

plt.figure(figsize=(10, 6))
plot_importance(final_model, max_num_features=13)
plt.title("XGBoost Feature Importance")
plt.show()
