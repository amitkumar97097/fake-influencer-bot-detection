import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load dataset
df = pd.read_csv(r"C:\Users\udayk\OneDrive\Desktop\FakeInfluencerProject\Twitter_DataSET\bot_detection_data.csv")

print("Dataset Loaded Successfully!")
print(df.head())
print(df.info())

# Drop missing values
df = df.dropna()

# Selecting important features
X = df[["Tweet", "Retweet Count", "Mention Count", "Follower Count", "Verified"]]
y = df["Bot Label"]

# Convert boolean 'Verified' to int
df["Verified"] = df["Verified"].astype(int)

# TF-IDF for Tweet text
transformer = ColumnTransformer(
    transformers=[
        ('tfidf', TfidfVectorizer(max_features=5000), 'Tweet')
    ],
    remainder='passthrough'
)

# --- SVM Model ---
svm_model = Pipeline([
    ('transform', transformer),
    ('svm', SVC(kernel='linear'))
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

svm_model.fit(X_train, y_train)
svm_pred = svm_model.predict(X_test)

print("\nSVM Accuracy:", accuracy_score(y_test, svm_pred))
print("\nSVM Classification Report:\n", classification_report(y_test, svm_pred))

joblib.dump(svm_model, "svm_model.pkl")


# --- RANDOM FOREST Model ---
rf_model = Pipeline([
    ('transform', transformer),
    ('rf', RandomForestClassifier())
])

rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)

print("\nRandom Forest Accuracy:", accuracy_score(y_test, rf_pred))
print("\nRandom Forest Classification Report:\n", classification_report(y_test, rf_pred))

joblib.dump(rf_model, "rf_model.pkl")

print("\nModels Saved Successfully!")
