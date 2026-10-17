import pandas as pd
import numpy as np
import re
import os
import joblib
from scipy.sparse import hstack
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

print("🚀 Hybrid Model Training Started...")

# ===============================
# 1️⃣ Load Dataset
# ===============================

df = pd.read_csv("Twitter_DataSET/bot_detection_data.csv")
print("Dataset Loaded:", df.shape)

# ===============================
# 2️⃣ Feature Engineering
# ===============================

# Text-based features
df["tweet_length"] = df["Tweet"].apply(lambda x: len(str(x)))
df["num_urls"] = df["Tweet"].apply(lambda x: len(re.findall(r'http|www|\.com', str(x))))

# Hashtag count
df["Hashtag_Count"] = df["Hashtags"].fillna("").apply(lambda x: len(str(x).split()))

# Convert Verified
df["Verified"] = df["Verified"].map({True: 1, False: 0})

# Account Age (in days)
df["Created At"] = pd.to_datetime(df["Created At"], errors="coerce")
df = df.dropna(subset=["Created At"])
current_date = pd.Timestamp.now()
df["Account_Age_Days"] = (current_date - df["Created At"]).dt.days

# Engagement rate
df["Engagement_Rate"] = df["Retweet Count"] / (df["Follower Count"] + 1)

# Log followers
df["Log_Followers"] = np.log1p(df["Follower Count"])

# ===============================
# 3️⃣ Define Features
# ===============================

metadata_features = [
    "Retweet Count",
    "Mention Count",
    "Log_Followers",
    "Verified",
    "Account_Age_Days",
    "Hashtag_Count",
    "Engagement_Rate",
    "tweet_length",
    "num_urls"
]

X_meta = df[metadata_features]
y = df["Bot Label"]

print("Metadata features:", metadata_features)

# ===============================
# 4️⃣ TF-IDF
# ===============================

vectorizer = TfidfVectorizer(max_features=3000)
X_text = vectorizer.fit_transform(df["Tweet"])

# ===============================
# 5️⃣ Scale Metadata
# ===============================

scaler = StandardScaler()
X_meta_scaled = scaler.fit_transform(X_meta)

# Convert to sparse
from scipy.sparse import csr_matrix
X_meta_sparse = csr_matrix(X_meta_scaled)

# ===============================
# 6️⃣ Combine Text + Metadata
# ===============================

X_combined = hstack([X_text, X_meta_sparse])

# ===============================
# 7️⃣ Train-Test Split
# ===============================

X_train, X_test, y_train, y_test = train_test_split(
    X_combined, y, test_size=0.2, random_state=42, stratify=y
)

# ===============================
# 8️⃣ Train Random Forest
# ===============================

print("🌲 Training Hybrid Random Forest...")

rf = RandomForestClassifier(
    n_estimators=150,
    max_depth=25,
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train, y_train)

print("✅ Training Completed.")

# ===============================
# 9️⃣ Evaluation
# ===============================

y_pred = rf.predict(X_test)

print("\n📈 Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n")
print(confusion_matrix(y_test, y_pred))

# ===============================
# 🔟 Save Everything
# ===============================

os.makedirs("ml_outputs", exist_ok=True)
joblib.dump(rf, "ml_outputs/hybrid_model_v2.pkl")
joblib.dump(vectorizer, "ml_outputs/hybrid_tfidf.pkl")
joblib.dump(scaler, "ml_outputs/hybrid_scaler.pkl")

print("\n🎉 Hybrid Model Saved Successfully!")