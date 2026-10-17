import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

print("🚀 Script started...")

# ===============================
# 1️⃣ Load Dataset
# ===============================

DATA_PATH = "Twitter_DataSET/bot_detection_data.csv"

if not os.path.exists(DATA_PATH):
    print("❌ Dataset not found at:", DATA_PATH)
    exit()

df = pd.read_csv(DATA_PATH)
print("✅ Dataset loaded successfully.")
print("Dataset shape:", df.shape)

# ===============================
# 2️⃣ Basic Cleaning & Conversion
# ===============================

df["Verified"] = df["Verified"].map({True: 1, False: 0})
df["Created At"] = pd.to_datetime(df["Created At"], errors="coerce")
df = df.dropna(subset=["Created At"])

# ===============================
# 3️⃣ Feature Engineering
# ===============================

print("⚙️ Performing feature engineering...")

current_date = pd.Timestamp.now()
df["Account_Age_Days"] = (current_date - df["Created At"]).dt.days
df["Hashtag_Count"] = df["Hashtags"].fillna("").apply(lambda x: len(str(x).split()))
df["Engagement_Rate"] = df["Retweet Count"] / (df["Follower Count"] + 1)
df["Log_Followers"] = np.log1p(df["Follower Count"])

# ===============================
# 4️⃣ Select Features
# ===============================

features = [
    "Retweet Count",
    "Mention Count",
    "Log_Followers",
    "Verified",
    "Account_Age_Days",
    "Hashtag_Count",
    "Engagement_Rate"
]

X = df[features]
y = df["Bot Label"]

print("📊 Features selected:", features)

# ===============================
# 5️⃣ Train-Test Split
# ===============================

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("✅ Train-Test split completed.")

# ===============================
# 6️⃣ Scaling
# ===============================

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

os.makedirs("ml_outputs", exist_ok=True)
joblib.dump(scaler, "ml_outputs/metadata_scaler.pkl")
print("💾 Scaler saved.")

# ===============================
# 7️⃣ Train Random Forest
# ===============================

print("🌲 Training Random Forest...")

rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=None,
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train_scaled, y_train)

print("✅ Training completed.")
joblib.dump(rf, "ml_outputs/metadata_model.pkl")
print("💾 Model saved.")

# ===============================
# 8️⃣ Evaluation
# ===============================

print("\n📈 Model Evaluation:")

y_pred = rf.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:\n")
print(confusion_matrix(y_test, y_pred))

# ===============================
# 9️⃣ Feature Importance
# ===============================

print("\n🔥 Feature Importances:")
for name, importance in zip(features, rf.feature_importances_):
    print(f"{name}: {importance:.4f}")

print("\n🎉 Phase-2 Metadata Model Training Completed Successfully!")