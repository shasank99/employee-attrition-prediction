import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


# ==========================================
# 1. Load Dataset
# ==========================================

file_path = "data/real/WA_Fn-UseC_-HR-Employee-Attrition.csv"

data = pd.read_csv(file_path)

print("Dataset loaded successfully!")
print("Dataset shape:", data.shape)


# ==========================================
# 2. Select Useful Features
# ==========================================

features = [
    "Age",
    "MonthlyIncome",
    "JobSatisfaction",
    "YearsAtCompany",
    "OverTime",
    "DistanceFromHome"
]

X = data[features].copy()

y = data["Attrition"].copy()


# ==========================================
# 3. Convert Text to Numbers
# ==========================================

X["OverTime"] = X["OverTime"].map({
    "Yes": 1,
    "No": 0
})

y = y.map({
    "Yes": 1,
    "No": 0
})


# ==========================================
# 4. Check Missing Values
# ==========================================

print("\nMissing values:")
print(X.isnull().sum())


# ==========================================
# 5. Split Dataset
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# ==========================================
# 6. Create Machine Learning Model
# ==========================================

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced"
)


# ==========================================
# 7. Train Model
# ==========================================

print("\nTraining model...")

model.fit(X_train, y_train)

print("Training completed!")


# ==========================================
# 8. Make Predictions
# ==========================================

predictions = model.predict(X_test)


# ==========================================
# 9. Evaluate Model
# ==========================================

accuracy = accuracy_score(y_test, predictions)

print("\nModel Accuracy:", round(accuracy * 100, 2), "%")

print("\nClassification Report:")
print(classification_report(y_test, predictions))


# ==========================================
# 10. Save Model
# ==========================================

joblib.dump(
    model,
    "models/attrition_real_model.pkl"
)

print("\nModel saved successfully!")
print("Location: models/attrition_real_model.pkl")
