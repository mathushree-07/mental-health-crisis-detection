import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)


# ---------------------------------------------------
# 1. LOAD DATASET
# ---------------------------------------------------

file_path = "data/Student Depression Dataset.csv"

df = pd.read_csv(file_path)

print("Dataset loaded successfully!")
print("Original shape:", df.shape)


# ---------------------------------------------------
# 2. REMOVE ID COLUMN
# ---------------------------------------------------

df = df.drop(columns=["id"])

print("ID column removed.")


# ---------------------------------------------------
# 3. REMOVE INVALID FINANCIAL STRESS VALUES
# ---------------------------------------------------

df = df[df["Financial Stress"].astype(str) != "?"]

print("Invalid Financial Stress records removed.")
print("Shape after cleaning:", df.shape)


# ---------------------------------------------------
# 4. CONVERT NUMERIC COLUMNS
# ---------------------------------------------------

numeric_columns = [
    "Age",
    "Academic Pressure",
    "Work Pressure",
    "CGPA",
    "Study Satisfaction",
    "Job Satisfaction",
    "Work/Study Hours",
    "Financial Stress"
]

for column in numeric_columns:
    df[column] = pd.to_numeric(df[column], errors="coerce")


# ---------------------------------------------------
# 5. CONVERT YES / NO COLUMNS
# ---------------------------------------------------

binary_columns = [
    "Have you ever had suicidal thoughts ?",
    "Family History of Mental Illness"
]

for column in binary_columns:
    df[column] = df[column].map({
        "Yes": 1,
        "No": 0
    })


# ---------------------------------------------------
# 6. CONVERT SLEEP DURATION
# ---------------------------------------------------

sleep_mapping = {
    "Less than 5 hours": 1,
    "5-6 hours": 2,
    "7-8 hours": 3,
    "More than 8 hours": 4
}

df["Sleep Duration"] = df["Sleep Duration"].map(sleep_mapping)


# ---------------------------------------------------
# 7. CONVERT DIETARY HABITS
# ---------------------------------------------------

diet_mapping = {
    "Unhealthy": 1,
    "Moderate": 2,
    "Healthy": 3
}

df["Dietary Habits"] = df["Dietary Habits"].map(diet_mapping)


# ---------------------------------------------------
# 8. REMOVE INVALID / MISSING VALUES
# ---------------------------------------------------

df = df.dropna()

print("Missing/invalid records removed.")
print("Final cleaned shape:", df.shape)


# ---------------------------------------------------
# 9. SAVE CLEANED DATASET
# ---------------------------------------------------

df.to_csv("data/cleaned_dataset.csv", index=False)

print("Cleaned dataset saved successfully!")


# ---------------------------------------------------
# 10. SEPARATE FEATURES AND TARGET
# ---------------------------------------------------

X = df.drop(columns=["Depression"])
y = df["Depression"]


# ---------------------------------------------------
# 11. DEFINE COLUMN TYPES
# ---------------------------------------------------

categorical_columns = [
    "Gender",
    "City",
    "Profession",
    "Degree"
]

numerical_columns = [
    "Age",
    "Academic Pressure",
    "Work Pressure",
    "CGPA",
    "Study Satisfaction",
    "Job Satisfaction",
    "Sleep Duration",
    "Dietary Habits",
    "Work/Study Hours",
    "Financial Stress",
    "Have you ever had suicidal thoughts ?",
    "Family History of Mental Illness"
]


# ---------------------------------------------------
# 12. PREPROCESSING
# ---------------------------------------------------

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_columns
        ),
        (
            "numerical",
            StandardScaler(),
            numerical_columns
        )
    ]
)


# ---------------------------------------------------
# 13. CREATE LOGISTIC REGRESSION MODEL
# ---------------------------------------------------

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000
            )
        )
    ]
)


# ---------------------------------------------------
# 14. TRAIN / TEST SPLIT
# ---------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))


# ---------------------------------------------------
# 15. TRAIN MODEL
# ---------------------------------------------------

print("\nTraining Logistic Regression model...")

model.fit(X_train, y_train)

print("Model training completed!")


# ---------------------------------------------------
# 16. MAKE PREDICTIONS
# ---------------------------------------------------

y_pred = model.predict(X_test)


# ---------------------------------------------------
# 17. EVALUATE MODEL
# ---------------------------------------------------

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)


print("\n==============================")
print("MODEL PERFORMANCE")
print("==============================")

print("Accuracy :", round(accuracy, 4))
print("Precision:", round(precision, 4))
print("Recall   :", round(recall, 4))
print("F1 Score :", round(f1, 4))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# ---------------------------------------------------
# 18. SAVE MODEL
# ---------------------------------------------------

joblib.dump(
    model,
    "depression_model.pkl"
)

print("\nModel saved as depression_model.pkl")

print("\n==============================")
print("TRAINING COMPLETED SUCCESSFULLY")
print("==============================")