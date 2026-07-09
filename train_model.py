import pandas as pd
import joblib
import re

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

# ===============================
# 1. Load dataset
# ===============================
df = pd.read_csv("data/car_data.csv")

print("📌 Original columns:")
print(df.columns)

# ===============================
# 2. Drop unnecessary column
# ===============================
df = df.drop("name", axis=1)

# ===============================
# 3. CLEAN NUMERIC COLUMNS
# ===============================

# Clean kms_driven (remove commas, text)
df["kms_driven"] = df["kms_driven"].astype(str)
df["kms_driven"] = df["kms_driven"].str.replace(",", "")
df["kms_driven"] = df["kms_driven"].str.extract(r"(\d+)").astype(float)

# Clean Price (remove currency / text)
df["Price"] = df["Price"].astype(str)
df["Price"] = df["Price"].str.replace(",", "")
df["Price"] = df["Price"].str.extract(r"(\d+\.?\d*)").astype(float)

# Clean year (just in case)
df["year"] = pd.to_numeric(df["year"], errors="coerce")

# Drop rows with missing values
df.dropna(inplace=True)

print("📌 Cleaned data preview:")
print(df.head())

# ===============================
# 4. Target & Features
# ===============================
X = df.drop("Price", axis=1)
y = df["Price"]

categorical_cols = ["company", "fuel_type"]
numerical_cols = ["year", "kms_driven"]

# ===============================
# 5. Preprocessing
# ===============================
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols)
    ],
    remainder="passthrough"
)

# ===============================
# 6. Model
# ===============================
model = RandomForestRegressor(
    n_estimators=300,
    random_state=42,
    n_jobs=-1
)

pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", model)
])

# ===============================
# 7. Train-test split
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ===============================
# 8. Train model
# ===============================
pipeline.fit(X_train, y_train)

# ===============================
# 9. Save model
# ===============================
joblib.dump(pipeline, "model/car_price_model.pkl")

print("✅ Model trained and saved successfully!")
