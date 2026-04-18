import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle

# Load dataset
data = pd.read_csv("hospital dataset.csv")

# Clean column names
data.columns = data.columns.str.strip().str.lower()

# 🔍 Print columns (for debugging once)
print("Columns in dataset:", data.columns)

# Remove missing values
data = data.dropna()

# ✅ Handle categorical safely
data['sex'] = data['sex'].astype(str).str.lower().map({'male': 0, 'female': 1})
data['smoker'] = data['smoker'].astype(str).str.lower().map({'no': 0, 'yes': 1})

# Remove rows where mapping failed
data = data.dropna(subset=['sex', 'smoker'])

# Convert numeric columns safely
data['age'] = pd.to_numeric(data['age'], errors='coerce')
data['bmi'] = pd.to_numeric(data['bmi'], errors='coerce')
data['children'] = pd.to_numeric(data['children'], errors='coerce')
data['charges'] = pd.to_numeric(data['charges'], errors='coerce')

# Drop any remaining NaN
data = data.dropna()

# ✅ FINAL FEATURES (same as app.py)
X = data[['age', 'sex', 'bmi', 'children', 'smoker']]
y = data['charges']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Save model
pickle.dump(model, open("model.pkl", "wb"))

print("✅ Model trained successfully!")