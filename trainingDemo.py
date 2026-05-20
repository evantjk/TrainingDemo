# ==========================================
# 1. IMPORT REQUIRED LIBRARIES
# ==========================================
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# ==========================================
# 2. LOAD & PREPROCESS THE CSV DATA (Step 2)
# ==========================================
print("Loading data from CSV...")
# Read the file directly from your folder
# ==========================================
# 2. LOAD & PREPROCESS THE CSV DATA (Step 2)
# ==========================================
print("Loading data from CSV...")
# CHANGED: Updated to your new filename
df = pd.read_csv('DataDemo.csv') 

# Drop columns that are completely useless for pattern recognition (like unique names/IDs)
# Also separate our target answer ('Requires_Intervention') from our tracking features (X)
X = df.drop(columns=['Employee_ID', 'Requires_Intervention'])
y = df['Requires_Intervention']

# Convert text-based categories (Gender, Department, Remote_Status) into columns of 0s and 1s.
X = pd.get_dummies(X, columns=['Gender', 'Department', 'Remote_Status'], drop_first=True)

# ==========================================
# 3. SPLIT THE DATASET (Step 3)
# ==========================================
# Reserve 20% of our employee rows for testing evaluation
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ==========================================
# 4. TRAIN THE MODEL (Steps 4 & 5)
# ==========================================
print("Training Random Forest model on screening data...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
print("Training complete!\n")

# ==========================================
# 5. EVALUATE PERFORMANCE (Step 6)
# ==========================================
predictions = model.predict(X_test)

print("--- Corporate Screening Model Metrics ---")
print(f"Overall Classification Accuracy: {accuracy_score(y_test, predictions) * 100:.2f}%\n")
print("Detailed Breakdown:")
print(classification_report(y_test, predictions, target_names=["Low Risk (0)", "Needs Intervention (1)"]))
# ==========================================
# 7. EXTRACT KEY RISK DRIVERS
# ==========================================
import numpy as np

# Get the importance math from the model
importances = model.feature_importances_
feature_names = X.columns

# Pair them up and sort them from highest to lowest
feature_importance_df = pd.DataFrame({
    'Workplace Factor': feature_names,
    'Impact Score': importances
}).sort_values(by='Impact Score', ascending=False)

print("\n--- 🚨 Top Corporate Risk Drivers 🚨 ---")
print("This tells you what most heavily influences employee mental health risk:")
print(feature_importance_df.to_string(index=False, formatters={'Impact Score': '{:,.2%}'.format}))
# ==========================================
# 8. SAVE THE MODEL & PREDICT NEW EMPLOYEES
# ==========================================
import joblib

# Save the trained model and the exact column layout to disk
joblib.dump(model, 'mental_health_model.pkl')
joblib.dump(X.columns, 'model_columns.pkl')
print("\n💾 Model and schema saved successfully to your folder!")

# --- SIMULATING NEW INCOMING SURVEYS ---
# Let's create two brand-new hypothetical employees to test our model:
# Employee A: High stress, poor sleep, low work-life balance, works 55 hours.
# Employee B: Low stress, great sleep, excellent work-life balance, works 38 hours.

print("\n🔮 Simulating real-time screening analysis...")

def predict_employee_risk(raw_data):
    # Load model and columns
    trained_model = joblib.load('mental_health_model.pkl')
    model_cols = joblib.load('model_columns.pkl')
    
    # Convert input to a DataFrame and process categories just like training data
    input_df = pd.DataFrame([raw_data])
    input_df = pd.get_dummies(input_df)
    
    # Align columns with what the model expects (fills missing columns with 0)
    input_df = input_df.reindex(columns=model_cols, fill_value=0)
    
    # Predict!
    prediction = trained_model.predict(input_df)[0]
    probability = trained_model.predict_proba(input_df)[0][1]
    
    return prediction, probability

# Employee A Data (Expected to flag as high risk)
employee_a = {
    'Age': 29, 'Years_at_Company': 2, 'Work_Hours_Per_Week': 55,
    'Screening_Stress_Level': 5, 'Screening_Sleep_Quality': 4,
    'Screening_Work_Life_Balance': 1, 'Screening_Anxiety_Frequency': 5,
    'Screening_Overwhelmed_Frequency': 4, 'Gender': 'Male',
    'Department': 'Engineering', 'Remote_Status': 'On-site'
}

# Employee B Data (Expected to flag as low risk)
employee_b = {
    'Age': 34, 'Years_at_Company': 5, 'Work_Hours_Per_Week': 38,
    'Screening_Stress_Level': 1, 'Screening_Sleep_Quality': 1,
    'Screening_Work_Life_Balance': 5, 'Screening_Anxiety_Frequency': 1,
    'Screening_Overwhelmed_Frequency': 1, 'Gender': 'Female',
    'Department': 'HR', 'Remote_Status': 'Hybrid'
}

# Run the live predictions
for name, data in [("Employee A (High Distress Profile)", employee_a), ("Employee B (Healthy Profile)", employee_b)]:
    flag, risk_percentage = predict_employee_risk(data)
    status = "🚨 REQUIRES INTERVENTION" if flag == 1 else "✅ LOW RISK / STABLE"
    print(f"\nResults for {name}:")
    print(f" -> Status: {status}")
    print(f" -> Burnout/Distress Probability: {risk_percentage * 100:.1f}%")