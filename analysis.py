"""
Medical Insurance Cost Prediction using Multiple Linear Regression
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# -----------------------------------------------------------
# TASK 1: DATA UNDERSTANDING
# -----------------------------------------------------------
df = pd.read_csv("insurance.csv")

print("="*60)
print("TASK 1: DATA UNDERSTANDING")
print("="*60)
print("\nFirst five records:")
print(df.head())

print("\nDataset shape:", df.shape)
print("\nColumn data types:")
print(df.dtypes)

numerical_features = ['age', 'bmi', 'children']
categorical_features = ['sex', 'smoker', 'region']
target_variable = 'charges'

print("\nNumerical features   :", numerical_features)
print("Categorical features :", categorical_features)
print("Target variable      :", target_variable)

# -----------------------------------------------------------
# TASK 2: DATA PREPROCESSING
# -----------------------------------------------------------
print("\n" + "="*60)
print("TASK 2: DATA PREPROCESSING")
print("="*60)

print("\nMissing values per column:")
print(df.isnull().sum())

df_encoded = df.copy()
df_encoded['sex'] = df_encoded['sex'].map({'male': 0, 'female': 1})
df_encoded['smoker'] = df_encoded['smoker'].map({'no': 0, 'yes': 1})
df_encoded = pd.get_dummies(df_encoded, columns=['region'], drop_first=True)

# ensure boolean dummy columns are numeric (0/1)
region_dummy_cols = [c for c in df_encoded.columns if c.startswith('region_')]
df_encoded[region_dummy_cols] = df_encoded[region_dummy_cols].astype(int)

print("\nEncoded dataset preview:")
print(df_encoded.head())

X = df_encoded.drop('charges', axis=1)
y = df_encoded['charges']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nTraining set size: {X_train.shape[0]} records")
print(f"Testing set size : {X_test.shape[0]} records")

# -----------------------------------------------------------
# TASK 3: MODEL DEVELOPMENT
# -----------------------------------------------------------
print("\n" + "="*60)
print("TASK 3: MODEL DEVELOPMENT")
print("="*60)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\nModel Intercept:", model.intercept_)
print("\nModel Coefficients:")
for feature, coef in zip(X.columns, model.coef_):
    print(f"  {feature:15s}: {coef:.3f}")

print("\nSample predictions (first 5 test records):")
comparison = pd.DataFrame({'Actual': y_test.values[:5], 'Predicted': y_pred[:5]})
print(comparison)

# -----------------------------------------------------------
# TASK 4: MODEL EVALUATION
# -----------------------------------------------------------
print("\n" + "="*60)
print("TASK 4: MODEL EVALUATION")
print("="*60)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"\nMean Absolute Error (MAE) : {mae:,.2f}")
print(f"Mean Squared Error  (MSE) : {mse:,.2f}")
print(f"Root Mean Squared Error   : {rmse:,.2f}")
print(f"R2 Score                  : {r2:.4f}")

# Actual vs Predicted scatter plot
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.6, color='steelblue', edgecolor='k')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
         'r--', lw=2, label='Perfect Prediction Line')
plt.xlabel('Actual Charges ($)')
plt.ylabel('Predicted Charges ($)')
plt.title('Actual vs Predicted Insurance Charges')
plt.legend()
plt.tight_layout()
plt.savefig('actual_vs_predicted.png', dpi=150)
print("\nSaved plot: actual_vs_predicted.png")

# Feature importance (coefficient magnitude) plot
plt.figure(figsize=(8, 5))
coef_df = pd.Series(model.coef_, index=X.columns).sort_values()
coef_df.plot(kind='barh', color='teal')
plt.xlabel('Coefficient Value')
plt.title('Feature Coefficients (Impact on Charges)')
plt.tight_layout()
plt.savefig('feature_coefficients.png', dpi=150)
print("Saved plot: feature_coefficients.png")

print("\nObservations:")
print("1. 'smoker' has by far the largest positive coefficient, meaning")
print("   smoking status is the strongest driver of higher insurance charges.")
print("2. Age and BMI also show positive relationships with charges, consistent")
print("   with higher medical risk in older / higher-BMI individuals.")
print(f"3. The model explains about {r2*100:.1f}% of the variance in charges (R2),")
print("   with the remaining unexplained variance likely due to non-linear")
print("   interactions (e.g., smoker x BMI) that plain linear regression can't capture.")
