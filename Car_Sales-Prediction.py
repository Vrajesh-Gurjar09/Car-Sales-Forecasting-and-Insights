import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error

# STEP 1: Load Dataset
df = pd.read_csv("FS-Data_80475.csv")

# STEP 2: Preprocessing  Encode categorical columns
le_account = LabelEncoder()
df['account_id_encoded'] = le_account.fit_transform(df['account_id'])

le_name = LabelEncoder()
df['english_name_encoded'] = le_name.fit_transform(df['english_name'])

# Features (X) and Target (y)
X = df[['account_id_encoded', 'english_name_encoded', 'year', 'month']]
y = df['monthly_value']

# STEP 3: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=42)

# STEP 4: Train Random Forest Model
model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# STEP 5: Evaluate Model
y_pred = model.predict(X_test)

print("✅ Model Evaluation")
print("R2 Score:", r2_score(y_test, y_pred))
print("MAE:", mean_absolute_error(y_test, y_pred))
print("-" * 60)

# STEP 6: Predict Next 3 Months
future_year = 2025
future_months = [1, 2, 3]

future_data = []
for acc_id in df['account_id'].unique():
    acc_enc = le_account.transform([acc_id])[0]
    names_for_acc = df[df['account_id'] == acc_id]['english_name'].unique()
    for name in names_for_acc:
        name_enc = le_name.transform([name])[0]
        for m in future_months:
            future_data.append([acc_enc, name_enc, future_year, m, acc_id, name])

future_df = pd.DataFrame(
    future_data,
    columns=['account_id_encoded', 'english_name_encoded', 'year', 'month', 'account_id', 'english_name']
)

future_df['predicted_monthly_value'] = model.predict(
    future_df[['account_id_encoded', 'english_name_encoded', 'year', 'month']]
)

# STEP 7: Save Forecast & Show Sample
forecast = future_df[['account_id', 'english_name', 'year', 'month', 'predicted_monthly_value']]
forecast.to_csv("forecast_random_forest.csv", index=False)

print("\n📊 Sample Future Predictions:")
print(forecast.head(10))
print(f"\nTotal rows predicted: {len(forecast)}")

# STEP 8: Merge Historical + Forecast
historical = df.copy()
historical['type'] = "historical"
historical = historical[['account_id', 'english_name', 'year', 'month', 'monthly_value', 'type']]

forecast = forecast.rename(columns={'predicted_monthly_value': 'monthly_value'})
forecast['type'] = "forecast"

# Combine both
combined = pd.concat([historical, forecast], ignore_index=True)
combined.to_csv("historical_plus_forecast.csv", index=False)
print(combined.head(15))


