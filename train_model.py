import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# =========================
# LOAD DATA
# =========================

try:
    data = pd.read_csv("House Price India.csv")
    print("✅ Dataset Loaded Successfully")
except Exception as e:
    print("❌ Error loading dataset:", e)
    exit()

# =========================
# FIX EXCEL SERIAL DATE
# =========================

data["Date"] = pd.to_datetime(
    data["Date"],
    origin='1899-12-30',
    unit='D',
    errors='coerce'
)

data["Year_sold"] = data["Date"].dt.year
data["Month_sold"] = data["Date"].dt.month

data.drop("Date", axis=1, inplace=True)

# =========================
# CLEAN DATA
# =========================

print("🔍 Checking data...")
data = data.dropna()
data = data.drop_duplicates()

print("✅ Data cleaned")
print("Number of rows:", len(data))

# =========================
# BASIC MODEL (3 FEATURES)
# =========================

basic_features = [
    "number of bedrooms",
    "number of bathrooms",
    "living area"
]

X_basic = data[basic_features]
y = data["Price"]

Xb_train, Xb_test, yb_train, yb_test = train_test_split(
    X_basic, y, test_size=0.2, random_state=42
)

basic_model = RandomForestRegressor(n_estimators=100, random_state=42)
basic_model.fit(Xb_train, yb_train)

pred_basic = basic_model.predict(Xb_test)

print("\n📊 BASIC MODEL Performance")
print("MAE:", mean_absolute_error(yb_test, pred_basic))
print("R2 Score:", r2_score(yb_test, pred_basic))

# Save Basic Model
joblib.dump(basic_model, "basic_model.pkl")
print("✅ Basic model saved")

# =========================
# ADVANCED MODEL (6 FEATURES)
# =========================

advanced_features = [
    "number of bedrooms",
    "number of bathrooms",
    "living area",
    "number of floors",
    "grade of the house",
    "Built Year"
]

X_adv = data[advanced_features]

Xa_train, Xa_test, ya_train, ya_test = train_test_split(
    X_adv, y, test_size=0.2, random_state=42
)

advanced_model = RandomForestRegressor(n_estimators=100, random_state=42)
advanced_model.fit(Xa_train, ya_train)

pred_adv = advanced_model.predict(Xa_test)

print("\n📊 ADVANCED MODEL Performance")
print("MAE:", mean_absolute_error(ya_test, pred_adv))
print("R2 Score:", r2_score(ya_test, pred_adv))

# Save Advanced Model
joblib.dump(advanced_model, "advanced_model.pkl")
print("✅ Advanced model saved")

print("\n🎉 Training Complete!")
