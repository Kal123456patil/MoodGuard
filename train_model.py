import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

FILE_NAME = "mood_data.csv"

df = pd.read_csv(FILE_NAME)

# Convert Exercise Yes/No to 1/0
df["Exercise"] = df["Exercise"].map({"No": 0, "Yes": 1})

# ✅ Create label (target)
# 1 = High Risk, 0 = Low Risk
df["BurnoutRisk"] = 0
df.loc[
    (df["Mood"] <= 4) |
    (df["SleepHours"] < 6) |
    (df["Stress"] >= 7) |
    (df["Energy"] <= 4) |
    (df["ScreenTime"] >= 8),
    "BurnoutRisk"
] = 1

X = df[["Mood", "SleepHours", "Stress", "Energy", "ScreenTime", "Exercise"]]
y = df["BurnoutRisk"]

# Need minimum data
if len(df) < 10:
    print("❌ Add at least 10 entries in mood_data.csv to train ML model.")
    exit()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

joblib.dump(model, "burnout_model.pkl")
print("✅ Model trained and saved as burnout_model.pkl")
