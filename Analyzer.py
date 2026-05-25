import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# =========================================
# LOAD DATASET
# =========================================

df = pd.read_csv("./drift_data.csv")

# =========================================
# FEATURES AND TARGET
# =========================================

X = df.drop("cognitive_drift", axis=1)

y = df["cognitive_drift"]

# =========================================
# TRAIN TEST SPLIT
# =========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# =========================================
# MODEL CREATION
# =========================================

model = LogisticRegression()

# =========================================
# MODEL TRAINING
# =========================================

model.fit(X_train, y_train)

# =========================================
# MODEL PREDICTIONS
# =========================================

predictions = model.predict(X_test)

# =========================================
# MODEL ACCURACY
# =========================================

accuracy = accuracy_score(y_test, predictions)

# =========================================
# SYSTEM HEADER
# =========================================

print("\n===================================")
print("      COGNITIVE DRIFT ANALYZER")
print("===================================")

print(f"\nModel Accuracy: {accuracy:.2f}")

# =========================================
# NEW USER ANALYSIS
# =========================================

new_user = [[45, 52, 68, 81, 9, 74, 86]]

prediction = model.predict(new_user)

probability = model.predict_proba(new_user)

drift_score = probability[0][1] * 100

stability_score = 100 - drift_score

# =========================================
# STATUS OUTPUT
# =========================================

print("\nBehavioral Analysis")
print("-----------------------------------")

if prediction[0] == 1:
    print("Status: Cognitive Drift Detected")

else:
    print("Status: Stable Cognitive State")

print(f"Drift Score: {drift_score:.2f}%")

print(f"Stability Index: {stability_score:.2f}%")

# =========================================
# SIGNAL BREAKDOWN
# =========================================

print("\nBehavioral Signal Breakdown")
print("-----------------------------------")

signals = {

    "Response Latency Variance": 45,

    "Task Completion Decay": 52,

    "Focus Fragmentation": 68,

    "Late Night Activity Ratio": 81,

    "Micro Goal Failure Streak": 9,

    "Interaction Withdrawal": 74,

    "Consistency Deviation": 86
}

for key, value in signals.items():

    print(f"{key}: {value}")

# =========================================
# DRIFT INTERPRETATION
# =========================================

print("\nSilent Drift Detection")
print("-----------------------------------")

if drift_score > 70:

    print("Severe behavioral deviation patterns detected.")

elif drift_score > 40:

    print("Moderate cognitive drift emerging.")

else:

    print("Behavioral stability maintained.")

print("\n===================================")