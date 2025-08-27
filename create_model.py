import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import joblib

# -------------------------------
# Load dataset
# -------------------------------
data = pd.read_csv("Fertilizer.csv")  # make sure this CSV is in the same folder

# -------------------------------
# Features and target
# -------------------------------
X = data[['Temparature', 'Moisture', 'Soil Type', 'Crop Type', 'Nitrogen', 'Potassium', 'Phosphorous']]
y = data['Fertilizer Name']  # target column

# -------------------------------
# Encode categorical columns
# -------------------------------
le_soil = LabelEncoder()
le_crop = LabelEncoder()
le_target = LabelEncoder()

X['Soil Type'] = le_soil.fit_transform(X['Soil Type'])
X['Crop Type'] = le_crop.fit_transform(X['Crop Type'])
y_encoded = le_target.fit_transform(y)

# -------------------------------
# Train model
# -------------------------------
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y_encoded)

# -------------------------------
# Save model and encoders
# -------------------------------
joblib.dump(model, "fertilizer_model.joblib")
joblib.dump({'Soil Type': le_soil, 'Crop Type': le_crop}, "label_encoders.pkl")
joblib.dump(le_target, "fertilizer_encoder.pkl")

print("✅ fertilizer_model.joblib and encoders created successfully!")