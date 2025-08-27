import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import joblib
import json

# Load dataset
dataset = pd.read_csv("Fertilizer.csv")

# Extract target variable
y = dataset['Fertilizer Name']
X = dataset[['Temparature', 'Moisture', 'Soil Type', 'Crop Type', 'Nitrogen', 'Potassium', 'Phosphorous']]

# Encode categorical variables
label_encoders = {}
categorical_columns = ['Soil Type', 'Crop Type']
for col in categorical_columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = le

# Encode target (fertilizer names)
fertilizer_encoder = LabelEncoder()
y_encoded = fertilizer_encoder.fit_transform(y)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Build Neural Network
model = Sequential([
    Dense(64, input_dim=X_train.shape[1], activation='relu'),
    Dense(32, activation='relu'),
    Dense(len(fertilizer_encoder.classes_), activation='softmax')
])

# Compile model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train
model.fit(X_train, y_train, epochs=30, batch_size=32, validation_split=0.2)

# Save model & encoders
model.save("fertilizer_model.h5")
joblib.dump(label_encoders, "label_encoders.pkl")
joblib.dump(fertilizer_encoder, "fertilizer_encoder.pkl")

# Save available soil and crop options
options = {
    "soil_types": sorted(dataset["Soil Type"].unique().tolist()),
    "crop_types": sorted(dataset["Crop Type"].unique().tolist())
}
with open("options.json", "w") as f:
    json.dump(options, f)

print("✅ Model, encoders, and options saved successfully!")
