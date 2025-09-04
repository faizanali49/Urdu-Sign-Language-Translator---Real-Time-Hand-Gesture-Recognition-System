import tensorflow as tf
import numpy as np
import os
from sklearn.model_selection import train_test_split

# Dataset paths
DATASET_DIR = "dataset"
LANDMARKS_DIR = os.path.join(DATASET_DIR, "landmarks")

# Automatically detect gesture labels based on directories
GESTURES = [d for d in os.listdir(LANDMARKS_DIR) if os.path.isdir(os.path.join(LANDMARKS_DIR, d))]
print(f"Detected Gestures: {GESTURES}")

X, y = [], []

# Load dataset
for idx, gesture in enumerate(GESTURES):
    gesture_path = os.path.join(LANDMARKS_DIR, gesture)
    for file in os.listdir(gesture_path):
        if file.endswith(".npy"):
            data = np.load(os.path.join(gesture_path, file))
            X.append(data)
            y.append(idx)

# Convert to NumPy arrays
X = np.array(X)
y = np.array(y)

# Ensure data has correct shape
if X.ndim == 1:  # If only one feature is found, reshape it
    X = np.expand_dims(X, axis=1)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define Model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(63,)),  # 21 landmarks * 3 (x, y, z)
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(len(GESTURES), activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train Model
model.fit(X_train, y_train, epochs=30, validation_data=(X_test, y_test))

# Save Model
os.makedirs("model", exist_ok=True)
model.save("model/gesture_model_enhanced.h5")

print("Training complete! Model saved to 'model/gesture_model_enhanced.h5'")
