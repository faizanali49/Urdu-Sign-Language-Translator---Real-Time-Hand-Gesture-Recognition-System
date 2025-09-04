import os
import traceback
import numpy as np
import joblib
import tensorflow as tf
from flask import Flask, request, jsonify
from urdu_data.labels import Label # Import your mapping

# --- Configuration ---
MODEL_PATH = os.path.join("model", "gesture_model_enhanced.h5")
SCALER_PATH = os.path.join("model", "gesture_scaler.joblib")
NUM_FEATURES = 63 # Should match your model's input features (21 landmarks * 3 coords)

# --- Load Model and Scaler ---
try:
    print("Loading TensorFlow model...")
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Model loaded successfully.")

    print("Loading scaler...")
    scaler = joblib.load(SCALER_PATH)
    print("Scaler loaded successfully.")

except Exception as e:
    print(f"Error loading model or scaler: {e}")
    print(traceback.format_exc())
    exit(1)

# --- Load Gesture Labels and Urdu Mapping ---
# Assumes GESTURES list order matches model output indices if not using Label class directly
# If your model predicts indices, ensure this list matches the order used during training's LabelEncoder
GESTURES = sorted(list(Label.label.keys())) + ["delete"] # Ensure 'delete' matches training label if applicable
URDU_LETTER_MAP = Label.label # Your mapping dictionary
print(f"Loaded {len(GESTURES)} gestures.")
print(f"Urdu map loaded with {len(URDU_LETTER_MAP)} entries.")


# --- Flask App Initialization ---
app = Flask(__name__)

# --- API Endpoint for Prediction ---
@app.route('/predict', methods=['POST'])
def predict_gesture():
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    data = request.get_json()

    if 'landmarks' not in data:
        return jsonify({"error": "Missing 'landmarks' in request"}), 400

    landmarks_list = data['landmarks']

    # Validate landmark data length
    if len(landmarks_list) != NUM_FEATURES:
        return jsonify({"error": f"Invalid number of landmarks. Expected {NUM_FEATURES}, got {len(landmarks_list)}"}), 400

    try:
        # Convert to NumPy array and reshape for scaler/model
        landmarks_np = np.array(landmarks_list, dtype=np.float32).reshape(1, -1)

        # Scale the landmarks using the loaded scaler
        landmarks_scaled = scaler.transform(landmarks_np)

        # Predict using the loaded model
        prediction = model.predict(landmarks_scaled, verbose=0)
        predicted_index = np.argmax(prediction)
        confidence = float(np.max(prediction)) # Get confidence score

        # Map index to gesture label
        if 0 <= predicted_index < len(GESTURES):
            gesture_label = GESTURES[predicted_index]
        else:
             return jsonify({"error": f"Predicted index {predicted_index} out of bounds"}), 500


        # Map gesture label to Urdu character
        urdu_char = URDU_LETTER_MAP.get(gesture_label, None) # Returns None if not in map
        action = "add" # Default action

        # Handle special cases like 'delete'
        if gesture_label == "delete":
            action = "delete"
            urdu_char = None # No specific character for delete

        elif urdu_char is None and gesture_label != "delete":
             # Handle cases where a gesture might be predicted but not in the map
             print(f"Warning: Predicted gesture '{gesture_label}' not found in URDU_LETTER_MAP.")
             # Decide how to handle this: return error, empty char, or the label itself?
             action = "ignore" # Or 'unknown'


        response = {
            "predicted_gesture": gesture_label,
            "urdu_char": urdu_char,
            "action": action, # 'add', 'delete', 'ignore', 'unknown' etc.
            "confidence": confidence
        }
        return jsonify(response), 200

    except Exception as e:
        print(f"Error during prediction: {e}")
        print(traceback.format_exc())
        return jsonify({"error": "Internal server error during prediction"}), 500

# --- Run the Flask App ---
if __name__ == '__main__':
    # Run on all available network interfaces (0.0.0.0)
    # Make sure your firewall allows connections to this port (e.g., 5000)
    print("Starting Flask server on http://0.0.0.0:5000")
    app.run(host='0.0.0.0', port=5000, debug=False) # Set debug=False for production/general use