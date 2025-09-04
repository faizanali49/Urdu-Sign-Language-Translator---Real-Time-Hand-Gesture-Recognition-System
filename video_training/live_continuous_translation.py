# live_continuous_translation.py
import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf # Keep TF for TFLite Interpreter
import time
from collections import deque
import os

# Import utility functions
from utils import normalize_landmarks, load_class_names # Assuming utils.py is accessible

# --- Configuration ---
MODEL_SAVE_DIR = "continuous_model" # Matches training script output
TFLITE_MODEL_PATH = os.path.join(MODEL_SAVE_DIR, "continuous_gesture_model.tflite")
CLASSES_PATH = "training/class_labels_continuous.json" # Matches training script output

# Prediction Parameters (tune these)
SEQUENCE_LENGTH = 40  # ** MUST MATCH the sequence length used during training! **
PREDICTION_THRESHOLD = 0.7 # Minimum confidence to consider a prediction valid
PREDICTION_INTERVAL = 5 # Predict every N frames (lower for faster response, higher for less computation)
SMOOTHING_WINDOW_SIZE = 5 # Number of past predictions to average for smoothing

# --- Load Model and Classes ---
print("Loading TFLite model and class names...")
if not os.path.exists(TFLITE_MODEL_PATH):
    print(f"Error: TFLite model not found at {TFLITE_MODEL_PATH}")
    exit()
if not os.path.exists(CLASSES_PATH):
    print(f"Error: Class labels file not found at {CLASSES_PATH}")
    exit()

try:
    interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print("TFLite model loaded.")

    # Get expected input shape and dtype (includes SEQUENCE_LENGTH)
    MODEL_SEQUENCE_LENGTH = input_details[0]['shape'][1]
    INPUT_DTYPE = input_details[0]['dtype']
    if SEQUENCE_LENGTH != MODEL_SEQUENCE_LENGTH:
         print(f"Warning: Configured SEQUENCE_LENGTH ({SEQUENCE_LENGTH}) does not match model input ({MODEL_SEQUENCE_LENGTH}). Using model's length.")
         SEQUENCE_LENGTH = MODEL_SEQUENCE_LENGTH

except Exception as e:
    print(f"Error loading TFLite model: {e}")
    exit()

class_names = load_class_names(CLASSES_PATH)
if not class_names:
    print("Error: Could not load class names.")
    exit()
print(f"Loaded {len(class_names)} classes: {class_names}")


# --- Initialize MediaPipe and Camera ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.6,
                       min_tracking_confidence=0.6)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    hands.close()
    exit()
print("Webcam opened.")

# --- Initialize Buffers and State ---
# Buffer to hold the sequence of raw landmarks for one hand
landmark_buffer = deque(maxlen=SEQUENCE_LENGTH)
# Buffer for smoothing prediction probabilities
prediction_probs_buffer = deque(maxlen=SMOOTHING_WINDOW_SIZE)

frame_count = 0
last_prediction_time = time.time()
display_gesture = "Initializing..."
display_confidence = 0.0

# Fill the initial buffer with zeros (or specific mask value if model expects it)
# MASK_VALUE = 0.0 # Should match training if Masking layer used 0.0
# for _ in range(SEQUENCE_LENGTH):
#     landmark_buffer.append(np.zeros(63, dtype=np.float32)) # Use float32 matching normalization output

# --- Main Loop ---
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break

    frame = cv2.flip(frame, 1)
    annotated_frame = frame.copy() # Draw on this copy

    # Process frame
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb_frame.flags.writeable = False
    results = hands.process(rgb_frame)
    rgb_frame.flags.writeable = True

    current_landmarks_raw = np.zeros(63, dtype=np.float32) # Placeholder for this frame
    hand_detected_this_frame = False

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0] # Assuming max_hands=1
        # Extract raw landmarks for this frame
        current_landmarks_raw = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark], dtype=np.float32).flatten()
        hand_detected_this_frame = True
        # Draw landmarks
        mp_draw.draw_landmarks(annotated_frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                               mp_draw.DrawingSpec(color=(255, 0, 255), thickness=2, circle_radius=2),
                               mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2))

    # Add current frame's landmarks (or placeholder) to the buffer
    landmark_buffer.append(current_landmarks_raw)

    # --- Prediction Logic ---
    # Only predict if the buffer is full and after interval
    if len(landmark_buffer) == SEQUENCE_LENGTH and frame_count % PREDICTION_INTERVAL == 0:

        # ** Apply Normalization to the entire sequence in the buffer **
        sequence_to_predict_raw = np.array(list(landmark_buffer))
        normalized_sequence = normalize_landmarks(sequence_to_predict_raw)

        if normalized_sequence is not None:
            # Reshape for model input: (1, SEQUENCE_LENGTH, num_features)
            input_data = np.expand_dims(normalized_sequence, axis=0).astype(INPUT_DTYPE)

            # Check input shape compatibility
            if input_data.shape != tuple(input_details[0]['shape']):
                 print(f"Error: Input data shape {input_data.shape} doesn't match model expected shape {input_details[0]['shape']}")
            else:
                try:
                    # Run inference
                    interpreter.set_tensor(input_details[0]['index'], input_data)
                    interpreter.invoke()
                    output_data = interpreter.get_tensor(output_details[0]['index'])[0] # Probabilities for each class

                    # Add current probabilities to smoothing buffer
                    prediction_probs_buffer.append(output_data)

                    # --- Smoothing ---
                    if len(prediction_probs_buffer) == SMOOTHING_WINDOW_SIZE:
                        # Average probabilities over the window
                        smoothed_probs = np.mean(np.array(list(prediction_probs_buffer)), axis=0)
                        predicted_index = np.argmax(smoothed_probs)
                        confidence = float(smoothed_probs[predicted_index])

                        # Update display only if confidence meets threshold
                        if confidence >= PREDICTION_THRESHOLD:
                            display_gesture = class_names[predicted_index]
                            display_confidence = confidence
                            # Optional: Reset buffer if a confident prediction is made? Depends on desired behavior.
                        else:
                            # If below threshold, display "Uncertain" or previous gesture?
                            # For simplicity, let's keep showing last confident one or "Uncertain"
                             if display_confidence > 0: # Only reset if we had a confident prediction before
                                 pass # Keep showing last good one
                             else:
                                 display_gesture = "..." # Indicate low confidence state
                             # We don't update display_confidence here if below threshold

                    else:
                         # Not enough data for smoothing yet
                         display_gesture = "Buffering..."
                         display_confidence = 0.0


                except Exception as e:
                    print(f"Error during prediction: {e}")
                    # Reset display on error?
                    display_gesture = "Error"
                    display_confidence = 0.0
        else:
            # Normalization failed (likely no hand in buffer)
            # Reset display or show "No Hand"?
            display_gesture = "No Hand / Norm Error"
            display_confidence = 0.0
            prediction_probs_buffer.clear() # Clear smoothing buffer if input is invalid


    # --- Display Prediction ---
    cv2.putText(annotated_frame, f"Gesture: {display_gesture}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 3, cv2.LINE_AA) # Black outline
    cv2.putText(annotated_frame, f"Gesture: {display_gesture}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA) # White text

    # Display confidence only if it's meaningful
    if display_confidence > 0:
        conf_text = f"Conf: {display_confidence:.2f}"
        cv2.putText(annotated_frame, conf_text, (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(annotated_frame, conf_text, (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)


    # --- Show Frame and Handle Exit ---
    cv2.imshow("Live Continuous Translation", annotated_frame)
    frame_count += 1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Exiting...")
        break

# --- Cleanup ---
hands.close()
cap.release()
cv2.destroyAllWindows()
print("Application finished.")