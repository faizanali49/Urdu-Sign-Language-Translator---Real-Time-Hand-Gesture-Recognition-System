import cv2
import mediapipe as mp
import numpy as np
import os

# === MediaPipe Setup ===
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
    max_num_hands=1
)
mp_draw = mp.solutions.drawing_utils

# === Dataset Directories ===
DATASET_DIR = "dataset"
LANDMARKS_DIR = os.path.join(DATASET_DIR, "landmarks")
IMAGES_DIR = os.path.join(DATASET_DIR, "images")

os.makedirs(LANDMARKS_DIR, exist_ok=True)
os.makedirs(IMAGES_DIR, exist_ok=True)

# === Crop Function ===
def crop_hand_region(frame, hand_landmarks, padding=30):
    h, w, _ = frame.shape
    x_coords = [int(lm.x * w) for lm in hand_landmarks.landmark]
    y_coords = [int(lm.y * h) for lm in hand_landmarks.landmark]

    min_x = max(min(x_coords) - padding, 0)
    max_x = min(max(x_coords) + padding, w)
    min_y = max(min(y_coords) - padding, 0)
    max_y = min(max(y_coords) + padding, h)

    return frame[min_y:max_y, min_x:max_x]

# === Webcam Init ===
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Cannot open webcam.")
    exit()

print("=== Hand Gesture Data Collector ===\n")

while True:
    # --- User input ---
    gesture = input("Enter gesture name (or 'exit' to quit): ").strip().lower()
    if gesture == 'exit':
        break

    try:
        num_samples = int(input(f"How many new samples to record for '{gesture}'? "))
        if num_samples <= 0:
            print("Number must be > 0.")
            continue
    except ValueError:
        print("Invalid number.")
        continue

    # --- Folder Setup ---
    gesture_landmark_path = os.path.join(LANDMARKS_DIR, gesture)
    gesture_image_path = os.path.join(IMAGES_DIR, gesture)
    os.makedirs(gesture_landmark_path, exist_ok=True)
    os.makedirs(gesture_image_path, exist_ok=True)

    # --- Get next starting index ---
    existing_imgs = [f for f in os.listdir(gesture_image_path) if f.endswith('.png')]
    existing_lmks = [f for f in os.listdir(gesture_landmark_path) if f.endswith('.npy')]
    start_index = max(len(existing_imgs), len(existing_lmks))
    print(f"Existing samples found: {start_index}")
    print(f"New samples will be saved as {start_index}.png/.npy onwards\n")
    print("➡️ Press 's' to start data capture, or 'q' to skip.")

    # --- Wait for 's' to start ---
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.flip(frame, 1)
        cv2.putText(frame, f"Press 's' to start collecting '{gesture}'", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.imshow("Gesture Data Collector", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            break
        elif key == ord('q'):
            print("Skipping gesture.\n")
            continue

    # --- Data Collection ---
    count = 0
    while count < num_samples:
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # --- Extract and Save Landmarks ---
                landmarks = [lm.x for lm in hand_landmarks.landmark] + \
                            [lm.y for lm in hand_landmarks.landmark] + \
                            [lm.z for lm in hand_landmarks.landmark]
                if len(landmarks) == 63:
                    npy_path = os.path.join(gesture_landmark_path, f"{start_index + count}.npy")
                    np.save(npy_path, np.array(landmarks, dtype=np.float32))

                    # --- Crop and Save Hand Image ---
                    cropped = crop_hand_region(frame, hand_landmarks)
                    if cropped.size == 0:
                        print("Warning: Skipping empty cropped image.")
                        continue

                    img_path = os.path.join(gesture_image_path, f"{start_index + count}.png")
                    cv2.imwrite(img_path, cropped)

                    count += 1
                    print(f"[{count}/{num_samples}] Sample saved.")

                # Draw hand overlay
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Show current frame
        cv2.putText(frame, f"Collecting '{gesture}': {count}/{num_samples}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("Gesture Data Collector", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Stopped early.")
            break

print("\n✅ Data collection complete.")
cap.release()
cv2.destroyAllWindows()
