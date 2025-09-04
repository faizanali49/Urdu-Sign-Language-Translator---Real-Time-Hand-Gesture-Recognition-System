import tensorflow as tf
import numpy as np
import os
import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# MediaPipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Load your trained model
model = tf.keras.models.load_model("old_model/gesture_model.h5")

# Dataset paths
DATASET_DIR = "dataset"
IMAGES_DIR = os.path.join(DATASET_DIR, "images")
TEST_IMAGES_DIR = os.path.join(DATASET_DIR, "test_images")  # New folder for test images
os.makedirs(TEST_IMAGES_DIR, exist_ok=True)

# Automatically detect gesture labels based on directories
GESTURES = [d for d in os.listdir(IMAGES_DIR) if os.path.isdir(os.path.join(IMAGES_DIR, d))]
print(f"Testing Gestures: {GESTURES}")

def extract_landmarks(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return None
    
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_image)
    
    if not result.multi_hand_landmarks:
        return None
    
    hand_landmarks = result.multi_hand_landmarks[0]
    landmarks = [lm.x for lm in hand_landmarks.landmark] + \
                [lm.y for lm in hand_landmarks.landmark] + \
                [lm.z for lm in hand_landmarks.landmark]
    
    return np.array(landmarks)

def test_with_images():
    all_predictions = []
    all_true_labels = []
    
    for idx, gesture in enumerate(GESTURES):
        gesture_dir = os.path.join(TEST_IMAGES_DIR, gesture)
        if not os.path.exists(gesture_dir):
            continue
            
        print(f"Testing {gesture}...")
        for image_file in os.listdir(gesture_dir):
            if not image_file.endswith(('.png', '.jpg', '.jpeg')):
                continue
                
            image_path = os.path.join(gesture_dir, image_file)
            landmarks = extract_landmarks(image_path)
            
            if landmarks is None:
                print(f"  No hand detected in {image_file}")
                continue
                
            # Make prediction
            prediction = model.predict(np.expand_dims(landmarks, axis=0), verbose=0)
            predicted_class = np.argmax(prediction)
            
            # Save results
            all_predictions.append(predicted_class)
            all_true_labels.append(idx)
            
            # Print result
            correct = "✓" if predicted_class == idx else "✗"
            print(f"  {image_file}: Predicted {GESTURES[predicted_class]} {correct}")
    
    return all_true_labels, all_predictions

def visualize_results(y_true, y_pred):
    # Create confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=GESTURES, yticklabels=GESTURES)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    plt.show()
    
    # Print classification report
    print("\nClassification Report:")
    report = classification_report(y_true, y_pred, target_names=GESTURES)
    print(report)
    
    with open('classification_report.txt', 'w') as f:
        f.write(report)

if __name__ == "__main__":
    print("Testing model with images...")
    y_true, y_pred = test_with_images()
    
    if len(y_true) > 0:
        visualize_results(y_true, y_pred)
        print(f"Total test images: {len(y_true)}")
        accuracy = sum(1 for t, p in zip(y_true, y_pred) if t == p) / len(y_true)
        print(f"Overall accuracy: {accuracy:.2%}")
    else:
        print("No test images were successfully processed.")