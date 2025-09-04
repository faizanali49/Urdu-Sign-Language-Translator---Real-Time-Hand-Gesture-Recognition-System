import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import arabic_reshaper
from bidi.algorithm import get_display
from gtts import gTTS
from playsound import playsound
import threading
import tkinter as tk
from tkinter import scrolledtext
from PIL import Image, ImageTk
import tempfile
import os
import time
from collections import Counter
from urdu_data.labels import Label

# Mapping for Urdu text output
urdu_letter_map = Label.label

GESTURES = list(urdu_letter_map.keys()) + ["delete"]

# Load the improved TFLite model
MODEL_PATH = "model/gesture_model.h5"
try:
    model = tf.keras.models.load_model(MODEL_PATH)
except Exception as e:
    print(f"Error loading Keras model: {e}")
    exit(1)


# Initialize MediaPipe Hands for landmark detection
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# OpenCV Video Capture
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit(1)

# Global variables for UI and inference
urdu_sentence = []
frame_skip = 0

# Time-based gesture recognition variables
current_gesture = None
gesture_start_time = 0
GESTURE_TIMEOUT = 2.0  # 2 seconds to confirm a gesture

# Tkinter UI Setup
root = tk.Tk()
root.title("Live Urdu Sign Language Translator - Time-Based")
root.geometry("950x850")
root.configure(bg='#f0f0f0')

def join_urdu_letters(letters):
    urdu_word = "".join(letters)
    reshaped_word = arabic_reshaper.reshape(urdu_word)
    bidi_word = get_display(reshaped_word)
    return bidi_word

def speak_text():
    text = text_box.get("1.0", tk.END).strip()
    if text:
        try:
            reversed_text = text[::-1]
            tts = gTTS(text=reversed_text, lang='ur', slow=False)
            audio_file = "urdu_output.mp3"
            tts.save(audio_file)
            playsound(audio_file)
            try:
                os.remove(audio_file)
            except PermissionError:
                print("Permission error: Could not delete the audio file.")
        except Exception as e:
            print("TTS Error:", e)

def clear_text():
    global urdu_sentence, current_gesture, gesture_start_time
    urdu_sentence = []
    current_gesture = None
    gesture_start_time = 0
    text_box.delete("1.0", tk.END)
    recognized_pose_textbox.delete("1.0", tk.END)
    gesture_time_label.config(text="Waiting...")

def process_frame():
    global frame_skip, urdu_sentence, current_gesture, gesture_start_time

    ret, frame = cap.read()
    if not ret:
        print("Warning: Could not read frame from webcam.")
        root.after(10, process_frame)
        return

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process every 3rd frame for efficiency
    if frame_skip % 3 == 0:
        result = hands.process(rgb_frame)
        detected_gesture = None

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                # Extract landmarks as a flat array (x, y, z for 21 landmarks)
                landmarks = [lm.x for lm in hand_landmarks.landmark] + \
                            [lm.y for lm in hand_landmarks.landmark] + \
                            [lm.z for lm in hand_landmarks.landmark]
                landmarks = np.array(landmarks, dtype=np.float32).reshape(1, -1)
                try:
                    
                    prediction = model.predict(landmarks)
                    predicted_index = np.argmax(prediction)
                    detected_gesture = GESTURES[predicted_index]
                except Exception as e:
                    print(f"Inference Error: {e}")
                break
        else:
            recognized_pose_textbox.delete("1.0", tk.END)
            recognized_pose_textbox.insert("1.0", "No hand detected")
            # Reset time tracking when no hand is detected
            current_gesture = None
            gesture_start_time = 0
            gesture_time_label.config(text="Waiting...")

        # Time-based gesture recognition
        if detected_gesture:
            recognized_pose_textbox.delete("1.0", tk.END)
            recognized_pose_textbox.insert("1.0", detected_gesture)
            
            current_time = time.time()
            
            # If this is a new gesture or we weren't tracking one
            if current_gesture != detected_gesture:
                current_gesture = detected_gesture
                gesture_start_time = current_time
                gesture_time_label.config(text=f"Hold for: 0.0/{GESTURE_TIMEOUT:.1f}s")
            else:
                # Same gesture, check elapsed time
                elapsed_time = current_time - gesture_start_time
                gesture_time_label.config(text=f"Hold for: {elapsed_time:.1f}/{GESTURE_TIMEOUT:.1f}s")
                
                # If 2 seconds elapsed with same gesture
                if elapsed_time >= GESTURE_TIMEOUT:
                    # Trigger action based on the gesture
                    if current_gesture == "delete":
                        if urdu_sentence:
                            urdu_sentence.pop()
                    elif current_gesture in urdu_letter_map:
                        letter = urdu_letter_map.get(current_gesture)
                        if letter:
                            urdu_sentence.append(letter)
                    
                    # Reset to start tracking for a new gesture
                    current_gesture = None
                    gesture_start_time = 0
                    gesture_time_label.config(text="Added! Waiting...")
                    
                    # Update the UI text box with the composed Urdu sentence
                    urdu_text = join_urdu_letters(urdu_sentence)
                    text_box.delete("1.0", tk.END)
                    text_box.insert(tk.END, urdu_text)

    frame_skip += 1
    # Update the video feed in the tkinter window
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    imgtk = ImageTk.PhotoImage(image=img)
    video_label.imgtk = imgtk
    video_label.configure(image=imgtk)
    root.after(10, process_frame)

# UI Setup: Video Frame
video_frame = tk.LabelFrame(root, text="Live Video Feed", padx=5, pady=5, bg='white')
video_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
video_label = tk.Label(video_frame, bg='black')
video_label.pack(padx=10, pady=10, fill="both", expand=True)

# UI Setup: Recognized Urdu Text
text_frame = tk.LabelFrame(root, text="Recognized Urdu Text", padx=5, pady=5, bg='white')
text_frame.grid(row=1, column=0, columnspan=2, padx=10, pady=10, sticky="ew")
text_box = scrolledtext.ScrolledText(text_frame, height=5, font=("Arial", 16), wrap=tk.WORD, bg='#e0f2f7')
text_box.pack(padx=10, pady=10, fill="both", expand=True)

# UI Setup: Recognized Pose
recognized_pose_frame = tk.LabelFrame(root, text="Recognized Pose", padx=5, pady=5, bg='white')
recognized_pose_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
recognized_pose_textbox = scrolledtext.ScrolledText(recognized_pose_frame, height=1, width=10, font=("Arial", 16), wrap=tk.WORD, bg='#e0f2f7')
recognized_pose_textbox.pack(padx=10, pady=5, fill="both", expand=True)

# Add a timer label to show progress
gesture_time_label = tk.Label(recognized_pose_frame, text="Waiting...", font=("Arial", 12), bg='white')
gesture_time_label.pack(padx=10, pady=5, fill="x")

# UI Setup: Control Buttons
buttons_frame = tk.Frame(root, bg='#f0f0f0')
buttons_frame.grid(row=2, column=0, columnspan=2, padx=10, pady=10, sticky="ew")

speak_button = tk.Button(buttons_frame, text="Speak", font=("Arial", 14), command=speak_text, bg='#4CAF50', fg='white', padx=20, pady=10)
speak_button.pack(side=tk.LEFT, padx=10, pady=5, fill="x", expand=True)

clear_button = tk.Button(buttons_frame, text="Clear All", font=("Arial", 14), command=clear_text, bg='#f44336', fg='white', padx=20, pady=10)
clear_button.pack(side=tk.LEFT, padx=10, pady=5, fill="x", expand=True)

start_button = tk.Button(buttons_frame, text="Start Recognition", font=("Arial", 14), command=process_frame, bg='#2196F3', fg='white', padx=20, pady=10)
start_button.pack(side=tk.RIGHT, padx=10, pady=5, fill="x", expand=True)

# Grid Configuration for Responsive Layout
root.grid_rowconfigure(0, weight=1)
root.grid_rowconfigure(1, weight=1)
root.grid_columnconfigure(0, weight=1)
root.grid_columnconfigure(1, weight=1)

# Start the Tkinter event loop
root.mainloop()
cap.release()
cv2.destroyAllWindows()