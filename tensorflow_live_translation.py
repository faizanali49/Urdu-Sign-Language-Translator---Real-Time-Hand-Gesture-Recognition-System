import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
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

# Load the full Keras .h5 model instead of TFLite
MODEL_PATH = "model/gesture_model_enhanced.h5"
try:
    model = load_model(MODEL_PATH)
    print("TensorFlow model loaded successfully")
except Exception as e:
    print(f"Error loading TensorFlow model: {e}")
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
prediction_buffer = []
SMOOTHING_WINDOW = 5  # Number of frames for majority vote
frame_skip = 0

# Tkinter UI Setup
root = tk.Tk()
root.title("Live Urdu Sign Language Translator - TensorFlow Model")
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
    global urdu_sentence, prediction_buffer
    urdu_sentence = []
    prediction_buffer = []
    text_box.delete("1.0", tk.END)
    recognized_pose_textbox.delete("1.0", tk.END)

def process_frame():
    global frame_skip, prediction_buffer, urdu_sentence

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
        current_prediction = None

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                # Extract landmarks as a flat array (x, y, z for 21 landmarks)
                landmarks = [lm.x for lm in hand_landmarks.landmark] + \
                            [lm.y for lm in hand_landmarks.landmark] + \
                            [lm.z for lm in hand_landmarks.landmark]
                landmarks = np.array(landmarks, dtype=np.float32).reshape(1, -1)
                try:
                    # Use direct model prediction instead of TFLite interpreter
                    prediction = model.predict(landmarks, verbose=0)
                    predicted_index = np.argmax(prediction)
                    current_prediction = GESTURES[predicted_index]
                except Exception as e:
                    print(f"Inference Error: {e}")
                break
        else:
            recognized_pose_textbox.delete("1.0", tk.END)
            recognized_pose_textbox.insert("1.0", "No hand detected")

        # If a valid prediction is made, add it to the smoothing buffer
        if current_prediction:
            prediction_buffer.append(current_prediction)
            # Limit the buffer size
            if len(prediction_buffer) > SMOOTHING_WINDOW:
                prediction_buffer.pop(0)
            # When the buffer is full, perform majority voting
            if len(prediction_buffer) == SMOOTHING_WINDOW:
                majority_vote = Counter(prediction_buffer).most_common(1)[0][0]
                recognized_pose_textbox.delete("1.0", tk.END)
                recognized_pose_textbox.insert("1.0", majority_vote)
                # Trigger action based on the majority vote and clear the buffer
                if majority_vote == "delete":
                    if urdu_sentence:
                        urdu_sentence.pop()
                elif majority_vote in urdu_letter_map:
                    letter = urdu_letter_map.get(majority_vote)
                    if letter:
                        urdu_sentence.append(letter)
                prediction_buffer = []  # reset buffer after confirming a gesture

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
recognized_pose_textbox.pack(padx=10, pady=10, fill="both", expand=True)

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