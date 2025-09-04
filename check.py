import cv2
import mediapipe as mp
import numpy as np
import tensorflow.lite as tflite
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

# Urdu mapping (same as before)
urdu_letter_map = {
    "alif": "برائے مہربانی ", "bay": "آپ کی محنت اور رہنمائی کے لیے ہم تہہ دل سے مشکور ہیں", "pay": "پ", "tay": "ت", "tay marbuta": "ة",
    "say": "ث", "jeem": "ج", "chay": "چ", "hay": "ح", "khay": "خ",
    "daal": "د", "zaal": "ذ", "ray": "ر", "zay": "ز", "zhay": "ژ",
    "seen": "س", "sheen": "ش", "saad": "ص", "zaad": "ض", "toay": "ط",
    "zoay": "ظ", "ain": "ع", "ghain": "غ", "fay": "ف", "qaaf": "ق",
    "kaaf": "ک", "gaaf": "گ", "laam": "ل", "meem": "م", "noon": "ن",
    "waw": "و", "hay": "ہ", "hamza": "ء", "yay": "ی", "barri yay": "ے",
    "برائے مہربانی": "برائے مہربانی "
}


GESTURES = list(urdu_letter_map.keys()) + ["delete"]

# TFLite model (same as before)
interpreter = tflite.Interpreter(model_path="model/gesture_model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# MediaPipe hands (same as before)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)
urdu_sentence = []
last_gesture = None
pose_start_time = None
current_prediction = None
frame_skip = 0

# GUI Window
root = tk.Tk()
root.title("Live Urdu Sign Language Translator")
root.geometry("950x850")  # Adjusted size for better layout
root.configure(bg='#f0f0f0') # Light gray background

# --- Function Definitions (Moved to the top) ---
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
            os.environ["TMPDIR"] = os.getcwd()
            tts = gTTS(text=reversed_text, lang='ur', slow=False)
            audio_file = "urdu_output.mp3"
            tts.save(audio_file)
            try:
                from playsound import playsound
                playsound(audio_file)
            except Exception as e:
                print(f"Playsound Error: {e}")
            try:
                os.remove(audio_file)
            except PermissionError:
                print("Could not delete the audio file due to permission error.")
        except Exception as e:
            print("TTS Error:", e)

def clear_text():
    global urdu_sentence
    urdu_sentence = []
    text_box.delete("1.0", tk.END)
    recognized_pose_textbox.delete("1.0", tk.END)

def process_frame():
    global current_prediction, last_gesture, pose_start_time, frame_skip

    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame from webcam.")
        root.after(10, process_frame)
        return

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    if frame_skip % 3 == 0:  # Check every 3rd frame for faster response
        result = hands.process(rgb_frame)
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                landmarks = [lm.x for lm in hand_landmarks.landmark] + \
                            [lm.y for lm in hand_landmarks.landmark] + \
                            [lm.z for lm in hand_landmarks.landmark]

                interpreter.set_tensor(input_details[0]['index'], np.array([landmarks], dtype=np.float32))
                interpreter.invoke()
                prediction = interpreter.get_tensor(output_details[0]['index'])
                predicted_index = np.argmax(prediction)
                current_prediction = GESTURES[predicted_index]
                recognized_pose_textbox.delete("1.0", tk.END)
                recognized_pose_textbox.insert("1.0", current_prediction)
                break
        else:
            current_prediction = None
            recognized_pose_textbox.delete("1.0", tk.END)
            recognized_pose_textbox.insert("1.0", "No hand detected")

        if current_prediction:
            if current_prediction != last_gesture:
                last_gesture = current_prediction
                pose_start_time = time.time()
            elif last_gesture == current_prediction and pose_start_time:
                elapsed_time = time.time() - pose_start_time
                if elapsed_time >= 1.0:
                    print(f"Gesture Confirmed after 1 second: {current_prediction}")
                    if current_prediction == "delete":
                        if urdu_sentence:
                            urdu_sentence.pop()
                    elif current_prediction in urdu_letter_map:
                        letter = urdu_letter_map.get(current_prediction)
                        if letter:
                            urdu_sentence.append(letter)
                    last_gesture = None
                    pose_start_time = None
        else:
            last_gesture = None
            pose_start_time = None

        urdu_text = join_urdu_letters(urdu_sentence)
        text_box.delete("1.0", tk.END)
        text_box.insert(tk.END, urdu_text)

    frame_skip += 1
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    imgtk = ImageTk.PhotoImage(image=img)
    video_label.imgtk = imgtk
    video_label.configure(image=imgtk)
    root.after(10, process_frame)

# --- UI Elements with Improved Layout ---

# Video Frame
video_frame = tk.LabelFrame(root, text="Live Video Feed", padx=5, pady=5, bg='white')
video_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
video_label = tk.Label(video_frame, bg='black') # Black background for video feed
video_label.pack(padx=10, pady=10, fill="both", expand=True)

# Text Area Frame
text_frame = tk.LabelFrame(root, text="Recognized Urdu Text", padx=5, pady=5, bg='white')
text_frame.grid(row=1, column=0, columnspan=2, padx=10, pady=10, sticky="ew")
text_box = scrolledtext.ScrolledText(text_frame, height=5, font=("Arial", 16), wrap=tk.WORD, bg='#e0f2f7') # Light blue background
text_box.pack(padx=10, pady=10, fill="both", expand=True)

# Recognized Pose Frame
recognized_pose_frame = tk.LabelFrame(root, text="Recognized Pose", padx=5, pady=5, bg='white')
recognized_pose_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
recognized_pose_textbox = scrolledtext.ScrolledText(recognized_pose_frame, height=1, width=10, font=("Arial", 16), wrap=tk.WORD, bg='#e0f2f7')
recognized_pose_textbox.pack(padx=10, pady=10, fill="both", expand=True)

# Buttons Frame
buttons_frame = tk.Frame(root, bg='#f0f0f0')
buttons_frame.grid(row=2, column=0, columnspan=2, padx=10, pady=10, sticky="ew")

speak_button = tk.Button(buttons_frame, text="Speak", font=("Arial", 14), command=speak_text, bg='#4CAF50', fg='white', padx=20, pady=10)
speak_button.pack(side=tk.LEFT, padx=10, pady=5, fill="x", expand=True)

clear_button = tk.Button(buttons_frame, text="Clear All", font=("Arial", 14), command=clear_text, bg='#f44336', fg='white', padx=20, pady=10)
clear_button.pack(side=tk.LEFT, padx=10, pady=5, fill="x", expand=True)

start_button = tk.Button(buttons_frame, text="Start Recognition", font=("Arial", 14), command=process_frame, bg='#2196F3', fg='white', padx=20, pady=10)
start_button.pack(side=tk.RIGHT, padx=10, pady=5, fill="x", expand=True)

# Configure grid row and column weights for resizing
root.grid_rowconfigure(0, weight=1)
root.grid_rowconfigure(1, weight=1)
root.grid_columnconfigure(0, weight=1)
root.grid_columnconfigure(1, weight=1)

# Start recognition loop (you can keep this or trigger via the button)
# process_frame()

root.mainloop()
cap.release()
cv2.destroyAllWindows()