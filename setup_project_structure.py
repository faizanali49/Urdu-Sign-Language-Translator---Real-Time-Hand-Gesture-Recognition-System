import os

# Define folder structure
folders = [
    "PSL_Urdu_Translator/app",
    "PSL_Urdu_Translator/data_acquisition",
    "PSL_Urdu_Translator/training",
    "PSL_Urdu_Translator/models",
    "PSL_Urdu_Translator/dataset/landmarks/alif",
    "PSL_Urdu_Translator/dataset/landmarks/bay",
    "PSL_Urdu_Translator/dataset/landmarks/no_gesture",
    "PSL_Urdu_Translator/dataset/images/alif",
    "PSL_Urdu_Translator/dataset/images/bay",
    "PSL_Urdu_Translator/dataset/images/no_gesture",
    "PSL_Urdu_Translator/assets/fonts"
]

# Define files and their placeholder content
files = {
    "PSL_Urdu_Translator/app/__init__.py": "",
    "PSL_Urdu_Translator/app/main.py": "# Starts the GUI app\n",
    "PSL_Urdu_Translator/app/gui.py": "# Tkinter layout + update loop\n",
    "PSL_Urdu_Translator/app/camera.py": "# Webcam handler class\n",
    "PSL_Urdu_Translator/app/hand_tracker.py": "# MediaPipe hand tracking class\n",
    "PSL_Urdu_Translator/app/translator.py": "# TFLite prediction + Urdu mapping + TTS\n",
    "PSL_Urdu_Translator/app/utils.py": "# Helper functions: reshaping, normalization, etc.\n",

    "PSL_Urdu_Translator/data_acquisition/__init__.py": "",
    "PSL_Urdu_Translator/data_acquisition/collect_gestures.py": "# Modified version of data_collection.py\n",
    "PSL_Urdu_Translator/data_acquisition/preprocess_data.py": "# (Optional) Preprocessing script\n",

    "PSL_Urdu_Translator/training/__init__.py": "",
    "PSL_Urdu_Translator/training/train_landmark_model.py": "# Improved train_model.py\n",
    "PSL_Urdu_Translator/training/convert_to_tflite.py": "# Convert Keras model to TFLite\n",
    "PSL_Urdu_Translator/training/class_labels.json": "# Will store gesture labels\n",

    "PSL_Urdu_Translator/models/psl_urdu_model.tflite": "",  # Placeholder for model file

    "PSL_Urdu_Translator/assets/fonts/Jameel Noori Nastaleeq.ttf": "",  # Add font manually later

    "PSL_Urdu_Translator/.gitignore": "# Bytecode, models, cache files to ignore\n*.pyc\n__pycache__/\nmodels/*.h5\n",
    "PSL_Urdu_Translator/requirements.txt": "# Add required Python packages here\nmediapipe\ntensorflow\ngtts\nopencv-python\npydub\narabic-reshaper\nbidi\n",
    "PSL_Urdu_Translator/README.md": "# PSL Urdu Translator\n\nThis project recognizes hand gestures and translates them into Urdu text and speech.\n"
}

# Create folders
for folder in folders:
    os.makedirs(folder, exist_ok=True)

# Create files
for path, content in files.items():
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)

print("✅ PSL Urdu Translator project structure created successfully.")
