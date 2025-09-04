# training/train_continuous_model.py
import numpy as np
import os
import json
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Masking, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard
from keras_preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
import datetime

# Import utility functions
from utils import load_class_names, save_class_names, normalize_landmarks

# --- Configuration ---
DATASET_DIR = "continuous_dataset"
LANDMARKS_DIR = os.path.join(DATASET_DIR, "landmarks_normalized")
MODEL_SAVE_DIR = "continuous_model"
H5_MODEL_PATH = os.path.join(MODEL_SAVE_DIR, "continuous_gesture_model.h5")
CLASSES_PATH = "class_labels_continuous.json"
TFLITE_MODEL_PATH = os.path.join(MODEL_SAVE_DIR, "continuous_gesture_model.tflite")

SEQUENCE_LENGTH = 40
BATCH_SIZE = 32
EPOCHS = 100
VALIDATION_SPLIT = 0.2

# --- 1. Prepare File List and Labels ---
print("Scanning dataset...")
if not os.path.exists(LANDMARKS_DIR):
    raise FileNotFoundError(f"Directory {LANDMARKS_DIR} does not exist. Please check your dataset path.")

gesture_folders = [d for d in os.listdir(LANDMARKS_DIR) if os.path.isdir(os.path.join(LANDMARKS_DIR, d))]
if not gesture_folders:
    raise ValueError(f"No gesture folders found in {LANDMARKS_DIR}. Did you run preprocessing?")

class_names = sorted(gesture_folders)
class_mapping = {name: i for i, name in enumerate(class_names)}
num_classes = len(class_names)
print(f"Found {num_classes} classes: {class_names}")

# Save class names
OUTPUT_DIR = "training"
os.makedirs(OUTPUT_DIR, exist_ok=True)
CLASSES_PATH = os.path.join(OUTPUT_DIR, "class_labels_continuous.json")
with open(CLASSES_PATH, 'w') as f:
    json.dump(class_names, f)
print(f"Class labels saved to: {CLASSES_PATH}")

# Get list of all .npy file paths and their corresponding integer labels
all_files = []
all_labels = []
for gesture_name, label_idx in class_mapping.items():
    gesture_path = os.path.join(LANDMARKS_DIR, gesture_name)
    if not os.path.exists(gesture_path):
        print(f"Warning: Directory {gesture_path} does not exist. Skipping...")
        continue
    for file in os.listdir(gesture_path):
        if file.endswith(".npy"):
            all_files.append(os.path.join(gesture_path, file))
            all_labels.append(label_idx)

if not all_files:
    raise ValueError("No .npy files found in the dataset directories.")

print(f"Found {len(all_files)} total sequences.")

# --- 2. Split File List into Train and Validation ---
indices = np.arange(len(all_files))
try:
    train_indices, val_indices = train_test_split(
        indices, test_size=VALIDATION_SPLIT, random_state=42, stratify=all_labels
    )
except ValueError as e:
    raise ValueError(f"Error during train-test split: {e}. Check if all_labels has sufficient samples per class.")

train_files = [all_files[i] for i in train_indices]
train_labels = [all_labels[i] for i in train_indices]
val_files = [all_files[i] for i in val_indices]
val_labels = [all_labels[i] for i in val_indices]

print(f"Training samples: {len(train_files)}, Validation samples: {len(val_files)}")

# --- 3. Data Generator ---
MASK_VALUE = 0.0

def data_generator(file_paths, labels, batch_size, seq_length, num_classes, shuffle=True):
    """Generates batches of padded sequences and labels."""
    num_samples = len(file_paths)
    indices = np.arange(num_samples)

    while True:
        if shuffle:
            np.random.shuffle(indices)

        for start in range(0, num_samples, batch_size):
            end = min(start + batch_size, num_samples)
            batch_indices = indices[start:end]

            batch_X_raw = [np.load(file_paths[i]) for i in batch_indices]
            batch_y_raw = [labels[i] for i in batch_indices]

            batch_X_padded = pad_sequences(batch_X_raw, maxlen=seq_length,
                                           dtype='float32', padding='post',
                                           truncating='post', value=MASK_VALUE)

            batch_y_categorical = to_categorical(batch_y_raw, num_classes=num_classes)

            yield batch_X_padded, batch_y_categorical

train_gen = data_generator(train_files, train_labels, BATCH_SIZE, SEQUENCE_LENGTH, num_classes, shuffle=True)
val_gen = data_generator(val_files, val_labels, BATCH_SIZE, SEQUENCE_LENGTH, num_classes, shuffle=False)

steps_per_epoch = int(np.ceil(len(train_files) / BATCH_SIZE))
validation_steps = int(np.ceil(len(val_files) / BATCH_SIZE))

# --- 4. Define Model with Masking ---
print("Defining the model...")
input_shape = (SEQUENCE_LENGTH, 63)

model = Sequential([
    Masking(mask_value=MASK_VALUE, input_shape=input_shape),

    LSTM(64, return_sequences=True),
    BatchNormalization(),
    Dropout(0.3),

    LSTM(128, return_sequences=True),
    BatchNormalization(),
    Dropout(0.4),

    LSTM(64, return_sequences=False),
    BatchNormalization(),
    Dropout(0.3),

    Dense(64, activation='relu'),
    Dropout(0.3),

    Dense(num_classes, activation='softmax', name="output_probabilities")
])

model.summary()

# --- 5. Compile Model ---
print("Compiling the model...")
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# --- 6. Train Model with Callbacks ---
print("Starting model training...")

log_dir = os.path.join("logs", "fit", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

early_stopping = EarlyStopping(monitor='val_loss', patience=15,
                               restore_best_weights=True, verbose=1)

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                              patience=5, min_lr=0.00001, verbose=1)

history = model.fit(train_gen,
                    steps_per_epoch=steps_per_epoch,
                    epochs=EPOCHS,
                    validation_data=val_gen,
                    validation_steps=validation_steps,
                    callbacks=[early_stopping, reduce_lr, tensorboard_callback])

print("Training finished.")

# --- 7. Evaluate Model ---
print("\nEvaluating model on the validation set (using the best weights)...")
val_loss, val_accuracy = model.evaluate(val_gen, steps=validation_steps, verbose=0)
print(f"Validation Loss: {val_loss:.4f}")
print(f"Validation Accuracy: {val_accuracy:.4f}")

# --- 8. Save the Final Trained Model ---
print(f"\nSaving the trained model to {H5_MODEL_PATH}")
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
model.save(H5_MODEL_PATH)

print("\nTraining script complete! Model and class labels saved.")
print(f"Run TensorBoard with: tensorboard --logdir {os.path.abspath('logs')}")

# --- 9. Convert to TFLite ---
print(f"\nConverting model to TFLite: {TFLITE_MODEL_PATH}")

try:
    model = tf.keras.models.load_model(H5_MODEL_PATH)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS
    ]
    converter._experimental_lower_tensor_list_ops = False

    tflite_model = converter.convert()

    with open(TFLITE_MODEL_PATH, 'wb') as f:
        f.write(tflite_model)
    print("✅ Model converted successfully to TFLite.")

except Exception as e:
    import traceback
    print(f"Error during TFLite conversion: {e}")
    print(traceback.format_exc())