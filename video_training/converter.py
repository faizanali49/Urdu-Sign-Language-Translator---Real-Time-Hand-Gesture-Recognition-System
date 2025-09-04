import tensorflow as tf

# Load the trained model
model = tf.keras.models.load_model("continuous_model/continuous_gesture_model.h5")

# Convert to TFLite format
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the converted model
with open("model/improved_gesture_model.tflite", "wb") as f:
    f.write(tflite_model)

print("Model converted to TFLite!")
