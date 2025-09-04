import tensorflow as tf

# Load the trained model
model = tf.keras.models.load_model("model/gesture_model_enhanced.h5")

# Convert to TFLite format
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the converted model
with open("model/gesture_model_enhanced.tflite", "wb") as f:
    f.write(tflite_model)

print("Model converted to TFLite!")
