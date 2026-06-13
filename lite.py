import tensorflow as tf

# Load the Keras model from the .h5 file
model = tf.keras.models.load_model('model.h5')

# Initialize the converter from the Keras model
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Convert the model to TFLite format
tflite_model = converter.convert()

# Enable the default optimizations, which include quantization
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Convert the model with quantization
quantized_tflite_model = converter.convert()

# Save the quantized model to a .tflite file
with open('quantized_model.tflite', 'wb') as f:
    f.write(quantized_tflite_model)

# Save the non-quantized model
with open('model.tflite', 'wb') as f:
    f.write(tflite_model)
