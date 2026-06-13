import tensorflow as tf
from tensorflow import keras
import numpy as np
from tensorflow.keras.preprocessing import image
import os

# 1️⃣ Load the saved model
model = keras.models.load_model('model.h5')
class_names = ['biodegradable', 'non-biodegradable']  # Example: ['garbage', 'recyclable']
# 2️⃣ Define image preprocessing function
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))  # Resize to (224, 224)
    img_array = image.img_to_array(img)
    img_array = (img_array / 127.5) - 1  # Normalize to [-1, 1]
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# 3️⃣ Predict function
def predict_image(img_path, class_names):
    img_array = preprocess_image(img_path)
    predictions = model.predict(img_array)  # Shape: (1, NUM_CLASSES)

    predicted_class_index = np.argmax(predictions[0])  # Get class index
    confidence = predictions[0][predicted_class_index]

    predicted_class_name = class_names[predicted_class_index]
    return predicted_class_name, confidence

# 4️⃣ Define class names (adjust based on your dataset)


# 5️⃣ Example usage
image_path = 'C:\\Users\\paras\\Downloads\\Datasets\\garbage-dataset\\biodegradable\\clothes\\clothes_99.jpg'  # Put your test image path here

predicted_class, confidence = predict_image(image_path, class_names)
print(f"Predicted class: {predicted_class}")
print(f"Confidence: {confidence:.4f}")
