from flask import Flask, request, jsonify
import numpy as np
from PIL import Image
import io
import tflite_runtime.interpreter as tflite

# Define the Flask application
app = Flask(__name__)

# --- Configuration ---
# Your TFLite model file and class labels
TFLITE_MODEL_PATH = 'quantized_model.tflite'
CLASS_LABELS = ['biodegradable', 'non-biodegradable']

# --- Model Loading and Initialization ---
# Load the TFLite model and allocate tensors. This is a one-time operation
# when the Flask app starts, which is much more efficient.
try:
    interpreter = tflite.Interpreter(model_path=TFLITE_MODEL_PATH)
    interpreter.allocate_tensors()

    # Get input and output tensor details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Get input shape and data type
    input_shape = input_details[0]['shape']
    input_dtype = input_details[0]['dtype']

except Exception as e:
    print(f"Error loading TFLite model: {e}")
    interpreter = None


# --- Preprocessing Function ---
def preprocess_image(image_bytes, target_size=(input_shape[1], input_shape[2])):
    """
    Preprocesses an image to match the model's input requirements.

    Args:
        image_bytes: The raw image data as bytes.
        target_size: A tuple (height, width) for resizing.

    Returns:
        A preprocessed numpy array ready for model inference.
    """
    try:
        # Open the image using Pillow from the byte stream
        img = Image.open(io.BytesIO(image_bytes))

        # Resize the image to the model's expected size
        img = img.resize(target_size)

        # Convert to a NumPy array
        img_array = np.array(img, dtype=np.float32)

        # Check if the model expects normalization
        if input_dtype == np.float32:
            # Normalize the pixel values (e.g., from 0-255 to 0-1)
            img_array = img_array / 255.0

        # Add a batch dimension (e.g., from (224, 224, 3) to (1, 224, 224, 3))
        img_array = np.expand_dims(img_array, axis=0)

        return img_array

    except Exception as e:
        print(f"Error during image preprocessing: {e}")
        return None


# --- Flask Routes ---
@app.route('/')
def home():
    """A simple home route to check if the server is running."""
    return "Waste Classification API is running!"


@app.route('/predict', methods=['POST'])
def predict():
    """
    Prediction endpoint that receives an image and returns a classification.
    """
    # Check if a model was successfully loaded
    if interpreter is None:
        return jsonify({'error': 'Model not loaded.'}), 500

    # Check if a file was uploaded in the request
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['file']

    # Check if the filename is empty
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file:
        try:
            # Read the image data from the file stream
            image_bytes = file.read()

            # Preprocess the image
            processed_image = preprocess_image(image_bytes)

            if processed_image is None:
                return jsonify({'error': 'Failed to preprocess image.'}), 500

            # Set the input tensor
            interpreter.set_tensor(input_details[0]['index'], processed_image)

            # Run the inference
            interpreter.invoke()

            # Get the output tensor and post-process
            output_data = interpreter.get_tensor(output_details[0]['index'])

            # Get the class with the highest probability
            predicted_class_index = np.argmax(output_data)
            predicted_class = CLASS_LABELS[predicted_class_index]
            confidence = float(output_data[0][predicted_class_index])

            # Return the prediction as a JSON response
            return jsonify({
                'prediction': predicted_class,
                'confidence': confidence
            })

        except Exception as e:
            # Catch any other errors and return a detailed message
            return jsonify({'error': str(e)}), 500


# To run the Flask application
if __name__ == '__main__':
    # Using `0.0.0.0` makes the server accessible from outside the local machine.
    # This is necessary for other devices on your network to connect to the RPi.
    app.run(host='0.0.0.0', port=5000)