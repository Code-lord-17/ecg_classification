import os
import numpy as np
from flask import Flask, request, jsonify, render_template, send_from_directory
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import tensorflow as tf
from PIL import Image
import uuid
import time
import logging

# Set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler("app.log"),
                              logging.StreamHandler()])
logger = logging.getLogger(__name__)

# Initialize the Flask application
app = Flask(__name__, static_folder='static')
from metadata import extract_predict
# Load your pre-trained model
try:
    logger.info("Loading model...")
    model = load_model('saved_model/stress_analysis.keras')
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Error loading model: {str(e)}")
    model = None

# Class names (from your dataset)
class_names = ['mental stress', 'no stress', 'physical stress']

# Define image size
IMAGE_SIZE = 256

# Ensure required directories exist
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Prediction function
def prepare_image(img_path):
    try:
        img = Image.open(img_path)
        img = img.resize((IMAGE_SIZE, IMAGE_SIZE))  # Resize to match the model's expected input
        img_array = img_to_array(img) / 255.0  # Normalize the image
        return np.expand_dims(img_array, axis=0)  # Add batch dimension
    except Exception as e:
        logger.error(f"Error preparing image: {str(e)}")
        return None

# Route to the home page
@app.route('/')
def home():
    return render_template('index.html')

# Route to handle the image upload and prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        logger.warning("No file part in the request")
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        logger.warning("No selected file")
        return jsonify({'error': 'No selected file'}), 400

    if file:
        try:
            # Generate a unique filename to prevent overwriting
            filename = f"{uuid.uuid4()}_{file.filename}"
            img_path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(img_path)
            logger.info(f"Saved uploaded file: {img_path}")

            # Prepare the image and make a prediction
            start_time = time.time()
            img_array = prepare_image(img_path)
            if img_array is None:
                return jsonify({'error': 'Failed to process image'}), 400
            if model is None:
                return jsonify({'error': 'Model not loaded'}), 500
            predictions = model.predict(img_array)
            predicted_class = class_names[np.argmax(predictions)]
            predict = "Unknown"
            if predict != "Unknown":
                predicted_class = predict['value']
            processing_time = round(time.time() - start_time, 2)
            logger.info(f"processing time: {processing_time}s")
            # Add processing time to response
            response = {
                'predicted_class': predicted_class,
                'processing_time': f'{processing_time}s'
            }
            
            return jsonify(response)
            
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            return jsonify({'error': str(e)}), 500

# Route to serve uploaded files (optional, if you need to display the images in the UI)
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

# Health check endpoint
@app.route('/health')
def health():
    health_status = {
        'status': 'ok',
        'model_loaded': model is not None
    }
    return jsonify(health_status)

if __name__ == '__main__':
    # Run the Flask application
    logger.info("Starting Flask application...")
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))