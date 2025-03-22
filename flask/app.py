import os
from werkzeug.utils import secure_filename
import tensorflow as tf
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
from flask import Flask, render_template, request
from huggingface_hub import hf_hub_download

# Download the model from Hugging Face
MODEL_PATH = hf_hub_download(repo_id="Ahamedinmotion/brainstroke-model", filename="brainstroke_model.h5")
model = load_model(MODEL_PATH)

app = Flask(__name__)

# Ensure the uploads folder exists
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return 'No file part'
    
    file = request.files['file']
    if file.filename == '':
        return 'No selected file'
    
    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path)

        # Load and preprocess the image
        img = image.load_img(file_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalize

        # Prediction
        prediction = model.predict(img_array)
        result = "Stroke Detected" if prediction[0][0] > 0.5 else "No Stroke Detected"
        
        return render_template('index.html', prediction=result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
