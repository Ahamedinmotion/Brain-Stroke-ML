import os
from werkzeug.utils import secure_filename
import tensorflow as tf
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
from flask import Flask, render_template, request

# Load the pre-trained model only once
model = load_model("C:/Users/khais/brainstroke_model.h5")

app = Flask(__name__)

# Ensure the uploads folder exists
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/')
def index():
    return render_template('old.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return 'No file part'
    
    file = request.files['file']
    
    if file.filename == '':
        return 'No selected file'
    
    if file:
        # Save the file temporarily
        filename = secure_filename(file.filename)
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path)

        # Load and process the image
        img = image.load_img(file_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0  # Normalization

        # Prediction
        prediction = model.predict(img_array)
        
        # Interpret prediction
        if prediction[0][0] > 0.5:
            result = "Stroke Detected"
        else:
            result = "No Stroke Detected"
        
        return render_template('old.html', prediction=result)

if __name__ == '__main__':
    # Bind to all available IPs on the host and set a specific port
    app.run(host='0.0.0.0', port=5000, debug=True)
