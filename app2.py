import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from PIL import Image
from flask import send_from_directory
from flask import jsonify
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

import pandas as pd
import matplotlib.pyplot as plt


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

# Replace this with the exact list from train_ds.class_names
class_names = ['Australian terrier', 'Beagle', 'Border terrier', 'Dingo', 'English foxhound', 'Golden retriever', 'Old English sheepdog', 'Rhodesian ridgeback', 'Samoyed', 'Shih-Tzu']

image_size = (224, 224)
model = keras.models.load_model("final_model.keras")




@app.route('/manifest.json')
def manifest():
    return send_from_directory('static', 'manifest.json', mimetype='application/json')

@app.route('/sw.js')
def service_worker():
    return send_from_directory('static', 'sw.js', mimetype='application/javascript')

def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


def preprocess_image(image_path):
    img = keras.utils.load_img(image_path, target_size=image_size)
    img_array = keras.utils.img_to_array(img)
    img_array = keras.ops.expand_dims(img_array, 0)  # Add batch dimension
    return img_array

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files["file"]
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(filepath)

            img = keras.utils.load_img(filepath, target_size=image_size)
            img_array = keras.utils.img_to_array(img)
            img_array = keras.ops.expand_dims(img_array, 0)

            predictions = model.predict(img_array)
            probabilities = keras.ops.softmax(predictions[0]).numpy()

            results = [
                (class_names[i], float(probabilities[i] * 100))
                for i in range(len(class_names))
            ]
            results.sort(key=lambda x: x[1], reverse=True)


            print("top_breed:", results[0][0])
            print("top_breed_prob:", results[0][1])
            # If it's an AJAX request, return JSON
            if request.accept_mimetypes['application/json']:
                return jsonify(predictions=results)
    
    return render_template("index.html")

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')