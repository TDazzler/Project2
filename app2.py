
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os

app = Flask(__name__)
model = load_model('model.keras')

UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

class_labels = ['Beagle', 'Boxer', 'Chihuahua', 'Doberman', 'French Bulldog', 
                'Golden Retriever', 'Labrador', 'Poodle', 'Rottweiler', 'Yorkshire Terrier']

def preprocess_image(image_path, target_size=(224, 224)):
    image = load_img(image_path, target_size=target_size)
    image = img_to_array(image) / 255.0
    return np.expand_dims(image, axis=0)


@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    filename = None
    if request.method == 'POST':
        file = request.files['file']
        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)

            img = preprocess_image(filepath)
            pred = model.predict(img)
            prediction = class_labels[np.argmax(pred)]
            filename = file.filename

    return render_template('index.html', prediction=prediction, filename=filename)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
