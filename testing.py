import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

from tensorflow.keras.models import load_model

import pathlib

import matplotlib.pyplot as plt
# Load the trained model
model = load_model("model.keras")



IMG_SIZE = (224, 224)  # Resize images
BATCH_SIZE = 32

train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    "dataset/",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    validation_split=0.2,
    subset="training",
    seed=123
)

# Function to preprocess and predict
def predict_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))  # Resize
    img_array = image.img_to_array(img) / 255.0  # Normalize (0-1)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)
    confidence = prediction[0][predicted_class]

    class_names = train_dataset.class_names
    print(f"Predicted breed: {class_names[predicted_class]} ({confidence:.2f})")

test_dir = pathlib.Path('test-images')
paths = list(test_dir.glob('*.jpg'))

for i, path in enumerate(paths):
    predict_image(path) 

