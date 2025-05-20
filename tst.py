
import numpy as np
import os
import PIL
import random

import pandas as pd

import tensorflow as tf

from tensorflow import keras

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

import plotly.graph_objects as go


import pathlib

import matplotlib.pyplot as plt


data_dir = pathlib.Path('dataset')
class_names = ['Australian terrier', 'Beagle', 'Border terrier', 'Dingo', 'English foxhound', 'Golden retriever', 'Old English sheepdog', 'Rhodesian ridgeback', 'Samoyed', 'Shih-Tzu']


folders = os.listdir('dataset')
image_counts = []
random_images = []

for folder in folders:
    folder_path = os.path.join('dataset', folder)
    if os.path.isdir(folder_path):
        images = [img for img in os.listdir(folder_path) if img.lower().endswith(('png', 'jpg', 'jpeg'))]
        image_counts.append(len(images))


fig = go.Figure([go.Bar(x=folders, y=image_counts, marker_color='blue')])
fig.update_layout(title='Image Count per Folder', xaxis_title='Dog Breed', yaxis_title='Number of Images')
fig.show()


model = keras.models.load_model("final_model.keras")

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    'dataset',
    validation_split=0.2,
    subset='validation',
    seed=1337,
    image_size=(224, 224),
    batch_size=64
)

# Collect predictions and labels
y_true = []
y_pred = []

for images, labels in val_ds:
    predictions = model.predict(images)
    y_true.extend(labels.numpy())
    y_pred.extend(np.argmax(predictions, axis=1))


cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)

plt.figure(figsize=(12, 10))
disp.plot(xticks_rotation=90, cmap='Blues')
plt.title("Confusion Matrix - Dog Breed Classification")
plt.tight_layout()
plt.show()

report_dict = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)

report_df = pd.DataFrame(report_dict).transpose()

metrics_df = report_df.loc[class_names, ['precision', 'recall', 'f1-score']]

# Round and format nicely
metrics_df = metrics_df.round(2)
print(metrics_df)

metrics_df.to_csv("classification_metrics.csv")
