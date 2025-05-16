
import numpy as np
import os
import PIL
import random
import cv2

import plotly.graph_objects as go


import pathlib

import matplotlib.pyplot as plt


data_dir = pathlib.Path('dataset')
image_count = len(list(data_dir.glob('*/*.jpg')))

# Get folder names and count images
folders = os.listdir('dataset')
image_counts = []
random_images = []

for folder in folders:
    folder_path = os.path.join('dataset', folder)
    if os.path.isdir(folder_path):
        images = [img for img in os.listdir(folder_path) if img.lower().endswith(('png', 'jpg', 'jpeg'))]
        image_counts.append(len(images))


# Create bar chart with Plotly
fig = go.Figure([go.Bar(x=folders, y=image_counts, marker_color='blue')])
fig.update_layout(title='Image Count per Folder', xaxis_title='Dog Breed', yaxis_title='Number of Images')
fig.show()
