import os
import numpy as np
import skimage
from skimage import transform
from skimage.color import rgb2gray


def load_data(data_dir):
    dirs = [
        d for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d))
    ]
    labels = []
    images = []
    for d in dirs:
        label_dir = os.path.join(data_dir, d)
        file_names = [
            os.path.join(label_dir, f) for f in os.listdir(label_dir)
            if f.endswith(".ppm")
        ]
        for f in file_names:
            images.append(skimage.data.imread(f))
            labels.append(int(d))
    return np.array(images), np.array(labels)


def convert_to_squared_grayscale(images):
    return rgb2gray(
        np.array([
            transform.resize(
                image, (28, 28), mode="constant", anti_aliasing=False)
            for image in images
        ]))
