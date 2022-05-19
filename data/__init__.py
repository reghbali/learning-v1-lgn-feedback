"""raw data including images"""

import matplotlib.pyplot as plt
import numpy as np
import os
from skimage import color

data_dir = os.path.abspath(os.path.dirname(__file__))

__all__ = ["willamette",
           "natural_images"]


def _load_image(image_name):
    return color.rgb2gray(plt.imread(os.path.join(data_dir, image_name)))


def willamette():
    return _load_image("willamette.jpg")


def alpine():
    return _load_image("alpine.jpg")


def alpine2():
    return _load_image('alpine2.jpg')


def waterfall():
    return _load_image("waterfall.jpg")

def cloud():
    return _load_image("cloud.jpg")


def store():
    return _load_image("store.jpeg")


def natural_images():
    return np.load(os.path.join(data_dir, "natural_images.npz"), allow_pickle=True)["arr_0"]
