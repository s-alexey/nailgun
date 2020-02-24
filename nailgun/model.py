from typing import NamedTuple

import cv2
import numpy as np
from skimage import morphology, measure
import tensorflow as tf
from tensorflow.keras import layers

from skimage.transform import resize

BAD, GOOD = 0, 1


class Prediction(NamedTuple):
    label: str
    score: float


nail_height, nail_width = 320, 320


def detect_nail(image):
    """Detect nail on an image using edge detection."""

    # detect nail edges
    edges = cv2.Canny(image, 50, 280, apertureSize=3)

    # enlarge them edge lines and fill gaps between them
    closed = morphology.dilation(edges, selem=np.ones((35, 35)))

    # find objects (connected components)
    labels = measure.label(closed)

    # output the biggest one
    return max(measure.regionprops(labels), key=lambda x: x.area)


def crop_nail(image, height, width, bbox=(150, 250, -50, -250)):
    """Detect and crop nail on an image."""

    y0, x0, y1, x1 = bbox
    region = detect_nail(image[y0:y1, x0:x1])

    y_min, x_min, y_max, x_max = region.bbox

    y_padd = (height - (y_max - y_min)) // 2
    x_padd = (width - (x_max - x_min)) // 2
    y0 = y_min - y_padd + y0
    x0 = x_min - x_padd + x0
    return image[y0:y0 + height, x0:x0 + width]


def preprocess_image(image):
    """Preprocess image before feeding it into a network."""
    return np.expand_dims(image, -1) / 255. - .5


def load(path):
    """Load model architecture and weights from a checkpoint."""
    model = create()
    model.load_weights(path)
    return model
    # return tf.keras.models.load_model(path)


def create(input_height=32, input_width=32):
    return tf.keras.Sequential([
        layers.InputLayer(input_shape=(input_height, input_width, 1)),
        layers.Conv2D(16, (3, 3), activation='relu'),
        layers.MaxPool2D((2, 2), strides=(2, 2)),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPool2D((2, 2), strides=(2, 2)),
        layers.BatchNormalization(),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])


def classify(model, image) -> Prediction:
    nail = crop_nail(image, height=nail_height, width=nail_width)
    image = resize(nail, model.input_shape[1:3])
    image = preprocess_image(image)
    image = np.expand_dims(image, 0)
    score = model.predict(image)[0]

    return Prediction(label='good' if score > .5 else 'bad', score=float(score))
