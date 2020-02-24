import os
import glob

import numpy as np
from skimage.io import imread


def load(directory):
    """Read images from a directory."""
    bad = [imread(file) for file in glob.glob(os.path.join(directory, 'bad', '*.jpeg'))]
    good = [imread(file) for file in glob.glob(os.path.join(directory, 'good', '*.jpeg'))]

    return good + bad, np.concatenate([np.zeros(len(bad)), np.ones(len(good))])
