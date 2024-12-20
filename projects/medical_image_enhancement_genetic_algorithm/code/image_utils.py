import numpy as np
from PIL import Image

from constants import GRAYSCALE_CONVERSION


def read_gray_scale_image(path):
    return np.array(Image.open(path).convert(GRAYSCALE_CONVERSION))
