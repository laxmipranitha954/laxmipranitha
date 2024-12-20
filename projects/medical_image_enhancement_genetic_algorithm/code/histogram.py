import numpy as np

from constants import MAX_BRIGHTNESS


def get_histogram(image):
    histogram = np.zeros(MAX_BRIGHTNESS + 1, int)
    for row in image:
        for value in row:
            histogram[value] += 1
    return histogram



