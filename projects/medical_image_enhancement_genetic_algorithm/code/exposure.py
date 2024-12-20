import numpy as np


def max_gray_value(image):
    return np.amax(image)


def calculate_exposure(histogram, L):
    histogram_sum = np.sum(histogram)
    normalize_sum = 0
    for k, h in np.ndenumerate(histogram):
        normalize_sum += k[0] * h
    factor = normalize_sum / histogram_sum
    return (1 / L) * factor


def calculate_exposure_threshold(histogram, L):
    return L * (1 - calculate_exposure(histogram, L))

