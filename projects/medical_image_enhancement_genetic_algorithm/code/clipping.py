import numpy as np


def compute_clipping_threshold(histogram):
    c_median = np.median(histogram)
    c_mean = np.mean(histogram)
    return (c_mean + c_median) / 2


def clipper(x, clipping_threshold):
    if x > clipping_threshold:
        return int(clipping_threshold)
    else:
        return x


def clip_histogram(histogram):
    clipping_threshold = compute_clipping_threshold(histogram)
    return np.array([clipper(x, clipping_threshold) for x in histogram])
