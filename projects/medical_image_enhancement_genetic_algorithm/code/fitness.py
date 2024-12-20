import math

import numpy as np
from skimage import filters, feature

from exposure import max_gray_value
from pdf import calculate_pdf


def calculate_fitness(original_image, equalized_image, equalized_histogram):
    cf1 = calculate_texture_parameter_fitness_function(equalized_image, equalized_histogram)
    cf2 = calculate_edge_parameter_fitness_function(equalized_image)
    cf3 = calculate_psnr(original_image, equalized_image)
    cf4 = calculate_ambe(original_image, equalized_image)
    return (0.25 * cf1) + (0.25 * cf2) + (0.25 * cf3) + (0.25 * cf4)


def calculate_fitness_gahe(original_image, equalized_image, equalized_histogram):
    cf1 = calculate_texture_parameter_fitness_function(equalized_image, equalized_histogram)
    cf2 = calculate_edge_parameter_fitness_function(equalized_image)
    cf3 = calculate_psnr(original_image, equalized_image)
    return (0.33 * cf1) + (0.33 * cf2) + (0.33 * cf3)


def calculate_ambe(original_image, equalized_image):
    mean_brightness_original = np.mean(original_image)
    mean_brightness_equalized = np.mean(equalized_image)
    return math.fabs(mean_brightness_original - mean_brightness_equalized)


def calculate_mean_squared_error(original_image, equalized_image):
    r, c = original_image.shape
    error = 0
    for (i, j), b in np.ndenumerate(original_image):
        error += (b - equalized_image[i, j]) ** 2
    return (1 / (r * c)) * error


def calculate_psnr(original_image, equalized_image):
    mse = calculate_mean_squared_error(original_image, equalized_image)
    return 10 * math.log((255 ** 2) / mse, 10)


def calculate_entropy(histogram, n):
    pdf = calculate_pdf(histogram, 0, n - 1)
    entropy = 0
    for pd in pdf:
        if pd == 0:
            continue
        entropy += pd * math.log(pd, 2)
    return -entropy


def calculate_texture_parameter_fitness_function(equalized_image, equalized_histogram):
    glcm = feature.greycomatrix(equalized_image, [1], [0], levels=256)
    contrast = feature.greycoprops(glcm, 'contrast')[0, 0]
    energy = feature.greycoprops(glcm, 'energy')[0, 0]
    n = max_gray_value(equalized_image)
    entropy = calculate_entropy(equalized_histogram, n)
    result = (contrast * math.exp(entropy)) / energy
    if result == 0:
        return 0
    else:
        return math.log(result)


def calculate_edge_parameter_fitness_function(equalized_image):
    sobel = calculate_sobel(equalized_image)
    sum_of_intensity = np.sum(sobel)
    number_of_edges = np.count_nonzero(sobel)
    w, x = equalized_image.shape
    first_term = math.log(sum_of_intensity)
    second_term = (number_of_edges / (w * x))
    if first_term < 0:
        return first_term * second_term
    else:
        return math.log(first_term) * second_term


def calculate_sobel(equalized_image):
    return filters.sobel(equalized_image)
