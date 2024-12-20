import numpy as np
import cv2
from geneticalgorithm import geneticalgorithm as ga

from cdf import calculate_cdfs, calculate_transfer_functions
from clipping import clip_histogram
from equalize import histogram_equalize
from evaluate import evaluate_image
from exposure import calculate_exposure_threshold, max_gray_value
from fitness import calculate_fitness
from histogram import get_histogram
from image_utils import read_gray_scale_image
from pdf import calculate_pdfs_of_sub_images, calculate_modified_pdfs
from plotting_utils import plot_histogram_with_thresholds


def equalize_image(lower_threshold, upper_threshold):
    clipped_histogram = clip_histogram(original_histogram)
    pdfs = calculate_pdfs_of_sub_images(clipped_histogram, lower_threshold, exposure_threshold,
                                        upper_threshold, L)
    modified_pdfs = calculate_modified_pdfs(pdfs)
    cdfs = calculate_cdfs(modified_pdfs)
    transfer_functions = calculate_transfer_functions(cdfs, lower_threshold, exposure_threshold,
                                                      upper_threshold, L)
    return histogram_equalize(original_image, transfer_functions, lower_threshold,
                              exposure_threshold, upper_threshold, L)


def fitness_function(exposure_thresholds):
    lt, ut = exposure_thresholds
    resulting_image = equalize_image(lt, ut)
    resulting_histogram = get_histogram(resulting_image)
    return -calculate_fitness(original_image, resulting_image, resulting_histogram)


def fitness_gahe_function(exposure_thresholds):
    lt, ut = exposure_thresholds
    resulting_image = equalize_image(lt, ut)
    resulting_histogram = get_histogram(resulting_image)
    return -calculate_fitness(original_image, resulting_image, resulting_histogram)


def equalize_with_fitness(fitness_func, ga_parameters):
    var_bound = np.array([[0, exposure_threshold], [exposure_threshold + 1, L - 1]])
    model = ga(function=fitness_func, dimension=2, variable_type='int', variable_boundaries=var_bound,
               algorithm_parameters=ga_parameters)
    model.run()
    lower_exposure_threshold, upper_exposure_threshold = model.output_dict['variable']
    plot_histogram_with_thresholds(original_histogram,
                                   [exposure_threshold, lower_exposure_threshold, upper_exposure_threshold])
    return lower_exposure_threshold, upper_exposure_threshold, equalize_image(lower_exposure_threshold,
                                                                              upper_exposure_threshold)


original_image = read_gray_scale_image('images/brain mri.png')
print(f'Size of original image: {original_image.shape}')
L = int(max_gray_value(original_image))
original_histogram = get_histogram(original_image)
plot_histogram_with_thresholds(original_histogram)
exposure_threshold = int(calculate_exposure_threshold(original_histogram, L))
print(f'Exposure threshold is at : {exposure_threshold}')

## HE
equalized_image_he = cv2.equalizeHist(original_image)
equalized_histogram_he = get_histogram(equalized_image_he)
plot_histogram_with_thresholds(equalized_histogram_he)
cv2.imwrite("equalized_he.png", equalized_image_he)
evaluate_image(original_image, equalized_image_he, original_histogram, equalized_histogram_he, L)

## GAAHE
gahe_ga_param = {
    'max_num_iteration': 50,
    'population_size': 50,
    'mutation_probability': 0.01,
    'elit_ratio': 0.0,
    'crossover_probability': 0.8,
    'parents_portion': 0.2,
    'crossover_type': 'uniform',
    'max_iteration_without_improv': None
}
lower_exposure_threshold_gahe, upper_exposure_threshold_gahe, equalized_image_gahe = equalize_with_fitness(
    fitness_gahe_function, gahe_ga_param)
equalized_histogram_gahe = get_histogram(equalized_image_gahe)
cv2.imwrite("equalized_gahe.png", equalized_image_gahe)
evaluate_image(original_image, equalized_image_gahe, original_histogram, equalized_histogram_gahe, L)

## Current
ga_param = {
    'max_num_iteration': 60,
    'population_size': 45,
    'mutation_probability': 0.01,
    'elit_ratio': 0.0,
    'crossover_probability': 0.8,
    'parents_portion': 0.3,
    'crossover_type': 'uniform',
    'max_iteration_without_improv': None
}
lower_exposure_threshold, upper_exposure_threshold, equalized_image = equalize_with_fitness(fitness_function, ga_param)
equalized_histogram = get_histogram(equalized_image)
cv2.imwrite("equalized.png", equalized_image)
evaluate_image(original_image, equalized_image, original_histogram, equalized_histogram, L)
