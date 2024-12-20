import numpy as np


def calculate_cdf(pdf):
    return np.cumsum(pdf)


def calculate_cdfs(pdfs):
    return tuple(map(calculate_cdf, pdfs))


def calculate_transfer_functions(cdfs, lower_exposure_threshold, exposure_threshold, upper_exposure_threshold, L):
    first_lower_transfer_function = cdfs[0] * lower_exposure_threshold
    second_lower_transfer_function = (lower_exposure_threshold + 1) + cdfs[1] * (exposure_threshold - (lower_exposure_threshold + 1))
    first_upper_transfer_function = (exposure_threshold + 1) + cdfs[2] * (upper_exposure_threshold - (exposure_threshold + 1))
    second_upper_transfer_function = (upper_exposure_threshold + 1) + cdfs[3] * (L - (upper_exposure_threshold + 1))
    return first_lower_transfer_function, second_lower_transfer_function,first_upper_transfer_function, second_upper_transfer_function

