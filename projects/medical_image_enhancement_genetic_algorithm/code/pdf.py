import numpy as np


def calculate_pdf(histogram, lower_bound, upper_bound):
    lower_bound = int(lower_bound)
    upper_bound = int(upper_bound)
    num_elements = (upper_bound - lower_bound) + 1
    num_pixels = 0
    for k in range(lower_bound, upper_bound + 1):
        num_pixels += histogram[k]
    pdf = np.zeros(num_elements, float)
    for k in range(lower_bound, upper_bound + 1):
        if num_pixels == 0:
            pdf[k - lower_bound] = 0.0
        else:
            pdf[k - lower_bound] = float(histogram[k]) / num_pixels
    return pdf


def calculate_pdfs_of_sub_images(histogram, lower_exposure_threshold, exposure_threshold, upper_exposure_threshold, L):
    first_lower_pdf = calculate_pdf(histogram, 0, lower_exposure_threshold)
    second_lower_pdf = calculate_pdf(histogram, lower_exposure_threshold + 1, exposure_threshold)
    first_upper_pdf = calculate_pdf(histogram, exposure_threshold + 1, upper_exposure_threshold)
    second_upper_pdf = calculate_pdf(histogram, upper_exposure_threshold + 1, L - 1)
    return first_lower_pdf, second_lower_pdf, first_upper_pdf, second_upper_pdf


def modify_pdf(p, pdf_sum):
    if pdf_sum + p == 0:
        return 0.0
    else:
        return p / (pdf_sum + p)


def calculate_modified_pdf(pdf):
    pdf_sum = np.sum(pdf)
    return tuple(map(lambda p: modify_pdf(p, pdf_sum), pdf))


def calculate_modified_pdfs(pdfs):
    return tuple(map(calculate_modified_pdf, pdfs))

