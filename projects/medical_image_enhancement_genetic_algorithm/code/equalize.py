import numpy as np


def histogram_equalize(image, transfer_functions, lower_exposure_threshold, exposure_threshold, upper_exposure_threshold, L):
    equalized_image = np.zeros(image.shape, int)
    lower_exposure_threshold = int(lower_exposure_threshold)
    upper_exposure_threshold = int(upper_exposure_threshold)
    for (x, y), b in np.ndenumerate(image):
        if 0 <= b <= lower_exposure_threshold:
            equalized_image[x, y] = int(transfer_functions[0][b])
        elif lower_exposure_threshold + 1 <= b <= exposure_threshold:
            equalized_image[x, y] = int(transfer_functions[1][b - lower_exposure_threshold - 1])
        elif exposure_threshold + 1 <= b <= upper_exposure_threshold:
            equalized_image[x, y] = int(transfer_functions[2][b - exposure_threshold - 1])
        elif upper_exposure_threshold + 1 <= b <= L - 1:
            equalized_image[x, y] = int(transfer_functions[3][b - upper_exposure_threshold - 1])
    return equalized_image
