from fitness import calculate_entropy, calculate_ambe, calculate_psnr
from image_similarity_measures.quality_metrics import fsim, ssim


def print_comparison(method, original_image_score, equalized_image_score):
    print(f"{method} -> Original image: {original_image_score} Equalized image: {equalized_image_score}")


def evaluate_image(original_image, equalized_image, original_histogram, equalized_histogram, L):
    print_comparison("Entropy", calculate_entropy(original_histogram, L), calculate_entropy(equalized_histogram, L))
    print(f"AMBE between images is {calculate_ambe(original_image, equalized_image)}")
    print(f"PSNR between images is {calculate_psnr(original_image, equalized_image)}")
    print(f"SSIM between images is {ssim(original_image, equalized_image)}")