import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim


# ------------------------------
# Compute SSIM between two images
# ------------------------------
def compute_ssim(img1_path, img2_path):
    # Read images
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)

    if img1 is None:
        raise FileNotFoundError(f"Cannot load: {img1_path}")
    if img2 is None:
        raise FileNotFoundError(f"Cannot load: {img2_path}")

    # Convert to grayscale for SSIM
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Match resolution
    if gray1.shape != gray2.shape:
        gray2 = cv2.resize(gray2, (gray1.shape[1], gray1.shape[0]))

    # Compute SSIM
    score = ssim(gray1, gray2, data_range=255)
    return float(score)


# ------------------------------
# Main – evaluate 3 pairs
# ------------------------------
if __name__ == "__main__":

    content_list = [
        "./images/metrics/8.jpg",
        "./images/metrics/1.jpg",
        "./images/metrics/9.png"
    ]

    # stylized_list = [
    #     "./images/metrics/8.png",
    #     "./images/metrics/1m.png",
    #     "./images/metrics/9_new_lap.png"
    # ]
    stylized_list = [
        "./images/metrics/8j.png",
        "./images/metrics/1j.png",
        "./images/metrics/9j.png"
    ]
    scores = []

    print("\n===== SSIM Evaluation (3 pairs) =====")

    for c_path, s_path in zip(content_list, stylized_list):
        print(f"\nComparing:\nContent : {c_path}\nStylized: {s_path}")

        score = compute_ssim(c_path, s_path)
        print(f"SSIM Score = {score}")

        scores.append(score)

    avg_score = sum(scores) / len(scores)

    print("\n=====================================")
    print(f"Average SSIM Score = {avg_score}")
    print("=====================================\n")
