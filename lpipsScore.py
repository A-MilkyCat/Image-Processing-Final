import torch
import lpips
import cv2

# ------------------------------
# Load LPIPS model
# ------------------------------
lpips_model = lpips.LPIPS(net='alex')   # 可選 'vgg' or 'alex'
lpips_model = lpips_model.cuda()        # 使用 GPU

# ------------------------------
# Convert image to LPIPS tensor
# ------------------------------
def to_tensor(im):
    im = torch.tensor(im).permute(2, 0, 1).float() / 255.0
    im = im * 2 - 1   # [0,1] → [-1,1]
    return im.unsqueeze(0).cuda()

# ------------------------------
# Compute LPIPS between two images
# ------------------------------
def compute_lpips(img1_path, img2_path):
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)

    if img1 is None:
        raise FileNotFoundError(f"Cannot load: {img1_path}")
    if img2 is None:
        raise FileNotFoundError(f"Cannot load: {img2_path}")

    # Convert to RGB
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

    # Resize to match size
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

    t1 = to_tensor(img1)
    t2 = to_tensor(img2)

    score = lpips_model(t1, t2)
    return float(score.item())


# ------------------------------
# Main – compute 3-pair average
# ------------------------------
if __name__ == "__main__":

    content_list = [
        "./images/metrics/8.jpg",
        "./images/metrics/1.jpg",
        "./images/metrics/9.png"
    ]

    # stylized_list = [
    #     "./images/metrics/8j.png",
    #     "./images/metrics/1j.png",
    #     "./images/metrics/9j.png"
    # ]
    stylized_list = [
        "./images/metrics/8.png",
        "./images/metrics/1m.png",
        "./images/metrics/9_new_lap.png"
    ]
    scores = []

    print("\n===== LPIPS Evaluation (3 pairs) =====")

    for c_path, s_path in zip(content_list, stylized_list):
        print(f"\nComparing:\nContent : {c_path}\nStylized: {s_path}")
        score = compute_lpips(c_path, s_path)
        print(f"LPIPS Score = {score}")
        scores.append(score)

    avg_score = sum(scores) / len(scores)
    print("\n=====================================")
    print(f"Average LPIPS Score = {avg_score}")
    print("=====================================\n")
