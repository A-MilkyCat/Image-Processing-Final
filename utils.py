import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision.transforms.functional import pil_to_tensor, resize
from PIL import Image
from models import my_loss_models as loss_models
# from models import loss_models
import cv2

#CHANGED
# ---------------------------------------------------
# NEW LOSS — Laplacian Pyramid Loss（修正 batch）
# ---------------------------------------------------
class LaplacianStyleLoss(nn.Module):
    def __init__(self, style_image, weight, num_levels=3):
        super().__init__()
        self.weight = weight
        self.num_levels = num_levels

        device = style_image.device

        # 建立 Gaussian kernel on correct device
        k = torch.tensor([1, 4, 6, 4, 1], dtype=torch.float32, device=device)
        k = torch.outer(k, k)
        k = k / k.sum()
        k = k.unsqueeze(0).unsqueeze(0)  # (1,1,5,5)
        self.gauss_kernel = k  # already on device

        # Resize style image once (避免 256 vs 512 mismatch)
        H = 256
        style_resized = F.interpolate(style_image, size=(H, H),
                                      mode="bilinear", align_corners=False)

        # 預先計算 style pyramid
        self.style_pyr = self.build_pyr(style_resized)

    def gauss_blur(self, img):
        # 深度卷積（groups = channels）
        c = img.size(1)
        k = self.gauss_kernel.to(img.device)
        k = k.repeat(c, 1, 1, 1)  # (c,1,5,5)

        return F.conv2d(img, k, padding=2, groups=c)

    def build_pyr(self, img):
        pyr = []
        cur = img
        for _ in range(self.num_levels):
            blurred = self.gauss_blur(cur)
            lap = cur - blurred
            pyr.append(lap)
            cur = F.interpolate(cur, scale_factor=0.5, mode="bilinear",
                                align_corners=False)
        return pyr

    def forward(self, gen_img):

        # 尺寸對齊 style pyramid 第一層
        H, W = self.style_pyr[0].shape[-2:]
        gen_img = F.interpolate(gen_img, size=(H, W),
                                mode="bilinear", align_corners=False)

        gen_pyr = self.build_pyr(gen_img)

        total = 0
        for gl, sl in zip(gen_pyr, self.style_pyr):

            # 批大小對齊
            if gl.size(0) != sl.size(0):
                sl = sl.repeat(gl.size(0), 1, 1, 1)

            total += F.mse_loss(gl, sl)

        self.loss = self.weight * total
        return total
    
def create_whitespace_mask(img, threshold=0.95):
    """
    img: (B,3,H,W) normalized image
    threshold: 白色區域的亮度界線 (0~1)
    回傳 mask: (B,1,H,W), 白色越亮 → mask 越小
    """
    # 反正 gram 要用 feature map，所以這邊 mask 粗一點沒差
    gray = 0.299 * img[:,0:1] + 0.587 * img[:,1:2] + 0.114 * img[:,2:3]

    # 白色 → 高灰階 → mask → 越小（降低 style weight）
    mask = 1.0 - torch.clamp((gray - threshold) / (1 - threshold), 0, 1)
    # mask 範圍：白色區域 ~0.0，正常區域 ~1.0
    return mask

class StyleLoss(nn.Module):
    def __init__(self, target_feat, weight, batch_size=4):
        super(StyleLoss, self).__init__()
        self.target_gram = (
            get_gram_matrix(target_feat).detach().repeat(batch_size, 1, 1)
        )
        self.weight = weight
        self.mode = "capture"

    def forward(self, gen_feature):
        # if self.mode == "loss":
        #     gram_matrix = get_gram_matrix(gen_feature)
        #     self.loss = self.weight * F.mse_loss(gram_matrix, self.target_gram)
        # if self.mode == "loss":
        #     gram_gen = get_gram_matrix(gen_feature)
        #     diff = (gram_gen - self.target_gram) ** 2

        #     # compute whiteness ratio from the original input image
        #     with torch.no_grad():
        #         mask = create_whitespace_mask(self.input_image_original)
        #         white_ratio = 1.0 - mask.mean()  # 白色越多 → 留白越強

        #     white_strength = 1.0  # 可調
        #     style_weight = 1.0 - white_ratio * white_strength
        #     style_weight = torch.clamp(style_weight, min=0.5)  # <-- 加這行即可

        #     self.loss = self.weight * diff.mean() * style_weight
        if self.mode == "loss":
            B, C, H, W = gen_feature.shape

            # ---- 1. Compute Gram from generated feature ----
            gram_gen = get_gram_matrix(gen_feature)
            diff = (gram_gen - self.target_gram) ** 2  # (B, C, C)

            # ---- 2. Create spatial mask (background=0, subject=1) ----
            with torch.no_grad():
                mask = create_whitespace_mask(self.input_image_original)  # (B,1,H_full,W_full)
                mask = F.interpolate(mask, size=(H, W), mode="bilinear", align_corners=False)
                mask = mask.clamp(0, 1)

                # soften edges (讓主體邊緣更自然)
                mask = F.avg_pool2d(mask, kernel_size=5, stride=1, padding=2)

            # ---- 3. Apply spatial weighting (真正的局部留白!!) ----
            # diff 是全域 Gram loss → 沒 spatial 維度，所以我們需要一個 scalar weight
            # 方法：用 spatial mask 的平均值來代表這層 feature 的有效內容比例
            spatial_weight = mask.mean()   # 背景多→小；內容多→大

            # optional: 加強敏感度（可調 0.6 ~ 1.0）
            sensitivity = 0.7
            spatial_weight = spatial_weight.pow(sensitivity)

            # 避免風格太弱（不要低於 0.3）
            spatial_weight = torch.clamp(spatial_weight, min=0.3)

            # ---- 4. Final style loss ----
            weighted_style = diff.mean() * spatial_weight

            self.loss = self.weight * weighted_style

        return gen_feature


class ContentLoss(nn.Module):
    def __init__(self, weight):
        super(ContentLoss, self).__init__()
        self.weight = weight
        self.mode = "capture"
        self.loss = 0.0

    def forward(self, gen_feature):
        if self.mode == "capture":
            self.target_feature = gen_feature.detach()
        elif self.mode == "loss":
            self.loss = self.weight * F.mse_loss(gen_feature, self.target_feature)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        return gen_feature


def get_gram_matrix(featmaps):
    b, c, h, w = featmaps.shape
    featmaps = featmaps.view(b, c, h * w)
    output = (featmaps @ featmaps.transpose(1, 2)).div(c * h * w)
    return output


# Total variation loss
class TVLoss(nn.Module):
    def __init__(self, weight):
        super(TVLoss, self).__init__()
        self.weight = weight

    def forward(self, featmaps):
        self.x_diff = featmaps[:, :, 1:, :] - featmaps[:, :, :-1, :]
        self.y_diff = featmaps[:, :, :, 1:] - featmaps[:, :, :, :-1]
        self.loss = self.weight * (
            torch.sum(torch.abs(self.x_diff)) + torch.sum(torch.abs(self.y_diff))
        )
        return featmaps

def preprocess_batch(images: torch.Tensor, mean, std):
    """
    Preprocess a batch of images for the style transfer model.
    Note that the mean and std are stored inside the loss model.
    """
    img = images.float().div(255)
    img = (images - mean) / std

    return img

def deprocess_batch(images: torch.Tensor, mean, std):
    """
    De-process a batch of images for the style transfer model.
    Note that the mean and std are stored inside the loss model.
    """
    img = images * std + mean
    img = img.clamp(0, 1)
    return img

def preprocess_image(image: torch.Tensor, mean, std):
    """
    Preprocess an image for the style transfer model.
    Note that the mean and std are stored inside the loss model.
    """
    img = image.unsqueeze(0)
    return preprocess_batch(img, mean, std)

def deprocess_image(image: torch.Tensor, mean, std):
    """
    De-process an image for the style transfer model.
    Note that the mean and std are stored inside the loss model.
    """
    img = deprocess_batch(image, mean, std)
    return img.squeeze(0)


def display_images_in_a_grid(
    images: list[np.ndarray], cols: int = 5, titles: list[str] = None
):
    """Display a list of images in a grid."""
    assert (
        (titles is None) or (len(images) == len(titles)))
    n_images = len(images)
    if titles is None:
        titles = ["Image (%d)" % i for i in range(1, n_images + 1)]
    fig = plt.figure()
    for n, (image, title) in enumerate(zip(images, titles)):
        a = fig.add_subplot(int(np.ceil(n_images / float(cols))), cols, n + 1)
        if image.ndim == 2:
            plt.gray()
        plt.imshow(image)
        a.set_title(title)
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_images)
    plt.show()

def apply_style_grid(model, path_to_image, paths_to_models):
    """
    Produces a grid of images in matplotlib for the outputs of multiple models on the same image.
    I used this to compare multiple checkpoints of the same model.
    """

    img = resize(
        pil_to_tensor((Image.open(path_to_image)).convert("RGB"))
        .unsqueeze(0)
        .float()
        .div(255),
        512,
    )
    transformation_model = model.TransformationModel()

    # code to load pretrained models
    models = []
    for path in paths_to_models:
        checkpoint = torch.load(path, map_location=torch.device("cpu"))
        transformation_model.load_state_dict(checkpoint["model_state_dict"])
        models.append(transformation_model.eval())

    mean, std = loss_models.VGG16Loss.MEAN, loss_models.VGG16Loss.STD
    gen_images = []
    for model in models:
        gen_image = model(img)
        gen_image = gen_image * std + mean
        gen_image = gen_image.clamp(0, 1)
        gen_image = gen_image.squeeze(0).detach().cpu().numpy().transpose(1, 2, 0)
        gen_images.append(gen_image)

    # display images in a grid
    display_images_in_a_grid(gen_images, 4, paths_to_models)
    