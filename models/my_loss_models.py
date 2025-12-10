import torch
from torchvision.models import vgg16, VGG16_Weights
import torch.nn as nn
from utils import StyleLoss, ContentLoss, TVLoss, LaplacianStyleLoss


class VGG16Loss(nn.Module):
    """
    This model returns the loss of the style and content of the input image
    """

    # these assume an pixel values in the range [0, 1]
    MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    DEFAULT_CONTENT_LAYERS = ["relu2_2"]
    DEFAULT_STYLE_LAYERS = ["relu1_2", "relu2_2", "relu3_3", "relu4_3"]
    
    def __init__(
        self,
        style_img,
        content_weight=1e0,
        style_weight=1e5,
        tv_weight=0,
        gabor_weight=1.0,
        lap_weight=0.05, # CHANGED
        content_layers=DEFAULT_CONTENT_LAYERS,
        style_layers=DEFAULT_STYLE_LAYERS,
        batch_size=4,
        pooling="max",
        device="cpu",
    ):

        super(VGG16Loss, self).__init__()
        features = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features.eval().to(device)
        features.requires_grad_(False)

        # load stuff to device
        self.MEAN = self.MEAN.to(device)
        self.STD = self.STD.to(device)
        self.style_img = torch.nn.functional.interpolate(
            style_img, size=(256, 256), mode="bilinear", align_corners=False
        )
        style_img = style_img.to(device)
        self.style_img_source = style_img.clone().detach()

        self.gabor_losses = []
        self.laplacian_losses = []

        self.content_losses = []
        self.style_losses = []
        self.total_content_loss = 0.0
        self.total_style_loss = 0.0

        self.tv_loss = TVLoss(tv_weight)
        self.layers = nn.Sequential()

        pool_cnt, relu_count, conv_count = 1, 1, 1
        for i in range(len(features)):
            x = features[i]
            if isinstance(x, nn.Conv2d):
                name = f"conv{pool_cnt}_{conv_count}"
                conv_count += 1
            elif isinstance(x, nn.ReLU):
                name = f"relu{pool_cnt}_{relu_count}"
                relu_count += 1
            else:
                name = f"pool{pool_cnt}"
                if pooling == "avg":
                    x = nn.AvgPool2d(2, 2)

                relu_count = 1
                conv_count = 1
                pool_cnt += 1

            self.layers.add_module(name, x)

            style_img = x(style_img)

            if name in style_layers:
                # style feature at this layer
                target_feat = style_img.clone().detach()

                # Gram style loss
                style_loss = StyleLoss(target_feat, style_weight, batch_size)
                self.layers.add_module(f"{name}_style_loss", style_loss)
                self.style_losses.append(style_loss)

                # if len(self.gabor_losses) == 0:
                #     self.gabor_losses.append(GaborLoss(self.style_img_source, gabor_weight))
                if len(self.laplacian_losses) == 0:
                    self.laplacian_losses.append(LaplacianStyleLoss(self.style_img_source, lap_weight))

                style_layers.remove(name)

            if name in content_layers:
                loss_module = ContentLoss(content_weight)
                self.layers.add_module(f"{name}_content_loss", loss_module)
                self.content_losses.append(loss_module)
                content_layers.remove(name)

            # making sure it is cut off at the last loss layer to avoid unnecesarry computations
            if len(style_layers) == 0 and len(content_layers) == 0:
                break

    def switch_mode(self, mode):
        for content_layer in self.content_losses:
            content_layer.mode = mode
        for style_layer in self.style_losses:
            style_layer.mode = mode

    def forward(self, input, content_img):
        # 給 StyleLoss 使用的原始輸入（normalized）
        for sl in self.style_losses:
            sl.input_image_original = input.clone().detach()
        self.switch_mode("capture")
        self.layers(content_img)

        self.switch_mode("loss")
        self.tv_loss(input)
        self.layers(input)

        # for g in self.gabor_losses:
        #     g(input)
        for lap in self.laplacian_losses:
            lap(input)

        self.total_content_loss = sum([x.loss for x in self.content_losses])
        self.total_style_loss = sum([x.loss for x in self.style_losses])
        # self.total_gabor_loss = sum([l.loss for l in self.gabor_losses])
        self.total_laplacian_loss = sum([l.loss for l in self.laplacian_losses])
        total_loss = (
            self.total_style_loss +
            self.total_content_loss +
            self.tv_loss.loss +
            # self.total_gabor_loss +
            self.total_laplacian_loss
        )

        return total_loss