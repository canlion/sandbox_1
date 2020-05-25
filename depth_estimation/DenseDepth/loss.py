import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class DenseDepthLoss(nn.Module):
    """Calculate DenseDepth loss.

    lambda * L1_loss(prediction, target) + L1_loss(prediction_gradient, target_gradient) + SSIM(prediction, target)
    """

    def __init__(self):
        super().__init__()
        self.gradient_module = ImageGradient()
        self.ssim_module = SSIM(1000.-10.)

    def forward(self, pred, target):
        l1_loss = F.l1_loss(pred, target).mean()

        pred_grad = self.gradient_module(pred)
        target_grad = self.gradient_module(target)
        grad_loss = F.l1_loss(pred_grad, target_grad)

        ssim_loss = (1. - self.ssim_module(pred, target)) / 2.

        return .1 * l1_loss + grad_loss + ssim_loss


class ImageGradient(nn.Module):
    """Calculate depth-map gradient."""

    def __init__(self):
        super().__init__()
        self.gradient_kernel = torch.tensor([[[0, 0, 0], [-1, 0, 1], [0, 0, 0]],
                                             [[0, -1, 0], [0, 0, 0], [0, 1, 0]]],
                                            dtype=torch.float, requires_grad=False)
        self.gradient_kernel = self.gradient_kernel.view(2, 1, 3, 3)

    def forward(self, depth):
        kernel = self.gradient_kernel.to(depth.device)
        return F.conv2d(depth, kernel, stride=1, padding=1)


class SSIM(nn.Module):
    """Calculate the structure similarity."""
    def __init__(self, dynamic_range=9.6, exponents=(1, 1, 1), sigma=1.5, window_size=11):
        super().__init__()
        self.C1 = (.01 * dynamic_range) ** 2
        self.C2 = (.03 * dynamic_range) ** 2
        self.C3 = self.C2 / 2

        self.exponents = exponents
        self.gaussian_kernel = torch.tensor(self.make_gaussian_filter(window_size, sigma), dtype=torch.float)

    def forward(self, pred, target):
        mu_kernel = self.gaussian_kernel.to(pred.device)

        mu_pred = F.conv2d(pred, mu_kernel)
        mu_target = F.conv2d(target, mu_kernel)
        var_pred = F.conv2d(pred.square(), mu_kernel) - mu_pred.square()
        var_target = F.conv2d(target.square(), mu_kernel) - mu_target.square()
        cov_pred_target = F.conv2d(target * pred, mu_kernel) - mu_target * mu_pred

        l = (2 * mu_pred * mu_target + self.C1) / (mu_pred.square() + mu_target.square() + self.C1)
        c = (2 * var_pred.sqrt() * var_target.sqrt() + self.C2) / (var_pred + var_target + self.C2)
        s = (cov_pred_target + self.C3) / (var_pred.sqrt() * var_target.sqrt() + self.C3)

        alpha, beta, gamma = self.exponents
        ssim = l.pow(alpha) * c.pow(beta) * s.pow(gamma)

        return ssim.mean()

    @staticmethod
    def make_gaussian_filter(window_size, sigma):
        assert (window_size >= 3) and (window_size % 2 == 1)
        k = (window_size-1)//2
        probs = [np.exp(-z * z / (2 * (sigma ** 2))) / np.sqrt(2 * np.pi * (sigma ** 2)) for z in range(-k, k + 1)]
        kernel = np.outer(probs, probs).reshape(1, 1, window_size, window_size)
        return kernel


if __name__ == '__main__':
    from PIL import Image

    # load ssim test images
    origin = np.array(Image.open('ssim_origin.gif').convert('L'))[None, None, ...]
    blur = np.array(Image.open('ssim_blur.gif').convert('L'))[None, None, ...]
    square = np.array(Image.open('ssim_square.gif').convert('L'))[None, None, ...]
    gray = np.array(Image.open('ssim_gray.gif').convert('L'))[None, None, ...]

    ssim_loss = SSIM(255)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    tensor_org = torch.tensor(origin, dtype=torch.float, device=device)
    tensor_blu = torch.tensor(blur, dtype=torch.float, device=device)
    tensor_sq = torch.tensor(square, dtype=torch.float, device=device)
    tensor_gray = torch.tensor(gray, dtype=torch.float, device=device)

    print(ssim_loss(tensor_org, tensor_org))
    print(ssim_loss(tensor_org, tensor_blu))
    print(ssim_loss(tensor_org, tensor_sq))
    print(ssim_loss(tensor_org, tensor_gray))

    #  https://www.google.com/url?sa=i&url=http%3A%2F%2Fwww.fmwconcepts.com%2Fimagemagick%2Fssim%2Findex.php&psig=AOvV \
    #  aw1HMWs31zuP4uPE8UP3AOTs&ust=1589882577863000&source=images&cd=vfe&ved=0CAIQjRxqFwoTCKjYvqaUvekCFQAAAAAdAAAAABAJ
    # small diff..
