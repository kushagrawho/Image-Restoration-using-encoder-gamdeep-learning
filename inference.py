
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import cv2
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import random

# =========================
# 🔴 REQUIRED BLOCKS (ADD THESE)
# =========================

class PartialConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, bias=True):

        super().__init__()

        self.kernel_size = kernel_size * kernel_size

        self.input_conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            bias=bias
        )

        self.mask_conv = nn.Conv2d(
            1,
            1,
            kernel_size,
            stride,
            padding,
            bias=False
        )

        torch.nn.init.constant_(self.mask_conv.weight, 1.0)

        for param in self.mask_conv.parameters():
            param.requires_grad = False


    def forward(self, x, mask):

        # ensure mask is single channel
        if mask.shape[1] != 1:
            mask = mask[:, :1, :, :]

        # expand mask to match channels
        mask_expanded = mask.repeat(1, x.shape[1], 1, 1)

        # apply mask to input
        x_masked = x * mask_expanded

        # standard convolution
        output = self.input_conv(x_masked)

        with torch.no_grad():

            mask_sum = self.mask_conv(mask)

        # compute mask ratio
        mask_ratio = self.kernel_size / (mask_sum + 1e-8)
        mask_ratio = mask_ratio * (mask_sum > 0)

        # apply correction
        output = output * mask_ratio

        # new binary mask
        new_mask = (mask_sum > 0).float()

        return output, new_mask

class PConvBlock(nn.Module):

    def __init__(self, in_c, out_c):
        super().__init__()

        self.pconv = PartialConv2d(
            in_c,
            out_c,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )

        self.norm = nn.InstanceNorm2d(out_c, affine=True)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x, mask):

        out, new_mask = self.pconv(x, mask)


        eps = 1e-8

        # expand mask to channels
        mask_expanded = new_mask.repeat(1, out.shape[1], 1, 1)

        # compute mean only on valid pixels
        valid_sum = mask_expanded.sum(dim=[2,3], keepdim=True) + eps
        mean = (out * mask_expanded).sum(dim=[2,3], keepdim=True) / valid_sum

        # compute variance only on valid pixels
        var = ((out - mean) * mask_expanded).pow(2).sum(dim=[2,3], keepdim=True) / valid_sum

        # normalize
        out = (out - mean) / torch.sqrt(var + eps)

        # apply affine params
        if self.norm.affine:
            out = out * self.norm.weight.view(1,-1,1,1) + self.norm.bias.view(1,-1,1,1)

        # apply activation ONLY on valid regions
        out = self.act(out)

        # keep invalid regions zeroed
        out = out * mask_expanded

        return out, new_mask


class SelfAttention(nn.Module):

    def __init__(self, in_dim):
        super().__init__()

        self.query = nn.Conv2d(in_dim, in_dim // 8, 1)
        self.key = nn.Conv2d(in_dim, in_dim // 8, 1)
        self.value = nn.Conv2d(in_dim, in_dim, 1)

        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):

        B, C, H, W = x.size()

        query = self.query(x).view(B, -1, H*W).permute(0,2,1)
        key = self.key(x).view(B, -1, H*W)

        attention = torch.bmm(query, key)
        attention = F.softmax(attention, dim=-1)

        value = self.value(x).view(B, -1, H*W)

        out = torch.bmm(value, attention.permute(0,2,1))
        out = out.view(B, C, H, W)

        out = self.gamma*out + x

        return out, attention

class AttentionGate(nn.Module):

    def __init__(self, F_g, F_l, F_int):
        super().__init__()

        self.W_g = nn.Conv2d(F_g, F_int, kernel_size=1)
        self.W_x = nn.Conv2d(F_l, F_int, kernel_size=1)

        self.psi = nn.Conv2d(F_int, 1, kernel_size=1)

        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, g, x):

        g1 = self.W_g(g)
        x1 = self.W_x(x)

        psi = self.relu(g1 + x1)
        psi = self.sigmoid(self.psi(psi))

        return x * psi


# =========================
# 🔴 YOUR GENERATOR
# =========================

class UNetGenerator(nn.Module):

    def __init__(self):
        super().__init__()

        self.pool = nn.MaxPool2d(2)

        # -------- Encoder --------
        self.down1 = PConvBlock(4, 64)
        self.down2 = PConvBlock(64, 128)
        self.down3 = PConvBlock(128, 256)

        # -------- Bottleneck --------
        self.bottleneck = PConvBlock(256, 512)

        self.attn = SelfAttention(512)

        self.skip_attn3 = AttentionGate(256, 256, 128)
        self.skip_attn2 = AttentionGate(128, 128, 64)
        self.skip_attn1 = AttentionGate(64, 64, 32)

        # -------- Decoder --------
        self.up3 = nn.ConvTranspose2d(512, 256, 2, 2)
        self.dec3 = PConvBlock(512, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.dec2 = PConvBlock(256, 128)

        self.up1 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.dec1 = PConvBlock(128, 64)

        # -------- Refinement --------
        self.refine = nn.Conv2d(64, 64, 3, padding=1)

        # -------- Output --------
        self.final = nn.Sequential(
            nn.Conv2d(64, 3, 3, padding=1),
            nn.Tanh()
        )

    def forward(self, img, mask):

        x = torch.cat([img, mask], dim=1)

        d1, m1 = self.down1(x, mask)

        p1 = self.pool(d1)
        m1_down = F.interpolate(m1, scale_factor=0.5, mode="nearest")

        d2, m2 = self.down2(p1, m1_down)

        p2 = self.pool(d2)
        m2_down = F.interpolate(m2, scale_factor=0.5, mode="nearest")

        d3, m3 = self.down3(p2, m2_down)

        p3 = self.pool(d3)
        m3_down = F.interpolate(m3, scale_factor=0.5, mode="nearest")

        # -------- Bottleneck --------
        bn, m4 = self.bottleneck(p3, m3_down)

        bn, attn_map = self.attn(bn)

        # -------- Decoder Level 3 --------
        up3 = self.up3(bn)

        mask_up3 = F.interpolate(m4, scale_factor=2, mode="nearest")

        d3 = self.skip_attn3(up3, d3)
        up3 = torch.cat([up3, d3], dim=1)
        mask3 = torch.max(mask_up3, m3)

        up3, m_dec3 = self.dec3(up3, mask3)

        # -------- Decoder Level 2 --------
        up2 = self.up2(up3)

        mask_up2 = F.interpolate(m_dec3, scale_factor=2, mode="nearest")
        d2 = self.skip_attn2(up2, d2)
        up2 = torch.cat([up2, d2], dim=1)

        mask2 = torch.max(mask_up2, m2)

        up2, m_dec2 = self.dec2(up2, mask2)

        # -------- Decoder Level 1 --------
        up1 = self.up1(up2)

        mask_up1 = F.interpolate(m_dec2, scale_factor=2, mode="nearest")
        d1 = self.skip_attn1(up1, d1)
        up1 = torch.cat([up1, d1], dim=1)

        mask1 = torch.max(mask_up1, m1)

        up1, _ = self.dec1(up1, mask1)

        # -------- Refinement --------
        up1 = F.relu(self.refine(up1))

        # -------- Output --------
        return self.final(up1), attn_map


# =========================
# 🔴 LOAD MODEL (FIXED)
# =========================

def load_model():
    model = UNetGenerator()   # ✅ FIXED
    model.load_state_dict(torch.load("generator.pth", map_location="cpu"))
    model.eval()
    return model


# =========================
# 🔴 MASK + PREDICT + METRICS
# =========================

def generate_mask(size=128):

    mask = np.zeros((size, size), np.float32)

    num_cracks = 3

    for _ in range(num_cracks):

        x = np.random.randint(0, size)
        y = np.random.randint(0, size)

        angle = np.random.uniform(0, 2*np.pi)
        length = np.random.randint(10, 30)

        for i in range(length):

            angle += np.random.uniform(-0.2, 0.2)

            dx = int(np.round(np.cos(angle)))
            dy = int(np.round(np.sin(angle)))

            new_x = np.clip(x + dx, 0, size-1)
            new_y = np.clip(y + dy, 0, size-1)

            cv2.line(mask, (x, y), (new_x, new_y), 1, 1)

            x, y = new_x, new_y

    return mask


def apply_mask(image, mask):

    img = np.array(image).astype(np.float32) / 255.0

    # 🔴 IMPORTANT: mask must be 1=valid, 0=hole
    mask = (mask > 0).astype(np.float32)

    mask = 1 - mask   # invert to match training

    mask = mask[..., None]  # expand channel

    patched = img * mask   # ✅ EXACT training logic

    patched = (patched * 255).astype(np.uint8)

    return Image.fromarray(patched)


def predict(image, mask, model):

    transform = transforms.Compose([
        transforms.Resize((128,128)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
    ])

    # 🔴 convert ORIGINAL image (not masked)
    img = transform(image)

    # mask processing
    mask = (mask > 0).astype(np.float32)
    mask = 1 - mask
    mask = torch.tensor(mask).unsqueeze(0).unsqueeze(0).float()

    img = img.unsqueeze(0)

    # 🔥 CREATE PATCHED EXACTLY LIKE TRAINING
    patched = img * mask

    with torch.no_grad():
        prediction, _ = model(patched, mask)

    # reconstruction
    reconstructed = prediction * (1 - mask) + patched * mask

    output = reconstructed.squeeze().permute(1,2,0).numpy()

    output = (output + 1) / 2
    output = (output * 255).clip(0,255).astype(np.uint8)

    return output


def calculate_metrics(original, restored):

    original = np.array(original.resize((256,256)))
    restored = np.array(Image.fromarray(restored).resize((256,256)))

    psnr = peak_signal_noise_ratio(original, restored, data_range=255)
    ssim = structural_similarity(original, restored, channel_axis=2)

    return psnr, ssim



def visualize_features_streamlit(image, mask, model):

    model.eval()

    transform = transforms.Compose([
        transforms.Resize((128,128)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
    ])

    img = transform(image)

    mask_np = (mask > 0).astype(np.float32)
    mask_np = 1 - mask_np

    mask_t = torch.tensor(mask_np).unsqueeze(0).unsqueeze(0).float()

    img = img.unsqueeze(0)

    patched = img * mask_t

    features = {}

    def save_feature(name):
        def hook(module, input, output):
            if isinstance(output, tuple):
                features[name] = output[0].detach()
            else:
                features[name] = output.detach()
        return hook

    hooks = []
    hooks.append(model.down1.register_forward_hook(save_feature("Encoder1")))
    hooks.append(model.down2.register_forward_hook(save_feature("Encoder2")))
    hooks.append(model.down3.register_forward_hook(save_feature("Encoder3")))
    hooks.append(model.bottleneck.register_forward_hook(save_feature("Bottleneck")))
    hooks.append(model.dec3.register_forward_hook(save_feature("Decoder3")))
    hooks.append(model.dec2.register_forward_hook(save_feature("Decoder2")))
    hooks.append(model.dec1.register_forward_hook(save_feature("Decoder1")))

    with torch.no_grad():
        prediction, _ = model(patched, mask_t)

    reconstructed = prediction * (1 - mask_t) + patched * mask_t

    for h in hooks:
        h.remove()

    return features, patched, prediction, reconstructed
