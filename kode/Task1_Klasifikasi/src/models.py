"""
Neural Network Architectures for Multimodal Defect Classification
Task (b) Architectures:
- Image Encoder: ResNet-like 2D-CNN optimized for 64x64 single-channel images.
- Signal Encoder: Multi-scale Residual 1D-CNN for 3-channel 128-timestep time series.
- Fusion Strategies:
  1. Early / Feature Concatenation (MultimodalConcat)
  2. Late Decision Fusion with Adaptive Gating (MultimodalLateFusion)
  3. Cross-Attention Fusion (MultimodalCrossAttention)
- Unimodal Baselines:
  - ImageOnlyModel
  - SignalOnlyModel
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------------------------------------------------------------
# 1. Image Encoder (ResNet2D for 64x64 single channel)
# -------------------------------------------------------------------------
class ConvBlock2D(nn.Module):
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU(inplace=True)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_c != out_c:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_c)
            )

    def forward(self, x):
        res = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.relu(out + res)
        return out

class ImageEncoder(nn.Module):
    def __init__(self, in_channels=1, feat_dim=128):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        # 64x64 -> 32x32 -> 16x16 -> 8x8
        self.stage1 = ConvBlock2D(32, 32, stride=1)
        self.stage2 = ConvBlock2D(32, 64, stride=2)   # 32x32
        self.stage3 = ConvBlock2D(64, 128, stride=2)  # 16x16
        self.stage4 = ConvBlock2D(128, 256, stride=2) # 8x8
        
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(256, feat_dim),
            nn.BatchNorm1d(feat_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, return_tokens=False):
        # x: (B, 1, 64, 64)
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        feat_map = self.stage4(x) # (B, 256, 8, 8)
        
        pooled = self.pool(feat_map).flatten(1) # (B, 256)
        out_feat = self.fc(pooled) # (B, feat_dim)
        
        if return_tokens:
            # Flatten spatial tokens: (B, 64, 256)
            B, C, H, W = feat_map.shape
            tokens = feat_map.view(B, C, H * W).permute(0, 2, 1)
            return out_feat, tokens
        return out_feat


# -------------------------------------------------------------------------
# 2. Signal Encoder (ResNet1D for 3-channel 128-timestep time series)
# -------------------------------------------------------------------------
class ConvBlock1D(nn.Module):
    def __init__(self, in_c, out_c, stride=1, kernel_size=5):
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(in_c, out_c, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm1d(out_c)
        self.conv2 = nn.Conv1d(out_c, out_c, kernel_size=kernel_size, stride=1, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm1d(out_c)
        self.relu = nn.ReLU(inplace=True)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_c != out_c:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_c, out_c, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_c)
            )

    def forward(self, x):
        res = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.relu(out + res)
        return out

class SignalEncoder(nn.Module):
    def __init__(self, in_channels=3, feat_dim=128):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=7, stride=1, padding=3, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True)
        )
        # 128 -> 64 -> 32 -> 16
        self.stage1 = ConvBlock1D(32, 32, stride=1, kernel_size=5)
        self.stage2 = ConvBlock1D(32, 64, stride=2, kernel_size=5)  # 64
        self.stage3 = ConvBlock1D(64, 128, stride=2, kernel_size=5) # 32
        self.stage4 = ConvBlock1D(128, 256, stride=2, kernel_size=3)# 16
        
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(256, feat_dim),
            nn.BatchNorm1d(feat_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, return_tokens=False):
        # x: (B, 3, 128)
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        feat_seq = self.stage4(x) # (B, 256, 16)
        
        pooled = self.pool(feat_seq).flatten(1) # (B, 256)
        out_feat = self.fc(pooled) # (B, feat_dim)
        
        if return_tokens:
            # (B, 16, 256)
            tokens = feat_seq.permute(0, 2, 1)
            return out_feat, tokens
        return out_feat


# -------------------------------------------------------------------------
# 3. Fusion Strategies
# -------------------------------------------------------------------------

class MultimodalConcat(nn.Module):
    """
    Strategy 1: Early / Feature Concatenation.
    Concatenates Image features, Sensor features, and the explicit Sensor Availability Mask.
    """
    def __init__(self, num_classes=7, feat_dim=128, dropout=0.3):
        super().__init__()
        self.img_encoder = ImageEncoder(in_channels=1, feat_dim=feat_dim)
        self.sig_encoder = SignalEncoder(in_channels=3, feat_dim=feat_dim)
        
        # Concat: feat_img (feat_dim) + feat_sig (feat_dim) + mask (1)
        fusion_dim = feat_dim * 2 + 1
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, feat_dim),
            nn.BatchNorm1d(feat_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(feat_dim, num_classes)
        )

    def forward(self, image, sensor, sensor_mask):
        f_img = self.img_encoder(image) # (B, feat_dim)
        f_sig = self.sig_encoder(sensor) # (B, feat_dim)
        
        # Zero out sensor features when sensor is missing / dropped
        f_sig = f_sig * sensor_mask
        
        fused = torch.cat([f_img, f_sig, sensor_mask], dim=1) # (B, 2*feat_dim + 1)
        logits = self.classifier(fused)
        return logits


class MultimodalLateFusion(nn.Module):
    """
    Strategy 2: Late Fusion with Adaptive Decision Gating.
    Separate prediction heads for each modality; combines logits adaptively.
    When sensor is missing (mask=0), automatically falls back to pure image logits.
    """
    def __init__(self, num_classes=7, feat_dim=128, dropout=0.3):
        super().__init__()
        self.img_encoder = ImageEncoder(in_channels=1, feat_dim=feat_dim)
        self.sig_encoder = SignalEncoder(in_channels=3, feat_dim=feat_dim)
        
        self.head_img = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feat_dim, num_classes)
        )
        self.head_sig = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feat_dim, num_classes)
        )
        
        # Learnable modality blending gate
        self.gate = nn.Sequential(
            nn.Linear(feat_dim * 2, 1),
            nn.Sigmoid()
        )

    def forward(self, image, sensor, sensor_mask):
        f_img = self.img_encoder(image) # (B, feat_dim)
        f_sig = self.sig_encoder(sensor) # (B, feat_dim)
        
        logits_img = self.head_img(f_img) # (B, 7)
        logits_sig = self.head_sig(f_sig) # (B, 7)
        
        # Compute dynamic gating weight for signal: alpha in [0, 1]
        concat_feat = torch.cat([f_img, f_sig], dim=1)
        alpha = self.gate(concat_feat) # (B, 1)
        
        # When sensor_mask is 0, alpha is forced to 0 (100% image)
        effective_alpha = alpha * sensor_mask
        
        fused_logits = (1.0 - 0.5 * effective_alpha) * logits_img + (0.5 * effective_alpha) * logits_sig
        return fused_logits


class MultimodalCrossAttention(nn.Module):
    """
    Strategy 3: Cross-Attention Fusion.
    Image spatial tokens query Sensor temporal tokens to correlate visual surface defects
    with acoustic/thermal patterns.
    """
    def __init__(self, num_classes=7, feat_dim=128, d_model=128, nhead=4, dropout=0.3):
        super().__init__()
        self.img_encoder = ImageEncoder(in_channels=1, feat_dim=feat_dim)
        self.sig_encoder = SignalEncoder(in_channels=3, feat_dim=feat_dim)
        
        # Token projection from 256 to d_model
        self.proj_img = nn.Linear(256, d_model)
        self.proj_sig = nn.Linear(256, d_model)
        
        self.cross_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, batch_first=True, dropout=dropout)
        self.norm = nn.LayerNorm(d_model)
        
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(feat_dim + d_model + 1, feat_dim),
            nn.BatchNorm1d(feat_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(feat_dim, num_classes)
        )

    def forward(self, image, sensor, sensor_mask):
        f_img, img_tokens = self.img_encoder(image, return_tokens=True)   # f_img: (B, 128), tokens: (B, 64, 256)
        f_sig, sig_tokens = self.sig_encoder(sensor, return_tokens=True) # f_sig: (B, 128), tokens: (B, 16, 256)
        
        q = self.proj_img(img_tokens) # (B, 64, d_model)
        k = v = self.proj_sig(sig_tokens) * sensor_mask.unsqueeze(-1) # (B, 16, d_model)
        
        attn_out, _ = self.cross_attn(query=q, key=k, value=v)
        attn_out = self.norm(attn_out + q) # (B, 64, d_model)
        
        # Pool attention output across spatial dimension
        attn_pooled = self.pool(attn_out.permute(0, 2, 1)).squeeze(-1) # (B, d_model)
        attn_pooled = attn_pooled * sensor_mask # zero if sensor missing
        
        fused = torch.cat([f_img, attn_pooled, sensor_mask], dim=1) # (B, feat_dim + d_model + 1)
        logits = self.classifier(fused)
        return logits


# -------------------------------------------------------------------------
# 4. Unimodal Models (for Task c: Modal Value Assessment)
# -------------------------------------------------------------------------

class ImageOnlyModel(nn.Module):
    def __init__(self, num_classes=7, feat_dim=128, dropout=0.3):
        super().__init__()
        self.encoder = ImageEncoder(in_channels=1, feat_dim=feat_dim)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feat_dim, num_classes)
        )

    def forward(self, image, sensor=None, sensor_mask=None):
        feat = self.encoder(image)
        return self.classifier(feat)


class SignalOnlyModel(nn.Module):
    def __init__(self, num_classes=7, feat_dim=128, dropout=0.3):
        super().__init__()
        self.encoder = SignalEncoder(in_channels=3, feat_dim=feat_dim)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feat_dim, num_classes)
        )

    def forward(self, image=None, sensor=None, sensor_mask=None):
        feat = self.encoder(sensor)
        return self.classifier(feat)


def get_model(model_name='concat', num_classes=7, feat_dim=128, dropout=0.3):
    """Factory function for model instantiation."""
    model_name = model_name.lower()
    if model_name in ['concat', 'early_fusion', 'feature_concat']:
        return MultimodalConcat(num_classes=num_classes, feat_dim=feat_dim, dropout=dropout)
    elif model_name in ['late_fusion', 'late']:
        return MultimodalLateFusion(num_classes=num_classes, feat_dim=feat_dim, dropout=dropout)
    elif model_name in ['cross_attention', 'cross_attn', 'attention']:
        return MultimodalCrossAttention(num_classes=num_classes, feat_dim=feat_dim, dropout=dropout)
    elif model_name in ['image_only', 'image']:
        return ImageOnlyModel(num_classes=num_classes, feat_dim=feat_dim, dropout=dropout)
    elif model_name in ['signal_only', 'sensor_only', 'signal', 'sensor']:
        return SignalOnlyModel(num_classes=num_classes, feat_dim=feat_dim, dropout=dropout)
    else:
        raise ValueError(f"Unknown model name: {model_name}")
