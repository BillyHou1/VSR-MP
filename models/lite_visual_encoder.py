# Zhenning
# Extracts visual features from video frames (lip region) for the audio model.
# Two versions: EncoderA uses a pretrained 2D backbone for transfer learning,
# EncoderB is a custom 3D CNN that trains from scratch.
# Both take video [B, 3, Tv, 96, 96] and output:
# visual_raw [B, 512, T_audio] — generator uses this for VCE and for fusion
import torch
import torch.nn as nn
import torch.nn.functional as F
# EncoderA needs: from torchvision import models

# TODO LiteVisualEncoderA and LiteVisualEncoderB
# Both take video [B, 3, Tv, 96, 96] and output [B, 512, T_audio].
# Output is always 512 channels. Use F.interpolate to match the audio frame count.
# EncoderA uses a pretrained 2D backbone to extract per-frame features,
# then some temporal modelling for lip dynamics, then project to 512.
# Most lightweight backbones don't output 512 so you'll need a projection.
# cfg['visual_cfg']['freeze_visual_encoder'] controls whether to freeze the backbone.
# EncoderB is a custom 3D CNN, trains from scratch.
# Downsample spatial dims, pool out H/W, project to 512.
# Try to keep it under 1M params.


class LiteVisualEncoderA(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # TODO pretrained backbone, pool spatial, temporal conv, project to 512
        # freeze backbone if cfg says so
        raise NotImplementedError

    def forward(self, video, T_audio):
        """
        Args:
            video:   [B, 3, Tv, 96, 96]
            T_audio: int, target temporal length (= number of STFT frames)
        Returns:
            [B, 512, T_audio]
        """
        raise NotImplementedError


class LiteVisualEncoderB(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # TODO 3D conv backbone, pool spatial, project to 512, keep it small
        raise NotImplementedError

    def forward(self, video, T_audio):
        """Same interface as EncoderA: [B, 3, Tv, 96, 96] -> [B, 512, T_audio]"""
        raise NotImplementedError
