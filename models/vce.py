# Billy
# VCE scores each video frame's reliability, outputs alpha in [0,1]. High means
# the face is clear and useful, low means blocked or blurry.

import torch
import torch.nn as nn

# TODO VCE and VCEWithTemporalSmoothing
# VCE is a small net that scores each video frame's reliability.
# Takes [B, T, 512] from the visual encoder, returns [B, T, 1] in [0,1].
# Each frame is independent so T acts like an extra batch dim.
# Architecture is up to you, just keep the I/O shape.
# VCEWithTemporalSmoothing adds causal smoothing on top of VCE
# so alpha doesn't jump around frame to frame.
# Causal means only current + past frames, no future. We need that for streaming.
# Smoothing method is your call.


class VCE(nn.Module):
    def __init__(self, in_dim=512):
        super().__init__()
        # TODO 512 -> scalar in [0,1], each frame independently
        raise NotImplementedError

    def forward(self, x):
        """x: [B, T, 512] -> [B, T, 1] in [0, 1]"""
        raise NotImplementedError


class VCEWithTemporalSmoothing(VCE):
    def __init__(self, in_dim=512, smooth_kernel=5):
        super().__init__(in_dim)
        # TODO causal smoothing, only look at current + past frames
        # make sure output stays in [0,1] after smoothing
        raise NotImplementedError

    def forward(self, x):
        """x: [B, T, 512] -> [B, T, 1] in [0, 1], temporally smoothed"""
        raise NotImplementedError
