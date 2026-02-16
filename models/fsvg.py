# Dominic part
# FSVG controls how much visual info gets injected at each frequency bin.
# The idea is lip movements help most with speech frequencies (300Hz-3kHz)
# but don't do much for high-freq noise, so the gate learns to be selective.
import torch
import torch.nn as nn

# TODO FSVG and FSVGWithPrior
# FSVG produces a gate [B, 1, T, F] in [0,1] from audio and visual features.
# It's a soft mask that controls which T-F bins get visual info.
# Input is audio_feat [B, C, T, F] and visual_feat [B, C, T, F].
# This works on the T-F plane so use Conv2d, not Conv1d.
# FSVGWithPrior inherits FSVG and adds a learnable frequency prior.
# Think about where to inject the prior so it interacts well with the gating.
# _logits() is split out from forward() so subclasses can override the gating logic.


class FSVG(nn.Module):
    def __init__(self, in_channels):
        """
        Args:
            in_channels: C, the channel dim of EACH of audio_feat and visual_feat.
                         The conv input is 2*C because we concatenate them along dim=1.
        """
        super().__init__()
        # TODO conv layers that produce 1-channel gate logits from audio and visual features
        raise NotImplementedError

    def _logits(self, audio_feat, visual_feat):
        """Compute raw gate logits before sigmoid. [B, 1, T, F]"""
        raise NotImplementedError

    def forward(self, audio_feat, visual_feat):
        """
        audio_feat:  [B, C, T, F]
        visual_feat: [B, C, T, F] (already projected and expanded to match)
        returns [B, 1, T, F] in [0, 1]
        """
        raise NotImplementedError


class FSVGWithPrior(FSVG):
    def __init__(self, in_channels, n_freq):
        super().__init__(in_channels)
        # TODO learnable frequency prior as nn.Parameter
        raise NotImplementedError

    def forward(self, audio_feat, visual_feat):
        """gate = sigmoid(logits + prior)"""
        raise NotImplementedError
