# Zhenning work
# What this file does:
# This is where the actual denoising happens. Mamba scans along time and
# frequency to clean up the audio spectrogram. Think of it like an RNN that
# goes through the spectrogram row by row (time) and column by column (freq).
# What you need to do:
# Fix the imports, they break on mamba-ssm 2.3.0 and nothing runs
# Let MambaBlock run forward-only, right now it always does forward+backward
# Add a helper so we can switch between Mamba1 and Mamba2 from config
# Add CausalTFMambaBlock, this is the main one, LiteAVSEMamba uses it
# Why CausalTFMambaBlock matters:
# Real-time speech enhancement can't look into the future. You can't wait for
# someone to finish talking before you start denoising. So the time dimension
# has to be causal (forward-only). But frequency is different 100Hz and 4kHz
# exist at the same instant, there's no "future" in frequency, so freq stays
# bidirectional. This is one of the key differences from the original SEMamba.

# Recommended order: do 1 first otherwise nothing runs, then 2 and 4, then 3 last.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.nn.parameter import Parameter
from functools import partial
from einops import rearrange

# TODO fix imports for mamba-ssm >= 2.0
# right now these work on 1.2.x (the bundled version in mamba-1_2_0_post1/)
# but on 2.0+ Block moved to mamba_ssm.modules.block and
# Mamba2 lives in mamba_ssm.modules.mamba2
# use try/except so it works on both versions
from mamba_ssm.modules.mamba_simple import Mamba, Block
from mamba_ssm.models.mixer_seq_simple import _init_weights
from mamba_ssm.ops.triton.layernorm import RMSNorm


# TODO _get_mixer_cls(cfg, layer_idx) helper
# reads cfg['model_cfg']['mamba_version'] and returns the right class
# so we can swap Mamba1/Mamba2 from YAML. Not urgent, Mamba1 works fine without this.


def create_block(
    d_model, cfg, layer_idx=0, rms_norm=True, fused_add_norm=False, residual_in_fp32=False,
    ):
    d_state = cfg['model_cfg']['d_state'] # 16
    d_conv = cfg['model_cfg']['d_conv'] # 4
    expand = cfg['model_cfg']['expand'] # 4
    norm_epsilon = cfg['model_cfg']['norm_epsilon'] # 0.00001

    #TODO swap this with _get_mixer_cls(cfg, layer_idx) once you have it
    mixer_cls = partial(Mamba, layer_idx=layer_idx, d_state=d_state, d_conv=d_conv, expand=expand)
    norm_cls = partial(
        nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon
    )
    #TODO on mamba-ssm 2.x you need mlp_cls=nn.Identity here
    # 1.2.x doesn't accept that arg so don't add it if you're on the bundled version
    block = Block(
            d_model,
            mixer_cls,
            norm_cls=norm_cls,
            fused_add_norm=fused_add_norm,
            residual_in_fp32=residual_in_fp32,
            )
    block.layer_idx = layer_idx
    return block


# TODO the bidirectional param is in __init__ but forward() ignores it.
# store it and use it: when False, skip the backward pass and return [B, T, C].
# when True, keep current behaviour and return [B, T, 2C].
class MambaBlock(nn.Module):
    def __init__(self, in_channels, cfg, bidirectional=True):
        super(MambaBlock, self).__init__()
        # TODO store self.bidirectional
        n_layer = 1
        self.forward_blocks  = nn.ModuleList( create_block(in_channels, cfg) for i in range(n_layer) )
        self.backward_blocks = nn.ModuleList( create_block(in_channels, cfg) for i in range(n_layer) )

        self.apply(
            partial(
                _init_weights,
                n_layer=n_layer,
            )
        )

    def forward(self, x):
        # TODO check self.bidirectional, skip backward pass if False
        x_forward, x_backward = x.clone(), torch.flip(x, [1])
        resi_forward, resi_backward = None, None

        # Forward pass: processes the sequence in its original order
        for layer in self.forward_blocks:
            x_forward, resi_forward = layer(x_forward, resi_forward)
        y_forward = (x_forward + resi_forward) if resi_forward is not None else x_forward

        # Backward pass: flips the sequence, runs it through backward_blocks, then flips output back
        # this gives the model extra context by looking at the sequence in reverse
        # the two outputs get concatenated so the final output is [B, T, 2C]
        for layer in self.backward_blocks:
            x_backward, resi_backward = layer(x_backward, resi_backward)
        y_backward = torch.flip((x_backward + resi_backward), [1]) if resi_backward is not None else torch.flip(x_backward, [1])

        return torch.cat([y_forward, y_backward], -1)


class TFMambaBlock(nn.Module):
    """
    Both time and freq are bidirectional here. This is the original non-causal version,
    it needs the full audio before it can process anything so it only works offline.
    We keep it for ablation comparison. LiteAVSEMamba uses CausalTFMambaBlock below.
    """
    def __init__(self, cfg):
        super(TFMambaBlock, self).__init__()
        self.cfg = cfg
        self.hid_feature = cfg['model_cfg']['hid_feature']

        self.time_mamba = MambaBlock(in_channels=self.hid_feature, cfg=cfg)
        self.freq_mamba = MambaBlock(in_channels=self.hid_feature, cfg=cfg)

        # both are 2C -> C cause both mamba blocks are bidirectional so they output 2C
        self.tlinear = nn.ConvTranspose1d(self.hid_feature * 2, self.hid_feature, 1, stride=1)
        self.flinear = nn.ConvTranspose1d(self.hid_feature * 2, self.hid_feature, 1, stride=1)

    def forward(self, x):
        b, c, t, f = x.size()

        x = x.permute(0, 3, 2, 1).contiguous().view(b*f, t, c)
        x = self.tlinear( self.time_mamba(x).permute(0,2,1) ).permute(0,2,1) + x
        x = x.view(b, f, t, c).permute(0, 2, 1, 3).contiguous().view(b*t, f, c)
        x = self.flinear( self.freq_mamba(x).permute(0,2,1) ).permute(0,2,1) + x
        x = x.view(b, t, f, c).permute(0, 3, 1, 2)
        return x


# TODO CausalTFMambaBlock
# Same idea as TFMambaBlock but time is causal (forward only), freq stays bidirectional.
# Think about what dimensions change when time_mamba is no longer bidirectional.
# in/out shape: [B, C, T, F]
class CausalTFMambaBlock(nn.Module):
    def __init__(self, cfg):
        super(CausalTFMambaBlock, self).__init__()
        self.cfg = cfg
        self.hid_feature = cfg['model_cfg']['hid_feature']
        # TODO time_mamba (causal) and freq_mamba (bidirectional)
        # then tlinear and flinear with matching input dims
        raise NotImplementedError

    def forward(self, x):
        # same reshape logic as TFMambaBlock.forward()
        raise NotImplementedError
