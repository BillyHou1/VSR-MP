# LiteAVSE
Low Latency and Lightweight Methods for Realistic Noise Suppression.
University of Bristol EEME30003 Major Project, 2025-26.

## Changelog
`2026-01-21`: Repository renamed to `LiteAVSE`. Focus shifted to lightweight, low-latency architectures for real-world deployment.  
`2026-01-13`: Project direction confirmed AVSE.  
`2026-01-10`: Repository initialized.  

## Overview
We study audio-visual speech enhancement using diffusion-based and flow-based generative models, and compare their performance under a unified experimental setup.

The core hypothesis is that lip movements, being immune to acoustic noise, can guide the enhancement process even in extreme low-SNR conditions. We aim to investigate which generative paradigm (diffusion vs. flow matching) offers a better trade-off between quality and efficiency.

We plan to explore two generative paradigms:
1. **Diffusion-based:** Audio-Visual Diffusion with U-Net (based on SGMSE+).
2. **Flow-based:** Audio-Visual Flow Matching, potentially adopting DiT backbone (inspired by FlowAVSE & VoiceDiT).

We also aim to develop a lightweight application prototype to demonstrate the final system.

```
Input:  Noisy Audio + Lip Video
                |
        Generative Model
       /                \
  Diffusion            Flow Matching
  (SGMSE+)          (FlowAVSE/VoiceDiT)
       \                /
                |
Output: Enhanced Audio
```

## Status
Current focus:
1. **Data:** Acquiring LRS3 dataset access from Oxford VGG.
2. **Dev:** Implementing the `Visual Front-end` shared module.
3. **Model:** Reproducing baselines under unified setup.
4. **App:** Planning lightweight demo prototype.

## Repository Structure
Code is organized by architecture:
- `models/diffusion/`: Diffusion-based model (SGMSE+ backbone).
- `models/flow/`: Flow Matching model (FlowAVSE/VoiceDiT backbone).
- `models/visual_frontend/`: Shared lip-reading encoder (ResNet/Conformer).
- `app/`: Lightweight application prototype (planned).

## Documentation
- Team & timeline: see [`docs/PLAN.md`](docs/PLAN.md)
- Experiments Log: see [`docs/LOG.md`](docs/LOG.md)
- Benchmark Results: see [`docs/RESULTS.md`](docs/RESULTS.md)

## Reproducibility
All experiments must have:
 A config file under `configs/` (e.g., `flow_lrs3.yaml`, `diff_lrs3.yaml`)

## References

### Core Papers

#### Diffusion-based
| Paper | arXiv | Role in Project |
|-------|-------|-----------------|
| [SGMSE+](https://arxiv.org/abs/2208.05830) | 2208.05830 | Diffusion backbone & training framework. |

#### Flow-based
| Paper | arXiv | Role in Project |
|-------|-------|-----------------|
| [FlowAVSE](https://arxiv.org/abs/2406.09286) | 2406.09286 | Flow Matching algorithm & loss function. |
| [VoiceDiT](https://arxiv.org/abs/2412.19259) | 2412.19259 | Dual-Condition Transformer (DiT) architecture. |

### Baseline References

| Paper | arXiv | Usage |
|-------|-------|-------|
| [VisualVoice](https://arxiv.org/abs/2101.03149) | 2101.03149 | Discriminative AVSE baseline. |

### Visual Front-end

| Paper | Venue | Usage |
|-------|-------|-------|
| [Auto-AVSR](https://arxiv.org/abs/2303.14307) | ICASSP 2023 | Lip-reading Encoder (ResNet+Conformer) |

## Acknowledgements
We acknowledge the open-source contributions from the authors of SGMSE, FlowAVSE, and VoiceDiT.

## License
Code will be released under an open-source license (TBD). Upstream components retain their original licenses.

## Contact
**Supervisor:** Dr. Fadi Karameh

For questions or access, please open an issue.
