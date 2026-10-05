# IBFM: Invariant-Based Fusion Module

PyTorch implementation of the Invariant-Based Fusion Module (IBFM) from the paper:

**Invariant-Based Channel Attention for Robust Visible-Infrared Fusion Under Sensor Degradation**
K. Palapanidis, T. Tsikrika, K. Ioannidis, S. Vrochidis and I. Kompatsiaris
*IEEE Access*, vol. 14, 2026. [doi:10.1109/ACCESS.2026.3706896](https://doi.org/10.1109/ACCESS.2026.3706896)

The IBFM is a channel attention module for visible-infrared fusion. Instead of reducing each channel to a
scalar (e.g. with global average pooling), it describes the feature map of each channel with a vector of
moment invariants (statistical, Hu or Zernike) and computes the channel attention weights from them with an MLP.

This repository contains the fusion module (`nn_fusion.py`). The backbone networks and the training and
evaluation code are not included.

## Requirements

Python 3 and PyTorch (developed with PyTorch 2.5.1).

```
pip install -r requirements.txt
```

## Usage

The configuration of the best model of the paper (Tri-Branch, element-wise addition, Zernike moments of degree 3):

```python
import torch
from nn_fusion import FusionBlock, ScaledSigmoid

fusion = FusionBlock(
    c_in_rgb=64, c_in_ir=64, c_in_fused=64,
    aggregator_type='add_rgb_ir_add_fused',
    invariant_type='zernike',
    invariant_params={'D': 3},
    attention_type='MLP',
    attention_params={
        'reduction_ratio': 48,
        'attention_activation': ScaledSigmoid,
        'attention_activation_params': {'y_scale': 4.7, 'x_scale': 0.34, 'x_offset': 0.0, 'y_offset': -0.7},
    },
    recalibration_params={'has_batch_norm': True, 'has_skip_connection': True, 'skip_con_param': 0.9},
)

x_vis = torch.randn(2, 64, 80, 80)
x_ir = torch.randn(2, 64, 80, 80)
x_fused = torch.randn(2, 64, 80, 80)

out = fusion(x_vis, x_ir, x_fused)  # (2, 64, 80, 80)
```

The inputs are given in the order (Vis, IR, Fused). In the code, `rgb` refers to the Vis stream.
For the Bi-Branch, omit `c_in_fused` and call `fusion(x_vis, x_ir)`; for the Uni-Branch, give only
`c_in_rgb` and call `fusion(x)`.

Other invariants:

```python
invariant_type='statistical', invariant_params={'moments_to_compute': [1, 2, 3, 4]}
invariant_type='hu',          invariant_params={'moments_to_compute': [1, 2, 3, 4]}
```

If `invariant_type` and `attention_type` are omitted, the block performs only the aggregation
(attention-free baselines of the ablation study).

## Contents of `nn_fusion.py`

| Class | Description |
|---|---|
| `FusionBlock` | The IBFM: aggregation, invariant calculation, attention MLP, attention application |
| `ChannelAggregator` | Feature aggregation strategies (addition, subtraction, maximum, channel selection, concatenation) |
| `StatisticalMomentsModule` | Mean, variance and standardized moments, Eq. (3)-(5) |
| `HuMomentsModule` | Hu moment invariants, Eq. (9)-(19) |
| `ZernikeMomentsModule` | Zernike moments, Eq. (20)-(24) |
| `AttentionMLPModule` | Bottleneck MLP of Eq. (26) |
| `ScaledSigmoid` | Adjusted sigmoid of Eq. (27) |
| `ApplySelfAttentionPrune` | Attention application, batch norm and weighted skip connection, Eq. (28)-(29) |

The module also includes some options that were not used in the paper: top-k channel pruning, dropout,
a learnable skip connection weight and normalisation of the invariants. They are disabled by default.

## Erratum

In Eq. (29) of the paper, $X_{skip}$ should read $X_{agg}$ (as shown in Fig. 1b).

## Citation

```bibtex
@article{palapanidis2026ibfm,
  author  = {Palapanidis, Konstantinos and Tsikrika, Theodora and Ioannidis, Konstantinos and
             Vrochidis, Stefanos and Kompatsiaris, Ioannis},
  title   = {Invariant-Based Channel Attention for Robust Visible-Infrared Fusion Under Sensor Degradation},
  journal = {IEEE Access},
  volume  = {14},
  year    = {2026},
  doi     = {10.1109/ACCESS.2026.3706896}
}
```

## Acknowledgement

This work was supported by the Frugal and Robust AI for Defence Advanced Intelligence (FaRADAI) project,
funded by the European Commission under the European Defence Fund, Grant 101103386.
Views and opinions expressed are however those of the author(s) only and do not necessarily reflect those
of the European Union or the European Commission. Neither the European Union nor the granting authority
can be held responsible for them.

## License

Copyright 2026 Konstantinos Palapanidis, Centre for Research and Technology Hellas (CERTH).

Licensed under the Apache License 2.0, see [LICENSE](LICENSE).
