# Copyright 2026 Konstantinos Palapanidis, Centre for Research and Technology Hellas (CERTH)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Demo of the Invariant-Based Fusion Module (IBFM).

Builds the IBFM in the configurations described in the paper, passes random non-negative Vis, IR and
Fused feature maps through it, checks the output shapes and the gradients, and writes a summary.

K. Palapanidis, T. Tsikrika, K. Ioannidis, S. Vrochidis and I. Kompatsiaris,
"Invariant-Based Channel Attention for Robust Visible-Infrared Fusion Under Sensor Degradation",
IEEE Access, vol. 14, 2026, doi: 10.1109/ACCESS.2026.3706896.

Usage:
    python demo.py

The summary is printed and saved to ../results/demo_output.txt if a 'results' folder exists next to
the code folder (as on Code Ocean), otherwise to ./results/demo_output.txt.
"""

import os
import time

import torch

from nn_fusion import FusionBlock, ScaledSigmoid

B, C, H, W = 2, 64, 80, 80

# Attention MLP and attention application settings of the paper
ATTENTION_PARAMS = {
    'reduction_ratio': 48,
    'attention_activation': ScaledSigmoid,
    'attention_activation_params': {'y_scale': 4.7, 'x_scale': 0.34, 'x_offset': 0.0, 'y_offset': -0.7},
}
RECALIBRATION_PARAMS = {'has_batch_norm': True, 'has_skip_connection': True, 'skip_con_param': 0.9}

INVARIANTS = {
    'Zernike (D=3)': ('zernike', {'D': 3}),
    'Hu (1-4)': ('hu', {'moments_to_compute': [1, 2, 3, 4]}),
    'Statistical (1-4)': ('statistical', {'moments_to_compute': [1, 2, 3, 4]}),
}

# Branch configuration: (channel arguments, number of input streams)
BRANCHES = {
    'Tri-Branch (Vis, IR, Fused)': ({'c_in_rgb': C, 'c_in_ir': C, 'c_in_fused': C}, 3),
    'Bi-Branch (Vis, IR)': ({'c_in_rgb': C, 'c_in_ir': C}, 2),
    'Uni-Branch': ({'c_in_rgb': C}, 1),
}


def results_dir() -> str:
    code_dir = os.path.dirname(os.path.abspath(__file__))
    code_ocean_results = os.path.join(os.path.dirname(code_dir), 'results')
    path = code_ocean_results if os.path.isdir(code_ocean_results) else os.path.join(code_dir, 'results')
    os.makedirs(path, exist_ok=True)
    return path


def run_config(branch_kwargs, n_inputs, invariant_type, invariant_params):
    block = FusionBlock(
        **branch_kwargs,
        aggregator_type='add_rgb_ir_add_fused',
        invariant_type=invariant_type,
        invariant_params=invariant_params,
        attention_type='MLP',
        attention_params=ATTENTION_PARAMS,
        recalibration_params=RECALIBRATION_PARAMS,
    )
    # Non-negative inputs, as in the network where the IBFM inputs come after the ReLU of the residual blocks
    inputs = [torch.relu(torch.randn(B, C, H, W)).requires_grad_() for _ in range(n_inputs)]

    start = time.perf_counter()
    out = block(*inputs)
    out.mean().backward()
    elapsed_ms = (time.perf_counter() - start) * 1000

    n_params = sum(p.numel() for p in block.parameters())
    finite = bool(torch.isfinite(out).all()) and all(
        p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in block.parameters())
    return tuple(out.shape), n_params, finite, elapsed_ms


def main():
    torch.manual_seed(0)
    lines = [
        'IBFM demo',
        f'PyTorch {torch.__version__}, input feature maps of shape ({B}, {C}, {H}, {W}), random non-negative values',
        'Aggregation: element-wise addition; attention: MLP (r = 48) with the adjusted sigmoid of Eq. (27)',
        '',
        f"{'Branches':30s} {'Invariants':20s} {'Output shape':20s} {'Params':>8s} {'Finite':>7s} {'Fwd+bwd ms':>11s}",
    ]
    all_ok = True
    for branch_name, (branch_kwargs, n_inputs) in BRANCHES.items():
        for inv_name, (inv_type, inv_params) in INVARIANTS.items():
            shape, n_params, finite, ms = run_config(branch_kwargs, n_inputs, inv_type, inv_params)
            all_ok = all_ok and finite and shape == (B, C, H, W)
            lines.append(f'{branch_name:30s} {inv_name:20s} {str(shape):20s} {n_params:8d} {str(finite):>7s} {ms:11.1f}')

    lines += ['', 'All checks passed.' if all_ok else 'Some checks FAILED.']
    report = '\n'.join(lines)
    print(report)

    out_path = os.path.join(results_dir(), 'demo_output.txt')
    with open(out_path, 'w') as f:
        f.write(report + '\n')
    print(f'\nSaved to {out_path}')

    if not all_ok:
        raise SystemExit(1)


if __name__ == '__main__':
    main()