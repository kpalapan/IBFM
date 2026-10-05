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
Invariant-Based Fusion Module (IBFM).

Implementation of the fusion module of the paper:
K. Palapanidis, T. Tsikrika, K. Ioannidis, S. Vrochidis and I. Kompatsiaris,
"Invariant-Based Channel Attention for Robust Visible-Infrared Fusion Under Sensor Degradation",
IEEE Access, vol. 14, 2026, doi: 10.1109/ACCESS.2026.3706896.

Naming: 'rgb' refers to the visible (Vis) stream, 'ir' to the infrared stream and 'fused' to the
fusion stream of the paper.
"""

import torch
import torch.nn as nn
import math
from typing import Dict, Any, Optional, Type, Union
import pydoc


##########################################################################################
# --------------------------------- Fusion Block -----------------------------------------
class FusionBlock(nn.Module):
    """
    The IBFM block. Performs the feature aggregation and, optionally, the invariant calculation,
    the channel attention generation and the attention application (Section III-C of the paper).

    If invariant_type and attention_type are both None, the block performs only the aggregation
    specified by aggregator_type and returns the result (used for the attention-free baselines of
    the ablation study). If aggregator_type is None it defaults to 'concat'.

    Otherwise the full pipeline is used:
    aggregation -> invariant calculation -> attention MLP -> attention application.

    Args:
        c_in_rgb (int): Number of rgb input channels.
        c_in_ir (Optional[int]): Number of ir input channels.
        c_in_fused (Optional[int]): Number of fused input channels.
        aggregator_type (Optional[str]): Type of aggregation (see ChannelAggregator). Defaults to 'concat'.
        invariant_type (Optional[str]): 'statistical', 'hu' or 'zernike'.
        invariant_params (Optional[Dict[str, Any]]): Parameters for the invariant module.
        attention_type (Optional[str]): Type of attention mechanism. Only 'MLP' is supported.
        attention_params (Optional[Dict[str, Any]]): Parameters for the attention module.
        recalibration_params (Optional[Dict[str, Any]]): Parameters for the attention application
            module (ApplySelfAttentionPrune), e.g. batch norm and skip connection.
        pruning_type (Optional[str]): None for no pruning (as in the paper), or 'top-k'.
            Pruning was not used in the paper.
    """

    def __init__(self,
                 c_in_rgb: int,
                 c_in_ir: Optional[int] = None,
                 c_in_fused: Optional[int] = None,
                 aggregator_type: Optional[str] = None,
                 invariant_type: Optional[str] = None,
                 invariant_params: Optional[Dict[str, Any]] = None,
                 attention_type: Optional[str] = None,
                 attention_params: Optional[Dict[str, Any]] = None,
                 recalibration_params: Optional[Dict[str, Any]] = None,
                 pruning_type: Optional[str] = None):
        super().__init__()

        # Default aggregation is concatenation
        self.aggregator_type = aggregator_type if aggregator_type is not None else 'concat'

        # The aggregator is always used
        self.aggregator_module = ChannelAggregator(
            c_in_rgb=c_in_rgb,
            c_in_ir=c_in_ir,
            c_in_fused=c_in_fused,
            aggregator_type=self.aggregator_type
        )
        C_aggregator_out = self.aggregator_module.C_aggregator_out

        # Without invariant and attention types the block only aggregates
        self.is_attention_pipeline_active = not (invariant_type is None and attention_type is None)

        if not self.is_attention_pipeline_active:
            if pruning_type is not None:
                raise ValueError("FusionBlock: pruning_type requires invariant_type and attention_type.")
            self.invariant_module = None
            self.attention_module = None
            self.attention_prune_module = None
            # Output channels equal the result of the aggregation
            self.C_FusionBlock_out = C_aggregator_out

        else:
            invariant_params = invariant_params or {}
            attention_params = attention_params or {}
            recalibration_params = recalibration_params or {}

            # Invariant module
            if invariant_type == 'statistical':
                self.invariant_module = StatisticalMomentsModule(**invariant_params)
            elif invariant_type == 'hu':
                self.invariant_module = HuMomentsModule(**invariant_params)
            elif invariant_type == 'zernike':
                self.invariant_module = ZernikeMomentsModule(**invariant_params)
            elif invariant_type is None:
                raise ValueError("FusionBlock: invariant_type is required for the attention pipeline.")
            else:
                raise ValueError(f"Unsupported invariant_type: {invariant_type}")

            m = self.invariant_module.m

            # Attention module
            self.attention_type = attention_type
            self.attention_params = attention_params
            if attention_type == 'MLP':
                self.attention_module = AttentionMLPModule(C=C_aggregator_out, m=m, **attention_params)
            elif attention_type is None:
                raise ValueError("FusionBlock: attention_type is required for the attention pipeline.")
            else:
                raise ValueError(f"Unsupported attention_type: {attention_type}")

            # Attention application (and optional pruning)
            if pruning_type is None:
                if recalibration_params.get('ch_prune_ratio') is not None:
                    raise ValueError("FusionBlock: ch_prune_ratio is set but pruning_type is None.")
            elif pruning_type != 'top-k':
                raise ValueError(f"Unsupported pruning_type: {pruning_type}")
            self.attention_prune_module = ApplySelfAttentionPrune(C_in=C_aggregator_out, **recalibration_params)

            # Output channels (they change only if pruning is applied)
            self.C_FusionBlock_out = self.attention_prune_module.C_attention_prune_out

    def forward(self, *inputs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: (rgb, ir [optional], fused [optional]), each of shape (B, C_i, H, W).

        Returns:
            Tensor of shape (B, C_FusionBlock_out, H, W).
        """
        num_inputs = len(inputs)
        rgb_in = inputs[0]
        ir_in = inputs[1] if num_inputs > 1 else None
        fused_in = inputs[2] if num_inputs > 2 else None

        x_agg = self.aggregator_module(rgb=rgb_in, ir=ir_in, fused=fused_in)

        # Aggregation only
        if not self.is_attention_pipeline_active:
            return x_agg

        x_inv = self.invariant_module(x_agg)  # Shape (B, C, m)
        x_attn = self.attention_module(x_inv)  # Shape (B, C)
        output = self.attention_prune_module(x_agg, x_attn)

        return output


##########################################################################################
# ------------------- Modules for Channel Aggregator -------------------------------------
class ElementwiseAdd(nn.Module):
    """
    Adds two tensors elementwise.
    Input tensors must have the same shape (B, C, H, W).
    """
    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        return x1 + x2


class ElementwiseSubtract(nn.Module):
    """
    Subtracts the second tensor from the first elementwise.
    Input tensors must have the same shape (B, C, H, W).
    """
    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        return x1 - x2


class ElementwiseMax(nn.Module):
    """
    Computes the elementwise maximum of two tensors.
    Input tensors must have the same shape (B, C, H, W). It returns a tensor of the same shape (B, C, H, W),
    where each position (b, c, h, w) contains the maximum of x1[b,c,h,w] and x2[b,c,h,w].
    """
    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        return torch.max(x1, x2)


class ChannelWiseSelection(nn.Module):
    """
    Compares two tensors channel-wise and keeps the channel with the higher mean activation.
    Input tensors must have the same shape (B, C, H, W).
    """
    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        # Spatial mean per channel, shape (B, C, 1, 1)
        map1_score = x1.mean(dim=(2, 3), keepdim=True)
        map2_score = x2.mean(dim=(2, 3), keepdim=True)

        # True where x1 is dominant
        mask = map1_score > map2_score

        # Broadcasts to (B, C, H, W)
        return torch.where(mask, x1, x2)


class ChannelConcat(nn.Module):
    """
    Concatenates input tensors along the channel dimension (dim=1).

    Accepts one or more tensors of shape (B, C_i, H, W) with the same H and W.
    Returns a single tensor of shape (B, sum(C_i), H, W).
    """
    def forward(self, *inputs: torch.Tensor) -> torch.Tensor:
        return torch.cat(inputs, dim=1)


# ------------------- Channel Aggregator -------------------------------------------------
# Takes tensors of shape (B, C_i, H, W) and applies the user specified aggregation method.
class ChannelAggregator(nn.Module):
    """
    Combines RGB, optionally IR, and optionally Fused feature tensors using various strategies.
    Convention for forward method: (rgb_features, ir_features [optional], fused_features [optional])

    Supported modes:
            - 'concat': Concatenates the input tensors along the channel dimension.
            - 'add_rgb_ir_add_fused': Elementwise addition of RGB, IR and Fused (used in the paper).
            - 'add_rgb_ir_subtract_fused': Elementwise addition of RGB and IR, then subtract Fused.
            - 'max_rgb_ir_fused': Elementwise maximum of RGB, IR and Fused.
            - 'max_rgb_ir_subtract_fused': Elementwise maximum of RGB and IR, then subtract Fused.
            - 'add_rgb_ir_concat_fused': Add RGB and IR, then concatenate the result with Fused.
            - 'subtract_rgb_ir_concat_fused': Subtract IR from RGB, then concatenate the result with Fused.
            - 'max_rgb_ir_concat_fused': Take max of RGB and IR, then concatenate the result with Fused.
            - 'select_rgb_ir_concat_fused': Compares RGB and IR channels,
                                          keeps the one with higher mean activation, then concatenates with Fused.
            - 'select_rgb_ir_fused': Compares RGB, IR and Fused channels, keeping the single
                                     channel (from the three) with the highest mean activation.
    If IR or Fused is None, the corresponding operation is skipped.
    """
    def __init__(self,
                 c_in_rgb: int,
                 c_in_ir: Optional[int] = None,
                 c_in_fused: Optional[int] = None,
                 aggregator_type: str = 'concat'):
        super().__init__()
        self.mode = aggregator_type

        # The basic operations
        self.add = ElementwiseAdd()
        self.subtract = ElementwiseSubtract()
        self.concat = ChannelConcat()
        self.max = ElementwiseMax()
        self.select = ChannelWiseSelection()

        # Number of output channels of the aggregator according to the available options.
        # If the options change, it should be changed accordingly.
        self.C_aggregator_out = c_in_rgb

        if self.mode == 'concat':
            if c_in_ir is not None:
                self.C_aggregator_out += c_in_ir
            if c_in_fused is not None:
                self.C_aggregator_out += c_in_fused
        elif self.mode in ['add_rgb_ir_add_fused', 'add_rgb_ir_subtract_fused', 'max_rgb_ir_fused',
                           'max_rgb_ir_subtract_fused', 'select_rgb_ir_fused']:
            # Assuming c_in_rgb == c_in_ir and c_in_rgb == c_in_fused
            self.C_aggregator_out = c_in_rgb
        elif self.mode in ['add_rgb_ir_concat_fused', 'subtract_rgb_ir_concat_fused', 'max_rgb_ir_concat_fused',
                           'select_rgb_ir_concat_fused']:
            # Assuming c_in_rgb == c_in_ir
            if c_in_fused is not None:
                self.C_aggregator_out += c_in_fused
        else:
            raise ValueError(f"Unsupported aggregator_type: {self.mode}")

    def forward(self, rgb: torch.Tensor,
                ir: Optional[torch.Tensor] = None,
                fused: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            rgb (torch.Tensor): Features from the RGB branch (B, C_rgb, H, W).
            ir (Optional[torch.Tensor]): Features from the IR branch (B, C_ir, H, W).
            fused (Optional[torch.Tensor]): Features from the fusion branch (B, C_fused, H, W).
        Returns:
            torch.Tensor: The aggregated feature tensor.
        """
        if self.mode == 'concat':
            # List of the tensors that are not None
            tensors_to_concat = [rgb]
            if ir is not None:
                tensors_to_concat.append(ir)
            if fused is not None:
                tensors_to_concat.append(fused)

            return self.concat(*tensors_to_concat)

        intermediate_result = None
        # Operation between rgb and ir
        if self.mode in ['add_rgb_ir_add_fused', 'add_rgb_ir_subtract_fused', 'add_rgb_ir_concat_fused']:
            intermediate_result = self.add(rgb, ir) if ir is not None else rgb

        elif self.mode in ['max_rgb_ir_fused', 'max_rgb_ir_subtract_fused', 'max_rgb_ir_concat_fused']:
            intermediate_result = self.max(rgb, ir) if ir is not None else rgb

        elif self.mode == 'subtract_rgb_ir_concat_fused':
            intermediate_result = self.subtract(rgb, ir) if ir is not None else rgb

        elif self.mode in ['select_rgb_ir_concat_fused', 'select_rgb_ir_fused']:
            intermediate_result = self.select(rgb, ir) if ir is not None else rgb

        # Operation with the fused tensor
        if self.mode == 'add_rgb_ir_add_fused':
            return self.add(intermediate_result, fused) if fused is not None else intermediate_result

        elif self.mode == 'max_rgb_ir_fused':
            return self.max(intermediate_result, fused) if fused is not None else intermediate_result

        elif self.mode == 'select_rgb_ir_fused':
            return self.select(intermediate_result, fused) if fused is not None else intermediate_result

        elif self.mode in ['add_rgb_ir_subtract_fused', 'max_rgb_ir_subtract_fused']:
            return self.subtract(intermediate_result, fused) if fused is not None else intermediate_result

        elif self.mode in ['add_rgb_ir_concat_fused', 'subtract_rgb_ir_concat_fused', 'max_rgb_ir_concat_fused',
                           'select_rgb_ir_concat_fused']:
            return self.concat(intermediate_result, fused) if fused is not None else intermediate_result

        else:
            raise ValueError(f"Unknown or unsupported mode: {self.mode}")


##########################################################################################
##########################################################################################
# -------------------------- Attention MLP -----------------------------------------------


# ---------- Custom activations ------------------------------------
class ScaledSigmoid(nn.Module):
    """
    Adjusted sigmoid of eq. (27): y_scale * sigmoid(x_scale * x + x_offset) + y_offset,
    i.e. a = y_scale, b = x_scale, c = x_offset, d = y_offset.
    The paper uses a = 4.7, b = 0.34, c = 0, d = -0.7.
    """
    def __init__(self, x_scale=1.0, x_offset=0.0, y_scale=1.0, y_offset=0.0):
        super().__init__()
        self.x_scale = x_scale
        self.x_offset = x_offset
        self.y_scale = y_scale
        self.y_offset = y_offset

    def extra_repr(self):
        return f'x_scale={self.x_scale}, x_offset={self.x_offset}, y_scale={self.y_scale}, y_offset={self.y_offset}'

    def forward(self, x):
        return self.y_scale * torch.sigmoid(x * self.x_scale + self.x_offset) + self.y_offset


# ---------------------------------

# Input Tensor of shape (B, C, m), m: number of invariants. Output Tensor of shape (B, C)
# This module uses an MLP to calculate the attention scores from the invariants, eq. (26).
class AttentionMLPModule(nn.Module):
    """
    Calculates channel attention weights using an MLP.

    Takes an input tensor (from the invariant calculation) of shape (B, C, m).
    If the input is complex (Zernike moments), the magnitude is used.
    The channel and moment dimensions are flattened and passed through an MLP
    with a hidden bottleneck layer, giving attention weights of shape (B, C).

    Args:
        C (int): Number of input channels.
        m (int): Number of invariants per channel.
        reduction_ratio (int): Reduction ratio r of the hidden layer.
                               Hidden layer size = max(1, (C * m) // reduction_ratio).
                               Defaults to 4 (the paper uses 48).
        has_bias (bool): Whether to include bias in the linear layers. Defaults to False.
        normalise_invariants (bool): Whether to normalise the invariants before the MLP.
                                     Not used in the paper. Defaults to False.
        normalisation_module (str): Normalisation type if normalise_invariants is True.
                                    Options: 'BatchNorm1d', 'LayerNorm'. Defaults to None.
        attention_activation (str or Type[nn.Module]): The class of the activation function
                                    (e.g. nn.Sigmoid, ScaledSigmoid), or its name.
        attention_activation_params (Optional[Dict[str, Any]]): Parameters passed to the
                                    constructor of the activation function.
    """
    def __init__(self, C: int, m: int, reduction_ratio: int = 4, has_bias: bool = False,
                 normalise_invariants: bool = False, normalisation_module: Optional[str] = None,
                 attention_activation: Union[str, Type[nn.Module]] = nn.Sigmoid,
                 attention_activation_params: Optional[Dict[str, Any]] = None):
        super().__init__()

        self.C = C  # Channels
        self.m = m  # Invariants
        self.normalise_invariants = normalise_invariants
        self.normalisation_module = normalisation_module

        # MLP input dimension is always C * m (magnitude is used for complex input)
        mlp_input_dim = C * m
        hidden_dim = max(1, mlp_input_dim // reduction_ratio)

        # Optional normalisation of the invariants
        if not self.normalise_invariants:
            self.inv_norm = nn.Identity()
        else:
            if self.normalisation_module == "BatchNorm1d":
                self.inv_norm = nn.BatchNorm1d(num_features=mlp_input_dim)
            elif self.normalisation_module == "LayerNorm":
                self.inv_norm = nn.LayerNorm(mlp_input_dim)
            else:
                raise ValueError(f"Error in normalisation_module value: '{self.normalisation_module}'")

        if isinstance(attention_activation, str):
            # If it's a short name, the class is looked up in this module
            if '.' not in attention_activation:
                full_path = f"{__name__}.{attention_activation}"
            # Else the full path to the class has been given (e.g. 'torch.nn.Sigmoid')
            else:
                full_path = attention_activation
            activation_class = pydoc.locate(full_path)
        # The activation_class is already a class (i.e. the default value of the Sigmoid)
        else:
            activation_class = attention_activation

        params = attention_activation_params or {}
        # MLP layers
        self.mlp = nn.Sequential(
            nn.Linear(mlp_input_dim, hidden_dim, bias=has_bias),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, C, bias=has_bias),
            activation_class(**params)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, C, m). Can be real or complex.

        Returns:
            Tensor of attention weights of shape (B, C).
        """
        B, C_in, m_in = x.shape

        # Magnitude for complex input (Zernike moments)
        if x.is_complex():
            x_proc = torch.abs(x)  # Shape (B, C, m), real
        else:
            x_proc = x  # Shape (B, C, m)

        # Flatten C and m dimensions -> (B, C * m)
        x_flat = x_proc.view(B, self.C*self.m)
        # LayerNorm or BatchNorm1d (Identity if no normalisation)
        x_flat = self.inv_norm(x_flat)
        attn_weights = self.mlp(x_flat)  # Shape (B, C)

        return attn_weights


##########################################################################################
##########################################################################################
# ------------------------ Apply channel attention ---------------------------------------

# Requires two tensors as input. Aggregated tensor (B, C, H, W), and attention weights (B, C).
# Returns a tensor of shape (B, C, H, W), eq. (28)-(29). With top-k pruning (not used in the paper)
# the shape is (B, k, H, W), with k = ceil(ch_prune_ratio * C).
class ApplySelfAttentionPrune(nn.Module):
    """
    Applies channel attention weights to the aggregated features, followed by optional batch norm
    and a weighted skip connection. Optionally prunes the channels, keeping the 'k' channels with
    the highest attention scores.

    Configuration used in the paper: has_batch_norm=True, has_skip_connection=True,
    has_learn_par_skip_con=False, skip_con_param=0.9 (alpha in eq. (29)), no pruning and no dropout.

    Args:
        C_in (int): Number of input channels.
        ch_prune_ratio (float): Ratio of channels to keep (0, 1), based on the highest attention scores.
                 If None, no pruning is applied. Not used in the paper. Defaults to None.
        has_batch_norm (bool): If true applies a batch norm 2d layer at the output.
        has_skip_connection (bool): If true applies a skip connection (after the batch norm layer).
        has_learn_par_skip_con (bool): If true the skip connection weight is a learnable parameter.
                 Not used in the paper.
        skip_con_param (float): Weight of the skip connection (or initial value, if learnable).
        dropout_type (str): "channel" to apply Dropout2d, or "element" to apply Dropout.
                 Not used in the paper.
        dropout_p (float): Probability of a channel or element to be dropped out.
    """
    def __init__(self, C_in: int,
                 ch_prune_ratio: Optional[float] = None,
                 has_batch_norm: bool = False,
                 has_skip_connection: bool = False,
                 has_learn_par_skip_con: bool = False,
                 skip_con_param: float = 0.15,
                 dropout_type: Optional[str] = None,
                 dropout_p: Optional[float] = None):
        super().__init__()

        self.ch_prune_ratio = ch_prune_ratio
        self.has_batch_norm = has_batch_norm
        self.has_skip_connection = has_skip_connection
        self.has_learn_par_skip_con = has_learn_par_skip_con
        self.dropout_type = dropout_type
        self.dropout_p = dropout_p
        self.C_in = C_in

        self.C_attention_prune_out = C_in

        # Number of channels kept, if pruning is applied
        if self.ch_prune_ratio is not None and 0.0 < self.ch_prune_ratio < 1.0:
            k_actual = max(1, math.ceil(self.ch_prune_ratio * self.C_in))
            k_actual = min(self.C_in, k_actual)
            self.C_attention_prune_out = k_actual

        if self.has_batch_norm:
            self.bn = nn.BatchNorm2d(self.C_attention_prune_out)

        p_value = 0.0 if dropout_p is None else dropout_p

        if dropout_type == 'channel':
            self.dropout = nn.Dropout2d(p=p_value)
        elif dropout_type == 'element':
            self.dropout = nn.Dropout(p=p_value)
        elif dropout_type is None:
            self.dropout = nn.Identity()
        else:
            raise ValueError(f"Invalid dropout_type: '{dropout_type}'.")

        # Skip connection with learnable parameter
        if self.has_skip_connection and self.has_learn_par_skip_con:
            self.skip_con_param = nn.Parameter(torch.tensor(skip_con_param, dtype=torch.float32))
        # Skip connection with constant parameter
        elif self.has_skip_connection and not self.has_learn_par_skip_con:
            self.skip_con_param = skip_con_param
        # No skip connection
        else:
            self.has_learn_par_skip_con = False

    def forward(self, x_agg: torch.Tensor, x_attn: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_agg: The aggregated feature tensor of shape (B, C, H, W).
            x_attn: The attention weights of shape (B, C).
        Returns:
            The recalibrated feature tensor of shape (B, C, H, W),
            or (B, k, H, W) with 0 < k <= C if pruning is applied.
        """
        B, C, H, W = x_agg.shape
        # Attention weights (B, C) -> (B, C, 1, 1) for broadcasting
        x_attn_reshaped = x_attn.unsqueeze(-1).unsqueeze(-1)
        # Channel-wise multiplication, eq. (28)
        x_scaled = x_agg * x_attn_reshaped  # Shape (B, C, H, W)

        processed_features = x_scaled
        skip_input_features = x_agg  # x_agg is kept for the skip connection

        # Pruning (if ch_prune_ratio is specified and valid)
        if self.ch_prune_ratio is not None and 0.0 < self.ch_prune_ratio < 1.0:
            # Indices of the top k channels for each batch element.
            # Indices shape: (B, self.C_attention_prune_out)
            topk_indices = torch.topk(x_attn, k=self.C_attention_prune_out, dim=1).indices

            # Indices for gather: (B, k) -> (B, k, 1, 1) -> (B, k, H, W)
            index_for_gather = topk_indices.unsqueeze(-1).unsqueeze(-1).expand(B, self.C_attention_prune_out, H, W)

            # Output shape: (B, self.C_attention_prune_out, H, W)
            processed_features = torch.gather(x_scaled, dim=1, index=index_for_gather)
            # If pruning, the skip connection also needs to be pruned
            if self.has_skip_connection:
                skip_input_features = torch.gather(x_agg, dim=1, index=index_for_gather)

        if self.has_batch_norm:
            processed_features = self.bn(processed_features)

        # Identity if no dropout is specified
        processed_features = self.dropout(processed_features)

        if self.has_skip_connection:
            # eq. (29), self.skip_con_param is either nn.Parameter or a float
            processed_features = (self.skip_con_param * processed_features +
                                  (1.0 - self.skip_con_param) * skip_input_features)

        return processed_features


##########################################################################################
##########################################################################################
# ------------------------ Statistical Moment invariants ---------------------------------
class StatisticalMomentsModule(nn.Module):
    """
    Calculates the specified moments for each channel across the spatial dimensions, eq. (3)-(5).

    Moments are requested by order k >= 1:
    - k=1: Mean (μ)
    - k=2: Variance (σ²)
    - k>=3: k-th Standardized Moment (E[z^k], where z=(X-μ)/σ)

    Args:
        moments_to_compute (list or tuple): Integers k (>=1) specifying which moments to compute.
            The order in the list determines the output order. Duplicates are allowed and
            produce repeated outputs.
        eps (float): Small value added to the variance before the sqrt, eq. (5), for numerical
                     stability. Defaults to 1e-9.
    """

    def __init__(self, moments_to_compute=(1, 2), eps=1e-9):
        super().__init__()

        if not moments_to_compute:
            raise ValueError("moments_to_compute cannot be empty.")
        if not all(isinstance(k, int) and k >= 1 for k in moments_to_compute):
            raise ValueError("All elements in moments_to_compute must be integers >= 1.")

        self.requested_orders = list(moments_to_compute)
        self.m = len(self.requested_orders)
        self.eps = eps

        # Map from order k to its index (or indices) in the output tensor
        self.output_map = {order: [] for order in set(self.requested_orders)}
        for i, order in enumerate(self.requested_orders):
            self.output_map[order].append(i)

        unique_orders = set(self.requested_orders)
        self._needs_var_or_std = any(k >= 2 for k in unique_orders)
        self.standardized_orders_needed = {k for k in unique_orders if k >= 3}
        self.max_standardized_order = max(self.standardized_orders_needed) if self.standardized_orders_needed else 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, C, H, W).

        Returns:
            Tensor of moments of shape (B, C, m), ordered according to moments_to_compute.
        """
        B, C, H, W = x.shape
        x = x.float()

        output = torch.zeros(B, C, self.m, device=x.device, dtype=x.dtype)

        # For a 1x1 feature map only the mean is non-zero
        if H == 1 and W == 1:
            mu = x.squeeze(-1).squeeze(-1)  # Shape (B, C)
            if 1 in self.output_map:
                for idx in self.output_map[1]:
                    output[:, :, idx] = mu
            # Variance and higher moments stay 0
            return output

        # H, W > 1
        mu = torch.mean(x, dim=(2, 3), keepdim=False)
        if 1 in self.output_map:
            for idx in self.output_map[1]:
                output[:, :, idx] = mu

        if self._needs_var_or_std:
            mu_b = mu.unsqueeze(-1).unsqueeze(-1)
            variance = torch.mean((x - mu_b) ** 2, dim=(2, 3), keepdim=False)
            if 2 in self.output_map:
                for idx in self.output_map[2]:
                    output[:, :, idx] = variance

            if self.max_standardized_order >= 3:
                sigma = torch.sqrt(variance + self.eps)
                sigma_b = sigma.unsqueeze(-1).unsqueeze(-1)
                z = (x - mu_b) / sigma_b

                if 3 in self.standardized_orders_needed:
                    moment_3 = torch.mean(z.pow(3), dim=(2, 3), keepdim=False)
                    for idx in self.output_map[3]:
                        output[:, :, idx] = moment_3

                if self.max_standardized_order >= 4:
                    # Higher powers of z are computed incrementally
                    current_z_power = z.pow(3)
                    for k in range(4, self.max_standardized_order + 1):
                        current_z_power = current_z_power * z
                        if k in self.standardized_orders_needed:
                            moment_k = torch.mean(current_z_power, dim=(2, 3), keepdim=False)
                            for idx in self.output_map[k]:
                                output[:, :, idx] = moment_k
        return output


##########################################################################################
##########################################################################################
# ------------------------------ Hu Moment invariants ------------------------------------

def signed_pow(x: torch.Tensor, p: float) -> torch.Tensor:
    """
    x^p for x >= 0 and -|x|^p for x < 0, so that a fractional power is defined also for negative x.
    The branch that is not used gets 1 as base, to avoid NaN in the gradients.
    """
    pos = x >= 0
    return torch.where(pos,
                       torch.where(pos, x, torch.ones_like(x)) ** p,
                       -torch.where(pos, torch.ones_like(x), -x) ** p)


# Input Tensor of shape (B, C, H, W). Output Tensor of shape (B, C, m), 1 <= m: number of invariants <= 7
class HuMomentsModule(nn.Module):
    """
    Calculates the selected Hu moment invariants, eq. (9)-(19).
    Takes a (B, C, H, W) tensor as input.
    Returns the selected moments (raw or log-transformed) as a (B, C, m) tensor.
    """
    def __init__(self, moments_to_compute=(1, 2, 3, 4, 5, 6, 7), use_log_transform=True, eps=1e-10):
        """
        Args:
            moments_to_compute (list): 1-based indices (1 to 7) of the Hu moments to calculate and return.
            use_log_transform (bool): If True, returns the signed log-transformed moments of eq. (19).
                                      Otherwise, returns the raw moments.
            eps (float): Small value used to keep m00 away from zero and inside the log, for numerical stability.
        """
        super().__init__()
        moment_indices = list(moments_to_compute)
        if not all(1 <= i <= 7 for i in moment_indices):
            raise ValueError("moments_to_compute must be between 1 and 7.")

        self.EPSILON = eps
        # 0-based indices for internal use
        self.zero_based_indices = sorted([i - 1 for i in moment_indices])
        self.use_log_transform = use_log_transform
        self.num_selected_moments = len(self.zero_based_indices)
        self.m = self.num_selected_moments

    def forward(self, feature_maps):
        """
        Args:
            feature_maps (torch.Tensor): Input tensor of shape (B, C, H, W).

        Returns:
            torch.Tensor: Selected Hu moments, shape (B, C, m), where m is
                          the number of indices in moments_to_compute.
        """

        B, C, H, W = feature_maps.shape
        device = feature_maps.device

        y_coords, x_coords = torch.meshgrid(torch.arange(H, device=device),
                                            torch.arange(W, device=device),
                                            indexing='ij')
        x_coords = x_coords.float().unsqueeze(0).unsqueeze(0)
        y_coords = y_coords.float().unsqueeze(0).unsqueeze(0)
        feature_maps = feature_maps.float()

        # Raw moments, eq. (9). Only the ones needed for the centroid are calculated.
        m00 = torch.sum(feature_maps, dim=(-1, -2), keepdim=True)
        # Feature maps can also be negative, so m00 is moved away from zero keeping its sign
        m00 = torch.where(m00 >= 0, m00 + self.EPSILON, m00 - self.EPSILON)
        m10 = torch.sum(x_coords * feature_maps, dim=(-1, -2), keepdim=True)
        m01 = torch.sum(y_coords * feature_maps, dim=(-1, -2), keepdim=True)

        x_bar = m10 / m00
        y_bar = m01 / m00

        # Central moments, eq. (10)
        x_rel = x_coords - x_bar
        y_rel = y_coords - y_bar
        mu00 = m00
        mu20 = torch.sum(x_rel**2 * feature_maps, dim=(-1, -2), keepdim=True)
        mu02 = torch.sum(y_rel**2 * feature_maps, dim=(-1, -2), keepdim=True)
        mu11 = torch.sum(x_rel * y_rel * feature_maps, dim=(-1, -2), keepdim=True)
        mu30 = torch.sum(x_rel**3 * feature_maps, dim=(-1, -2), keepdim=True)
        mu03 = torch.sum(y_rel**3 * feature_maps, dim=(-1, -2), keepdim=True)
        mu12 = torch.sum(x_rel * y_rel**2 * feature_maps, dim=(-1, -2), keepdim=True)
        mu21 = torch.sum(x_rel**2 * y_rel * feature_maps, dim=(-1, -2), keepdim=True)

        # Normalized central moments, eq. (11)
        eta20 = mu20 / (mu00 ** 2.0)
        eta02 = mu02 / (mu00 ** 2.0)
        eta11 = mu11 / (mu00 ** 2.0)
        eta30 = mu30 / signed_pow(mu00, 2.5)
        eta03 = mu03 / signed_pow(mu00, 2.5)
        eta12 = mu12 / signed_pow(mu00, 2.5)
        eta21 = mu21 / signed_pow(mu00, 2.5)

        eta20 = eta20.squeeze(-1).squeeze(-1)
        eta02 = eta02.squeeze(-1).squeeze(-1)
        eta11 = eta11.squeeze(-1).squeeze(-1)
        eta30 = eta30.squeeze(-1).squeeze(-1)
        eta03 = eta03.squeeze(-1).squeeze(-1)
        eta12 = eta12.squeeze(-1).squeeze(-1)
        eta21 = eta21.squeeze(-1).squeeze(-1)

        # Hu invariants, eq. (12)-(18)
        h1 = eta20 + eta02
        h2 = (eta20 - eta02)**2 + 4 * eta11**2
        h3 = (eta30 - 3 * eta12)**2 + (3 * eta21 - eta03)**2
        h4 = (eta30 + eta12)**2 + (eta21 + eta03)**2
        h5 = (eta30 - 3 * eta12) * (eta30 + eta12) * \
             (((eta30 + eta12)**2 - 3 * (eta21 + eta03)**2)) + \
             (3 * eta21 - eta03) * (eta21 + eta03) * \
             ((3 * (eta30 + eta12)**2 - (eta21 + eta03)**2))
        h6 = (eta20 - eta02) * ((eta30 + eta12)**2 - (eta21 + eta03)**2) + \
             4 * eta11 * (eta30 + eta12) * (eta21 + eta03)
        h7 = (3 * eta21 - eta03) * (eta30 + eta12) * \
             (((eta30 + eta12)**2 - 3 * (eta21 + eta03)**2)) - \
             (eta30 - 3 * eta12) * (eta21 + eta03) * \
             ((3 * (eta30 + eta12)**2 - (eta21 + eta03)**2))

        hu_moments_raw = torch.stack(tensors=[h1, h2, h3, h4, h5, h6, h7], dim=-1)  # Shape (B, C, 7)

        # Raw or log-transformed, eq. (19)
        if self.use_log_transform:
            hu_moments_log = torch.sign(hu_moments_raw) * torch.log10(torch.abs(hu_moments_raw) + self.EPSILON)
            chosen_moments = hu_moments_log
        else:
            chosen_moments = hu_moments_raw

        # Selection of the requested moments along the last dimension
        selected_moments = chosen_moments[:, :, self.zero_based_indices]  # Shape (B, C, m), m<=7

        return selected_moments


##########################################################################################
##########################################################################################
# ------------------- Zernike polynomials ------------------------------------------------
def zernike_poly_vectorized_pt(Y: torch.Tensor, X: torch.Tensor,
                               n: int, l: int) -> torch.Tensor:
    """
    Zernike polynomial V_nl of degree n and repetition l, eq. (21)-(22), evaluated at the
    normalized coordinates (Y, X). In the paper the repetition is denoted by m.
    Returns a complex tensor with the same shape as X.
    """
    if not isinstance(X, torch.Tensor):
        X = torch.tensor(X)
    if not isinstance(Y, torch.Tensor):
        Y = torch.tensor(Y)
    X = X.float()
    Y = Y.float()

    rho = torch.hypot(X, Y)
    phi = torch.atan2(Y, X)

    if n == 0 and l == 0:
        return torch.ones_like(rho, dtype=torch.complex64)

    abs_l = abs(l)
    radial_poly = torch.zeros_like(rho, dtype=torch.float32)

    # Radial polynomial, eq. (22)
    if (n - abs_l) % 2 == 0:
        for s in range((n - abs_l) // 2 + 1):
            coeff = ((-1.)**s * math.factorial(n - s)) / (
                        math.factorial(s) *
                        math.factorial((n + abs_l) // 2 - s) *
                        math.factorial((n - abs_l) // 2 - s))
            radial_poly += coeff * (rho**(n - 2 * s))

    angular_part_real = torch.cos(l * phi)
    angular_part_imag = torch.sin(l * phi)
    angular_part = torch.complex(angular_part_real, angular_part_imag)
    vxy = torch.complex(radial_poly, torch.zeros_like(radial_poly)) * angular_part
    return vxy


class ZernikeMomentsModule(nn.Module):
    """
    Calculates the Zernike moments of each channel up to degree D, eq. (23)-(24).
    The basis is computed at the first forward pass for the spatial size of the input,
    and recomputed if the spatial size changes.

    Args:
        D (int): Maximum Zernike polynomial degree (the paper uses D = 3).
    """

    def __init__(self, D: int):
        super().__init__()
        self.D = D
        self.m = (D + 1) * (D + 2) // 2
        self.nl_pairs = None  # (n, l) pairs, filled in _initialize

        # The buffers are registered here (empty) and filled in _initialize, so that they always
        # exist in the module and are not created lazily.
        self._initialized = False
        self._basis_hw = None  # (H, W) the basis was computed for
        self.register_buffer('_H', torch.tensor(0, dtype=torch.long), persistent=False)
        self.register_buffer('_W', torch.tensor(0, dtype=torch.long), persistent=False)
        self.register_buffer('zernike_basis_conj_flat', torch.empty(0, dtype=torch.complex64), persistent=False)
        self.register_buffer('norm_factors', torch.empty(0, dtype=torch.float32), persistent=False)

    # ----------------------------------------------------------------
    @property
    def H(self):
        return self._H.item() if self._H.item() != 0 else None

    @property
    def W(self):
        return self._W.item() if self._W.item() != 0 else None

    # ----------------------------------------------------------------
    @torch.no_grad()
    def _initialize(self, H: int, W: int, device: torch.device):
        """Computes the Zernike basis and stores it in the registered buffers."""
        self._H.data = torch.tensor(H, dtype=torch.long, device=device)
        self._W.data = torch.tensor(W, dtype=torch.long, device=device)

        # The unit circle circumscribes the feature map, eq. (23)
        cofy = (H - 1) / 2.0
        cofx = (W - 1) / 2.0
        corners_y = torch.tensor([0, 0, H - 1, H - 1],
                                 dtype=torch.float32, device=device)
        corners_x = torch.tensor([0, W - 1, 0, W - 1],
                                 dtype=torch.float32, device=device)
        distances_sq = (corners_x - cofx) ** 2 + (corners_y - cofy) ** 2
        radius_circum = torch.sqrt(torch.max(distances_sq))
        radius_circum = torch.clamp(radius_circum, min=1e-7)

        y_indices = torch.arange(H, dtype=torch.float32, device=device)
        x_indices = torch.arange(W, dtype=torch.float32, device=device)
        Y_img, X_img = torch.meshgrid(y_indices, x_indices, indexing='ij')
        Yn_all = (Y_img - cofy) / radius_circum
        Xn_all = (X_img - cofx) / radius_circum
        Yn_flat = Yn_all.flatten()
        Xn_flat = Xn_all.flatten()

        moments_list = []
        norm_factors = []
        self.nl_pairs = []
        npix = float(H * W)

        for n in range(self.D + 1):
            for l_val in range(-n, n + 1, 2):
                vxy = zernike_poly_vectorized_pt(Yn_flat, Xn_flat, n, l_val)
                moments_list.append(torch.conj(vxy))
                norm_factors.append((n + 1.0) / npix)
                self.nl_pairs.append((n, l_val))

        # Assignment to the already registered buffers (register_buffer is not called again here,
        # as it does not work inside torch.compile)
        self.zernike_basis_conj_flat = torch.stack(moments_list, dim=0)
        self.norm_factors = torch.tensor(
            norm_factors, dtype=torch.float32, device=device)

        self._initialized = True
        self._basis_hw = (H, W)

    # ----------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, C, H, W).

        Returns:
            Complex tensor of Zernike moments of shape (B, C, m).
        """
        B, C, H, W = x.shape
        device = x.device

        # The basis is (re)computed at the first pass and whenever the spatial size changes
        if not self._initialized or getattr(self, '_basis_hw', None) != (H, W):
            self._initialize(H, W, device)

        x_flat = x.view(B, C, H * W).to(torch.complex64)

        # Matrix multiplication with the basis, eq. (24)
        zernike_basis_conj_flat_t = self.zernike_basis_conj_flat.T
        moments = torch.matmul(x_flat, zernike_basis_conj_flat_t)

        norm_factors_complex = torch.complex(
            self.norm_factors, torch.zeros_like(self.norm_factors))
        norm_factors_complex = norm_factors_complex.unsqueeze(0).unsqueeze(0)

        normalized_moments = moments * norm_factors_complex
        return normalized_moments
