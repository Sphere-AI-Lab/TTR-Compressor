"""Uniform TTR block compression with a learned positive ridge diagonal."""

from abc import ABC, abstractmethod
import math

import torch
from torch import nn
from torch.nn import functional as F


class Compressor(nn.Module, ABC):
    """Map (batch, blocks, tokens, heads, channels) to one vector per block."""

    def __init__(self, num_heads: int, head_dim: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim

    def _validate_inputs(self, inputs: torch.Tensor) -> None:
        if inputs.ndim != 5 or inputs.shape[-2:] != (self.num_heads, self.head_dim):
            raise ValueError(
                f"Expected (B, N, S, {self.num_heads}, {self.head_dim}), "
                f"got {tuple(inputs.shape)}"
            )

    @abstractmethod
    def reset_parameters(self) -> None:
        """Initialize the compressor's trainable parameters."""


class TTRCompressor(Compressor):
    """Uniform ridge regression over a block, solved in token space."""

    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        value_init_std: float = 0.02,
        ridge_floor: float = 0.01,
        ridge_init: float = 0.1,
    ):
        super().__init__(num_heads, head_dim)
        if not all(math.isfinite(v) for v in (value_init_std, ridge_floor, ridge_init)):
            raise ValueError("Compressor initialization must be finite")
        if value_init_std < 0 or ridge_floor <= 0 or ridge_init <= ridge_floor:
            raise ValueError("Require value_init_std >= 0 and ridge_init > ridge_floor > 0")
        self.value_init_std = float(value_init_std)
        self.ridge_floor = float(ridge_floor)
        self.ridge_init = float(ridge_init)
        self.value_weight = nn.Parameter(torch.empty(num_heads, head_dim))
        self.ridge_rho = nn.Parameter(torch.empty(head_dim))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.value_weight, mean=0.0, std=self.value_init_std)
        nn.init.constant_(self.ridge_rho, math.log(math.expm1(self.ridge_init - self.ridge_floor)))

    def ridge_diagonal(self) -> torch.Tensor:
        return self.ridge_floor + F.softplus(self.ridge_rho.float())

    def _project_values(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.einsum("...hd,hd->...h", inputs, self.value_weight.to(inputs.dtype))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        self._validate_inputs(inputs)
        batch, blocks, tokens, heads, channels = inputs.shape
        if tokens == 0 or batch == 0 or blocks == 0:
            return inputs.new_zeros(batch, blocks, heads, channels)
        flat = inputs.reshape(batch * blocks, tokens, heads, channels)
        # Retain the input-dtype rounding of the original normalized keys.
        key = F.normalize(flat.float(), p=2.0, dim=-1).to(inputs.dtype)
        value = self._project_values(flat)
        key = key.transpose(1, 2).float().reshape(-1, tokens, channels)
        value = value.transpose(1, 2).float().reshape(-1, tokens)
        inverse_sqrt = self.ridge_diagonal().reciprocal().to(key.dtype).sqrt()
        scaled_key = key * inverse_sqrt
        gram = torch.matmul(scaled_key, scaled_key.transpose(-1, -2))
        eye = torch.eye(tokens, dtype=key.dtype, device=key.device)
        coef = torch.cholesky_solve(
            value.unsqueeze(-1), torch.linalg.cholesky(eye + gram)
        ).squeeze(-1)
        state = (
            torch.matmul(scaled_key.transpose(-1, -2), coef.unsqueeze(-1))
            .squeeze(-1)
            .mul(inverse_sqrt)
        )
        return state.reshape(batch, blocks, heads, channels).to(inputs.dtype)


class TTRConvCompressor(TTRCompressor):
    """Centered five-token value convolution, independently inside each block."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.value_conv_weight = nn.Parameter(torch.zeros(self.num_heads * self.head_dim, 1, 5))
        self._reset_value_conv()

    def _reset_value_conv(self) -> None:
        with torch.no_grad():
            self.value_conv_weight.zero_()
            self.value_conv_weight[..., 2].fill_(1.0)

    def reset_parameters(self) -> None:
        super().reset_parameters()
        if hasattr(self, "value_conv_weight"):
            self._reset_value_conv()

    def _project_values(self, inputs: torch.Tensor) -> torch.Tensor:
        length = inputs.shape[-3]
        channels = self.num_heads * self.head_dim
        sequence = inputs.reshape(-1, length, channels).transpose(1, 2)
        convolved = F.conv1d(
            F.pad(sequence, (2, 2)), self.value_conv_weight.to(inputs.dtype), groups=channels
        ).transpose(1, 2).reshape_as(inputs)
        return super()._project_values(convolved)


COMPRESSOR_REGISTRY = {"ttr": TTRCompressor, "ttr_conv": TTRConvCompressor}


def build_compressor(config, num_heads: int, head_dim: int) -> Compressor:
    try:
        constructor = COMPRESSOR_REGISTRY[config.compressor_type]
    except KeyError as exc:
        raise ValueError(f"Unknown TTR compressor: {config.compressor_type!r}") from exc
    return constructor(
        num_heads=num_heads,
        head_dim=head_dim,
        value_init_std=config.compressor_ttr_value_init_std,
        ridge_floor=config.compressor_ttr_ridge_learnable_floor,
        ridge_init=config.compressor_ttr_ridge_learnable_init,
    )


__all__ = ["Compressor", "TTRCompressor", "TTRConvCompressor", "build_compressor"]
