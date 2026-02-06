"""Factorised Noisy Layers for exploration in DQN."""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _factorised_noise(size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Sample factorised noise: f(x) = sign(x) * sqrt(|x|)."""
    x = torch.randn(size, device=device, dtype=dtype)
    return x.sign() * x.abs().sqrt()


class NoisyLinear(nn.Module):
    """
    Factorised Noisy Linear layer.
    
    Replaces standard Linear with learnable noise parameters.
    Weight = weight_mu + weight_sigma * noise_weight
    Bias = bias_mu + bias_sigma * noise_bias
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        sigma_init: float = 0.5,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sigma_init = sigma_init

        # Learnable parameters
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))

        # Factorised noise (not learnable, resampled)
        self.register_buffer("noise_in", torch.zeros(in_features))
        self.register_buffer("noise_out", torch.zeros(out_features))

        self._init_parameters()
        self.reset_noise()

    def _init_parameters(self) -> None:
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.bias_mu.data.uniform_(-mu_range, mu_range)

        sigma_val = self.sigma_init / math.sqrt(self.in_features)
        self.weight_sigma.data.fill_(sigma_val)
        self.bias_sigma.data.fill_(sigma_val)

    def reset_noise(self) -> None:
        """Resample factorised noise."""
        self.noise_in = _factorised_noise(
            self.in_features, self.weight_mu.device, self.weight_mu.dtype
        )
        self.noise_out = _factorised_noise(
            self.out_features, self.weight_mu.device, self.weight_mu.dtype
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            # Factorised noise: outer product
            noise_weight = self.noise_out.unsqueeze(1) * self.noise_in.unsqueeze(0)
            noise_bias = self.noise_out

            weight = self.weight_mu + self.weight_sigma * noise_weight
            bias = self.bias_mu + self.bias_sigma * noise_bias
        else:
            # Eval mode: use mean only (no noise)
            weight = self.weight_mu
            bias = self.bias_mu

        return F.linear(x, weight, bias)

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, sigma_init={self.sigma_init}"


class NoisyConv1d(nn.Module):
    """
    Factorised Noisy Conv1d layer (kernel_size=1).
    
    For use in DQN heads where Conv1d(in_ch, out_ch, kernel_size=1) is used.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        sigma_init: float = 0.5,
    ):
        super().__init__()
        if kernel_size != 1:
            raise ValueError("NoisyConv1d only supports kernel_size=1")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.sigma_init = sigma_init

        # Learnable parameters (kernel shape: out_ch, in_ch, 1)
        self.weight_mu = nn.Parameter(torch.empty(out_channels, in_channels, 1))
        self.weight_sigma = nn.Parameter(torch.empty(out_channels, in_channels, 1))
        self.bias_mu = nn.Parameter(torch.empty(out_channels))
        self.bias_sigma = nn.Parameter(torch.empty(out_channels))

        # Factorised noise
        self.register_buffer("noise_in", torch.zeros(in_channels))
        self.register_buffer("noise_out", torch.zeros(out_channels))

        self._init_parameters()
        self.reset_noise()

    def _init_parameters(self) -> None:
        mu_range = 1 / math.sqrt(self.in_channels)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.bias_mu.data.uniform_(-mu_range, mu_range)

        sigma_val = self.sigma_init / math.sqrt(self.in_channels)
        self.weight_sigma.data.fill_(sigma_val)
        self.bias_sigma.data.fill_(sigma_val)

    def reset_noise(self) -> None:
        """Resample factorised noise."""
        self.noise_in = _factorised_noise(
            self.in_channels, self.weight_mu.device, self.weight_mu.dtype
        )
        self.noise_out = _factorised_noise(
            self.out_channels, self.weight_mu.device, self.weight_mu.dtype
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            # Factorised noise for kernel (out_ch, in_ch, 1)
            noise_weight = (self.noise_out.unsqueeze(1) * self.noise_in.unsqueeze(0)).unsqueeze(2)
            noise_bias = self.noise_out

            weight = self.weight_mu + self.weight_sigma * noise_weight
            bias = self.bias_mu + self.bias_sigma * noise_bias
        else:
            weight = self.weight_mu
            bias = self.bias_mu

        return F.conv1d(x, weight, bias)

    def extra_repr(self) -> str:
        return f"in_channels={self.in_channels}, out_channels={self.out_channels}, kernel_size={self.kernel_size}, sigma_init={self.sigma_init}"


class NoisyConv2d(nn.Module):
    """
    Factorised Noisy Conv2d layer (kernel_size=1 or 3).
    
    For use in DQN heads where Conv2d is used for per-cell outputs.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        padding: int = 0,
        sigma_init: float = 0.5,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.sigma_init = sigma_init

        # Learnable parameters (kernel shape: out_ch, in_ch, kh, kw)
        self.weight_mu = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size, kernel_size))
        self.weight_sigma = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size, kernel_size))
        self.bias_mu = nn.Parameter(torch.empty(out_channels))
        self.bias_sigma = nn.Parameter(torch.empty(out_channels))

        # Factorised noise
        fan_in = in_channels * kernel_size * kernel_size
        self.register_buffer("noise_in", torch.zeros(fan_in))
        self.register_buffer("noise_out", torch.zeros(out_channels))

        self._init_parameters()
        self.reset_noise()

    def _init_parameters(self) -> None:
        fan_in = self.in_channels * self.kernel_size * self.kernel_size
        mu_range = 1 / math.sqrt(fan_in)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.bias_mu.data.uniform_(-mu_range, mu_range)

        sigma_val = self.sigma_init / math.sqrt(fan_in)
        self.weight_sigma.data.fill_(sigma_val)
        self.bias_sigma.data.fill_(sigma_val)

    def reset_noise(self) -> None:
        """Resample factorised noise."""
        fan_in = self.in_channels * self.kernel_size * self.kernel_size
        self.noise_in = _factorised_noise(
            fan_in, self.weight_mu.device, self.weight_mu.dtype
        )
        self.noise_out = _factorised_noise(
            self.out_channels, self.weight_mu.device, self.weight_mu.dtype
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            # Factorised noise reshaped to kernel shape
            noise_weight = (self.noise_out.unsqueeze(1) * self.noise_in.unsqueeze(0)).view(
                self.out_channels, self.in_channels, self.kernel_size, self.kernel_size
            )
            noise_bias = self.noise_out

            weight = self.weight_mu + self.weight_sigma * noise_weight
            bias = self.bias_mu + self.bias_sigma * noise_bias
        else:
            weight = self.weight_mu
            bias = self.bias_mu

        return F.conv2d(x, weight, bias, padding=self.padding)

    def extra_repr(self) -> str:
        return f"in_channels={self.in_channels}, out_channels={self.out_channels}, kernel_size={self.kernel_size}, padding={self.padding}, sigma_init={self.sigma_init}"
