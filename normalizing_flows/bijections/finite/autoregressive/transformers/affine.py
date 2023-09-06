import math
from typing import Tuple

import torch

from normalizing_flows.bijections.finite.autoregressive.transformers.base import Transformer
from normalizing_flows.utils import get_batch_shape, sum_except_batch


class Affine(Transformer):
    """
    Affine transformer.

    Computes z = alpha * x + beta, where alpha > 0 and -inf < beta < inf.
    Alpha and beta have the same shape as x, i.e. the computation is performed elementwise.
    We use a minimum permitted scale m, 0 < m <= alpha, for numerical stability
    """

    def __init__(self, event_shape: torch.Size, min_scale: float = 1e-3):
        super().__init__(event_shape=event_shape)
        self.m = min_scale
        self.identity_unconstrained_alpha = math.log(1 - self.m)

    @property
    def n_parameters(self) -> int:
        return 2

    @property
    def default_parameters(self) -> torch.Tensor:
        default_u_alpha = torch.zeros(size=(1,))
        default_u_beta = torch.zeros(size=(1,))
        return torch.cat([default_u_alpha, default_u_beta], dim=0)

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        u_alpha = h[..., 0]
        alpha = torch.exp(self.identity_unconstrained_alpha + u_alpha / self.const) + self.m
        log_alpha = torch.log(alpha)

        u_beta = h[..., 1]
        beta = u_beta

        log_det = sum_except_batch(log_alpha, self.event_shape)
        return alpha * x + beta, log_det

    def inverse(self, z: torch.Tensor, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        u_alpha = h[..., 0]
        alpha = torch.exp(self.identity_unconstrained_alpha + u_alpha / self.const) + self.m
        log_alpha = torch.log(alpha)

        u_beta = h[..., 1]
        beta = u_beta

        log_det = -sum_except_batch(log_alpha, self.event_shape)
        return (z - beta) / alpha, log_det


class Affine2(Transformer):
    """
    Affine transformer with near-identity initialization.

    Computes z = alpha * x + beta, where alpha > 0 and -inf < beta < inf.
    Alpha and beta have the same shape as x, i.e. the computation is performed elementwise.

    In this implementation, we compute alpha and beta such that the initial map is near identity.
    We also use a minimum permitted scale m, 0 < m <= alpha, for numerical stability
    This means setting alpha = 1 + d(u_alpha) where -1 + m < d(u_alpha) < inf.

    We can verify that the following construction for function d is suitable:
    * g(u) = u / c + log(-log(m)); c>=1, 0<m<1  # c > 0 is enough, but c >= 1 desirably reduces input magnitude
    * f(u) = exp(g(u)) + log(m)
    * d(u) = exp(f(u)) - 1

    A change in u implies a change in log(log(d)), we may need a bigger step size when optimizing parameters of
    overarching bijections that use this transformer.
    """

    def __init__(self, event_shape: torch.Size, min_scale: float = 1e-3, **kwargs):
        super().__init__(event_shape=event_shape)
        assert 0 < min_scale < 1
        self.m = min_scale
        self.log_m = math.log(self.m)
        self.log_neg_log_m = math.log(-self.log_m)
        self.c = 100.0

    def compute_scale_and_shift(self, h):
        u_alpha = h[..., 0]
        g_alpha = u_alpha / self.c + self.log_neg_log_m
        f_alpha = torch.exp(g_alpha) + self.log_m
        d_alpha = torch.exp(f_alpha) - 1
        # d_alpha = self.m ** (1 - torch.exp(u_alpha / self.c)) - 1  # Rewritten
        # d_alpha = self.m * self.m ** (-torch.exp(u_alpha / self.c)) - 1  # Rewritten again

        alpha = 1 + d_alpha

        u_beta = h[..., 1]
        d_beta = u_beta / self.c
        beta = 0 + d_beta

        return alpha, beta

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        alpha, beta = self.compute_scale_and_shift(h)
        log_det = sum_except_batch(torch.log(alpha), self.event_shape)
        return alpha * x + beta, log_det

    def inverse(self, z: torch.Tensor, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        alpha, beta = self.compute_scale_and_shift(h)
        log_det = -sum_except_batch(torch.log(alpha), self.event_shape)
        return (z - beta) / alpha, log_det


class Shift(Transformer):
    def __init__(self, event_shape: torch.Size):
        super().__init__(event_shape=event_shape)

    @property
    def n_parameters(self) -> int:
        return 1

    @property
    def default_parameters(self) -> torch.Tensor:
        return torch.zeros(size=(1,))

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        beta = h[..., 0]
        batch_shape = get_batch_shape(x, self.event_shape)
        log_det = torch.zeros(batch_shape, device=x.device)
        return x + beta, log_det

    def inverse(self, z: torch.Tensor, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        beta = h[..., 0]
        batch_shape = get_batch_shape(z, self.event_shape)
        log_det = torch.zeros(batch_shape, device=z.device)
        return z - beta, log_det


class Scale(Transformer):
    """
    Scaling transformer.

    Computes z = alpha * x, where alpha > 0.
    We use a minimum permitted scale m, 0 < m <= alpha, for numerical stability
    """

    def __init__(self, event_shape: torch.Size, min_scale: float = 1e-3):
        super().__init__(event_shape=event_shape)
        self.m = min_scale
        self.const = 2.0
        self.u_alpha_1 = math.log(1 - self.m)

    @property
    def n_parameters(self) -> int:
        return 1

    def default_parameters(self) -> torch.Tensor:
        return torch.zeros(size=(1,))

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        u_alpha = h[..., 0]
        alpha = torch.exp(self.u_alpha_1 + u_alpha / self.const) + self.m
        log_alpha = torch.log(alpha)

        log_det = sum_except_batch(log_alpha, self.event_shape)
        return alpha * x, log_det

    def inverse(self, z: torch.Tensor, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        u_alpha = h[..., 0]
        alpha = torch.exp(self.u_alpha_1 + u_alpha / self.const) + self.m
        log_alpha = torch.log(alpha)

        log_det = -sum_except_batch(log_alpha, self.event_shape)
        return z / alpha, log_det
