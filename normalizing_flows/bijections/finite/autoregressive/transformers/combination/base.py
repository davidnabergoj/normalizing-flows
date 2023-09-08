import torch
from typing import Tuple

from normalizing_flows.bijections.finite.autoregressive.transformers.base import Transformer
from normalizing_flows.utils import get_batch_shape


class Combination(Transformer):
    def __init__(self, event_shape: torch.Size, components: list[Transformer]):
        super().__init__(event_shape)
        self.components = components
        self.n_components = len(self.components)

    @property
    def n_parameters(self) -> int:
        return sum([c.n_parameters for c in self.components])

    @property
    def default_parameters(self) -> torch.Tensor:
        return torch.cat([c.default_parameters for c in self.components], dim=0)

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # h.shape = (*batch_size, *event_shape, n_components * n_output_parameters)
        # We assume last dim is ordered as [c1, c2, ..., ck] i.e. sequence of parameter vectors, one for each component.
        batch_shape = get_batch_shape(x, self.event_shape)
        log_det = torch.zeros(size=batch_shape)
        start_index = 0
        for i in range(self.n_components):
            component = self.components[i]
            x, log_det_increment = component.forward(x, h[..., start_index:start_index + component.n_parameters])
            log_det += log_det_increment
            start_index += component.n_parameters
        z = x
        return z, log_det

    def inverse(self, z: torch.Tensor, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # h.shape = (*batch_size, *event_shape, n_components * n_output_parameters)
        # We assume last dim is ordered as [c1, c2, ..., ck] i.e. sequence of parameter vectors, one for each component.
        batch_shape = get_batch_shape(z, self.event_shape)
        log_det = torch.zeros(size=batch_shape)
        c = self.n_parameters
        for i in range(self.n_components):
            component = self.components[self.n_components - i - 1]
            c -= component.n_parameters
            z, log_det_increment = component.inverse(z, h[..., c:c + component.n_parameters])
            log_det += log_det_increment
        x = z
        return x, log_det