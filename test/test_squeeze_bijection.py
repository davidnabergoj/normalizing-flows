import torch
import pytest

from normalizing_flows.bijections.finite.multiscale.base import Squeeze


@pytest.mark.parametrize('batch_shape', [(1,), (2,), (2, 3)])
@pytest.mark.parametrize('channels', [1, 3, 10])
@pytest.mark.parametrize('height', [4, 16, 32])
@pytest.mark.parametrize('width', [4, 16, 32])
def test_reconstruction(batch_shape, channels, height, width):
    torch.manual_seed(0)
    x = torch.randn(size=(*batch_shape, channels, height, width))
    layer = Squeeze(event_shape=x.shape[-3:])
    z, log_det_forward = layer.forward(x)
    x_reconstructed, log_det_inverse = layer.inverse(z)

    assert z.shape == (*batch_shape, 4 * channels, height // 2, width // 2)
    assert torch.allclose(x_reconstructed, x)
    assert torch.allclose(log_det_forward, torch.zeros_like(log_det_forward))
    assert torch.allclose(log_det_forward, log_det_inverse)
