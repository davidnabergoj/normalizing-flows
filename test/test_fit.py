import pytest
import torch
from normalizing_flows import NICE, RealNVP, MAF, ElementwiseAffine, ElementwiseShift, ElementwiseRQSpline, Flow


@pytest.mark.parametrize('bijection_class', [
    NICE, RealNVP, MAF, ElementwiseAffine, ElementwiseShift, ElementwiseRQSpline
])
def test_standard_gaussian(bijection_class):
    torch.manual_seed(0)

    n_data = 10_000
    n_dim = 10
    x = torch.randn(size=(n_data, n_dim))

    bijection = bijection_class(event_shape=(n_dim,))
    flow = Flow(bijection)

    flow.fit(x, n_epochs=100)
    x_flow = flow.sample(100_000)

    x_mean = torch.mean(x_flow, dim=0)
    x_var = torch.var(x_flow, dim=0)

    assert torch.allclose(x_mean, torch.zeros(size=(n_dim,)), atol=0.05)
    assert torch.allclose(x_var, torch.ones(size=(n_dim,)), atol=0.1)


def test_diagonal_gaussian_elementwise_affine():
    torch.manual_seed(0)

    n_data = 10_000
    n_dim = 3
    sigma = torch.tensor([[1e-1, 1, 1e1]])
    x = torch.randn(size=(n_data, n_dim)) * sigma

    bijection = ElementwiseAffine(event_shape=(n_dim,))
    flow = Flow(bijection)

    # print(bijection.conditioner_transform.theta)

    flow.fit(x, n_epochs=500)

    # print(bijection.conditioner_transform.theta)

    x_flow = flow.sample(100_000)
    x_std = torch.std(x_flow, dim=0)
    relative_error = max((x_std - sigma.ravel()).abs() / sigma.ravel())

    assert relative_error < 0.1


@pytest.mark.parametrize('bijection_class', [ElementwiseAffine, ElementwiseRQSpline, RealNVP, MAF])
def test_diagonal_gaussian_1(bijection_class):
    torch.manual_seed(0)

    n_data = 10_000
    n_dim = 3
    sigma = torch.tensor([[1e-1, 1.0, 1e1]])
    x = torch.randn(size=(n_data, n_dim)) * sigma

    bijection = bijection_class(event_shape=(n_dim,))
    flow = Flow(bijection)

    flow.fit(x, n_epochs=500)
    x_flow = flow.sample(100_000)
    x_std = torch.std(x_flow, dim=0)
    relative_error = max((x_std - sigma.ravel()).abs() / sigma.ravel())

    assert relative_error < 0.1