from collections import Counter

import torch
import torch.nn as nn

from normalizing_flows.utils import Geometric, vjp_tensor


def power_series_log_abs_det_estimator(g: callable,
                                       x: torch.Tensor,
                                       training: bool,
                                       n_hutchinson_samples: int = 20,
                                       n_iterations: int = 8):
    # f(x) = x + g(x)
    # x.shape == (batch_size, event_size)
    # noise.shape == (batch_size, event_size, n_hutchinson_samples)
    # g(x).shape == (batch_size, event_size)

    batch_size, event_size = x.shape
    assert len(x.shape) == 2
    assert x.shape == (batch_size, event_size)
    assert n_iterations >= 2

    noise = torch.randn(size=(batch_size, event_size, n_hutchinson_samples))

    w = torch.clone(noise)
    log_abs_det_jac_f = torch.zeros(size=(batch_size,))
    g_value = None
    for k in range(1, n_iterations + 1):
        # Compute VJP, reshape appropriately for hutchinson averaging
        gs_r, ws_r = torch.autograd.functional.vjp(
            g,
            x[..., None].repeat(1, 1, n_hutchinson_samples).view(batch_size * n_hutchinson_samples, event_size),
            w.view(batch_size * n_hutchinson_samples, event_size)
        )

        if g_value is None:
            g_value = gs_r.view(batch_size, event_size, n_hutchinson_samples)[..., 0]

        w = ws_r.view(batch_size, event_size, n_hutchinson_samples)

        factor = (-1) ** (k + 1) / k
        # sum over event dim, average over hutchinson dim
        log_abs_det_jac_f += factor * torch.sum(w * noise, dim=1).mean(dim=1)
        assert log_abs_det_jac_f.shape == (batch_size,)
    return g_value, log_abs_det_jac_f


def roulette_log_abs_det_estimator(g: callable,
                                   x: torch.Tensor,
                                   training: bool,
                                   p: float = 0.5,
                                   n_hutchinson_samples: int = 20,
                                   n_roulette_samples: int = 20):
    """
    Estimate log[abs(det(grad(f)))](x) with a roulette approach, where f(x) = x + g(x); Lip(g) < 1.

    :param g:.
    :param x: input tensor.
    :param noise: noise tensor with the same shape as x.
    :param training: is the computation being performed for a model being trained.
    :return:
    """
    # f(x) = x + g(x)
    # x.shape == (batch_size, event_size)
    # g(x).shape == (batch_size, event_size)

    batch_size, event_size = x.shape
    dist = Geometric(probs=torch.tensor(p), minimum=4)

    total_roulette_iterations = sorted(dist.sample(torch.Size((n_roulette_samples,))).long().tolist())

    log_det_sum = torch.zeros(size=(batch_size, event_size), dtype=x.dtype)
    g_value = None
    for n_iterations, m in Counter(total_roulette_iterations).items():
        # Computing truncated power series with length n_iterations (m times), each with n_hutchinson_samples samples
        noise = torch.randn(size=(batch_size, event_size, n_hutchinson_samples, m), dtype=x.dtype)
        w = torch.clone(noise)
        vjp = torch.clone(noise)
        for k in range(1, n_iterations + 1):
            gs_r, ws_r = torch.autograd.functional.vjp(
                g,
                x[..., None, None].repeat(1, 1, n_hutchinson_samples, m).view(
                    batch_size * n_hutchinson_samples * m,
                    event_size
                ),
                w.view(batch_size * n_hutchinson_samples * m, event_size)
            )
            w = ws_r.view(batch_size, event_size, n_hutchinson_samples, m)
            if g_value is None:
                g_value = gs_r.view(batch_size, event_size, n_hutchinson_samples, m)[:, :, 0, 0]
            p_k = 1 - dist.cdf(torch.tensor(k - 1, dtype=torch.long))
            factor = (-1) ** (k + 1) / (k * p_k)
            vjp += factor * w

        _, vjp_jac_r = torch.autograd.functional.vjp(
            g,
            x[..., None, None].repeat(1, 1, n_hutchinson_samples, m).view(
                batch_size * n_hutchinson_samples * m,
                event_size
            ),
            vjp.view(batch_size * n_hutchinson_samples * m, event_size)
        )
        vjp_jac = vjp_jac_r.view(batch_size, event_size, n_hutchinson_samples, m)
        log_det_sum += torch.sum(vjp_jac, dim=(2, 3))  # shape: (batch_size, event_size)
    log_det_averaged = log_det_sum / (sum(total_roulette_iterations) * n_hutchinson_samples)  # (batch_size, event_size)
    log_det = torch.sum(log_det_averaged, dim=1)  # (batch_size,)
    return g_value, log_det


def roulette_log_abs_det_estimator____(g: callable,
                                       x: torch.Tensor,
                                       noise: torch.Tensor,
                                       training: bool,
                                       p: float = 0.5):
    """
    Estimate log[abs(det(grad(f)))](x) with a roulette approach, where f(x) = x + g(x); Lip(g) < 1.

    :param g:.
    :param x: input tensor.
    :param noise: noise tensor with the same shape as x.
    :param training: is the computation being performed for a model being trained.
    :return:
    """
    # f(x) = x + g(x)
    w = noise
    neumann_vjp = noise
    dist = Geometric(probs=torch.tensor(p), minimum=1)
    n_power_series = int(dist.sample())
    with torch.no_grad():
        for k in range(1, n_power_series + 1):
            # w = torch.autograd.grad(g_value, x, w, retain_graph=True)[0]
            g_value, w = torch.autograd.functional.vjp(g, x, w)
            # P(N >= k) = 1 - P(N < k) = 1 - P(N <= k - 1) = 1 - cdf(k - 1)
            p_k = 1 - dist.cdf(torch.tensor(k - 1, dtype=torch.long))
            neumann_vjp = neumann_vjp + (-1) ** k / (k * p_k) * w
    g_value, vjp_jac = torch.autograd.functional.vjp(g, x, neumann_vjp)
    # vjp_jac = torch.autograd.grad(g_value, x, neumann_vjp, create_graph=training)[0]
    log_abs_det_jac_f = torch.sum(vjp_jac * noise, dim=-1)
    return g_value, log_abs_det_jac_f


class LogDeterminantEstimator(torch.autograd.Function):
    """
    Given a function f(x) = x + g(x) with Lip(g) < 1, compute log[abs(det(grad(f)))](x) with Pytorch autodiff support.
    Autodiff support permits this function to be used in a computation graph.
    """

    # https://github.com/rtqichen/residual-flows/blob/master/lib/layers/iresblock.py#L186
    @staticmethod
    def forward(ctx,
                estimator_function: callable,
                g: nn.Module,
                x: torch.Tensor,
                training: bool,
                *g_params,
                **kwargs):
        ctx.training = training
        with torch.enable_grad():
            ctx.x = x
            g_value, log_det_f = estimator_function(g, x, training, **kwargs)
            ctx.g_value = g_value

            if training:
                # If a model is being trained,
                # compute the gradient of the log determinant in the forward pass and store it for later.
                # The gradient is computed w.r.t. x (first output) and w.r.t. the parameters of g (following outputs).
                grad_x, *grad_params = torch.autograd.grad(
                    log_det_f.sum(), (x,) + g_params, retain_graph=True, allow_unused=True
                )
                if grad_x is None:
                    grad_x = torch.zeros_like(x)
                ctx.save_for_backward(grad_x, *g_params, *grad_params)

        # Return g(x) and log(abs(det(grad(f))))(x)
        return (
            g_value.detach().requires_grad_(g_value.requires_grad),
            log_det_f.detach().requires_grad_(log_det_f.requires_grad)
        )

    @staticmethod
    def backward(ctx, grad_g, grad_logdetgrad):
        training = ctx.training
        if not training:
            raise ValueError('Provide training=True if using backward.')

        with torch.enable_grad():
            grad_x, *params_and_grad = ctx.saved_tensors
            g_value, x = ctx.g_value, ctx.x

            # Precomputed gradients
            g_params = params_and_grad[:len(params_and_grad) // 2]
            grad_params = params_and_grad[len(params_and_grad) // 2:]

            dg_x, *dg_params = torch.autograd.grad(g_value, [x] + g_params, grad_g, allow_unused=True)

        # Update based on gradient from log determinant.
        dL = grad_logdetgrad[0].detach()
        with torch.no_grad():
            grad_x.mul_(dL)
            grad_params = tuple([g.mul_(dL) if g is not None else None for g in grad_params])

        # Update based on gradient from g.
        with torch.no_grad():
            grad_x.add_(dg_x)
            grad_params = tuple([dg.add_(djac) if djac is not None else dg for dg, djac in zip(dg_params, grad_params)])

        return (None, None, grad_x, None, None, None, None) + grad_params


def log_det_roulette(g: nn.Module, x: torch.Tensor, training: bool = False, p: float = 0.5):
    return LogDeterminantEstimator.apply(
        lambda *args, **kwargs: roulette_log_abs_det_estimator(*args, **kwargs, p=p),
        g,
        x,
        training,
        *list(g.parameters())
    )


def log_det_power_series(g: nn.Module, x: torch.Tensor, training: bool = False, n_iterations: int = 8,
                         n_hutchinson_samples: int = 1):
    return LogDeterminantEstimator.apply(
        lambda *args, **kwargs: power_series_log_abs_det_estimator(*args, **kwargs, n_iterations=n_iterations),
        g,
        x,
        training,
        *list(g.parameters())
    )
