from typing import List

import torch

from normalizing_flows.bijections.finite.autoregressive.conditioner_transforms import ConditionerTransform
from normalizing_flows.bijections.finite.autoregressive.layers import (
    ShiftCoupling,
    AffineCoupling,
    AffineForwardMaskedAutoregressive,
    AffineInverseMaskedAutoregressive,
    RQSCoupling,
    RQSForwardMaskedAutoregressive,
    RQSInverseMaskedAutoregressive,
    InverseAffineCoupling,
    DSCoupling,
    ElementwiseAffine,
    UMNNMaskedAutoregressive,
    LRSCoupling,
    LRSForwardMaskedAutoregressive, ElementwiseShift
)
from normalizing_flows.bijections.base import BijectiveComposition, Bijection
from normalizing_flows.bijections.finite.linear import ReversePermutation


class AutoregressiveArchitecture(BijectiveComposition):
    def __init__(self, event_shape, layers: List[Bijection], **kwargs):
        if isinstance(event_shape, int):
            event_shape = (event_shape,)
        super().__init__(event_shape, layers, **kwargs)

    def l2_regularization(self):
        total = 0.0
        n_parameters = 0
        for m in self.modules():
            if isinstance(m, ConditionerTransform):
                n_parameters += sum(p.numel() for p in m.parameters())
                total += sum(torch.sum(torch.square(p)) for p in m.parameters())
        return total / n_parameters


class NICE(AutoregressiveArchitecture):
    def __init__(self, event_shape, n_layers: int = 2, **kwargs):
        bijections = [ElementwiseAffine(event_shape=event_shape)]
        for _ in range(n_layers):
            bijections.extend([
                ReversePermutation(event_shape=event_shape),
                ShiftCoupling(event_shape=event_shape)
            ])
        bijections.append(ElementwiseAffine(event_shape=event_shape))
        super().__init__(event_shape, bijections, **kwargs)


class RealNVP(AutoregressiveArchitecture):
    def __init__(self, event_shape, n_layers: int = 2, **kwargs):
        bijections = [ElementwiseAffine(event_shape=event_shape)]
        for _ in range(n_layers):
            bijections.extend([
                ReversePermutation(event_shape=event_shape),
                AffineCoupling(event_shape=event_shape)
            ])
        bijections.append(ElementwiseAffine(event_shape=event_shape))
        super().__init__(event_shape, bijections, **kwargs)


class InverseRealNVP(AutoregressiveArchitecture):
    def __init__(self, event_shape, n_layers: int = 2, **kwargs):
        bijections = [ElementwiseAffine(event_shape=event_shape)]
        for _ in range(n_layers):
            bijections.extend([
                ReversePermutation(event_shape=event_shape),
                InverseAffineCoupling(event_shape=event_shape)
            ])
        bijections.append(ElementwiseAffine(event_shape=event_shape))
        super().__init__(event_shape, bijections, **kwargs)


class MAF(AutoregressiveArchitecture):
    """
    Expressive bijection with slightly unstable inverse due to autoregressive formulation.
    """

    def __init__(self, event_shape, n_layers: int = 2, **kwargs):
        bijections = [ElementwiseAffine(event_shape=event_shape)]
        for _ in range(n_layers):
            bijections.extend([
                ReversePermutation(event_shape=event_shape),
                AffineForwardMaskedAutoregressive(event_shape=event_shape)
            ])
        bijections.append(ElementwiseAffine(event_shape=event_shape))
        super().__init__(event_shape, bijections, **kwargs)


class IAF(AutoregressiveArchitecture):
    def __init__(self, event_shape, n_layers: int = 2, **kwargs):
        bijections = [ElementwiseAffine(event_shape=event_shape)]
        for _ in range(n_layers):
            bijections.extend([
                ReversePermutation(event_shape=event_shape),
                AffineInverseMaskedAutoregressive(event_shape=event_shape)
            ])
        bijections.append(ElementwiseAffine(event_shape=event_shape))
        super().__init__(event_shape, bijections, **kwargs)


class CouplingRQNSF(AutoregressiveArchitecture):
    def __init__(self, event_shape, n_layers: int = 2, **kwargs):
        bijections = [ElementwiseAffine(event_shape=event_shape)]
        for _ in range(n_layers):
            bijections.extend([
                ReversePermutation(event_shape=event_shape),
                RQSCoupling(event_shape=event_shape)
            ])
        bijections.append(ElementwiseAffine(event_shape=event_shape))
        super().__init__(event_shape, bijections, **kwargs)


class MaskedAutoregressiveRQNSF(AutoregressiveArchitecture):
    """
    Expressive bijection with unstable inverse due to autoregressive formulation.
    """

    def __init__(self, event_shape, n_layers: int = 2, **kwargs):
        bijections = [ElementwiseAffine(event_shape=event_shape)]
        for _ in range(n_layers):
            bijections.extend([
                ReversePermutation(event_shape=event_shape),
                RQSForwardMaskedAutoregressive(event_shape=event_shape)
            ])
        bijections.append(ElementwiseAffine(event_shape=event_shape))
        super().__init__(event_shape, bijections, **kwargs)


class CouplingLRS(AutoregressiveArchitecture):
    def __init__(self, event_shape, n_layers: int = 2, **kwargs):
        bijections = [ElementwiseShift(event_shape=event_shape)]
        for _ in range(n_layers):
            bijections.extend([
                ReversePermutation(event_shape=event_shape),
                LRSCoupling(event_shape=event_shape)
            ])
        bijections.append(ElementwiseShift(event_shape=event_shape))
        super().__init__(event_shape, bijections, **kwargs)


class MaskedAutoregressiveLRS(AutoregressiveArchitecture):
    def __init__(self, event_shape, n_layers: int = 2, **kwargs):
        bijections = [ElementwiseShift(event_shape=event_shape)]
        for _ in range(n_layers):
            bijections.extend([
                ReversePermutation(event_shape=event_shape),
                LRSForwardMaskedAutoregressive(event_shape=event_shape)
            ])
        bijections.append(ElementwiseShift(event_shape=event_shape))
        super().__init__(event_shape, bijections, **kwargs)


class InverseAutoregressiveRQNSF(AutoregressiveArchitecture):
    def __init__(self, event_shape, n_layers: int = 2, **kwargs):
        bijections = [ElementwiseAffine(event_shape=event_shape)]
        for _ in range(n_layers):
            bijections.extend([
                ReversePermutation(event_shape=event_shape),
                RQSInverseMaskedAutoregressive(event_shape=event_shape)
            ])
        bijections.append(ElementwiseAffine(event_shape=event_shape))
        super().__init__(event_shape, bijections, **kwargs)


class CouplingDSF(AutoregressiveArchitecture):
    def __init__(self, event_shape, n_layers: int = 2, **kwargs):
        bijections = [ElementwiseAffine(event_shape=event_shape)]
        for _ in range(n_layers):
            bijections.extend([
                ReversePermutation(event_shape=event_shape),
                DSCoupling(event_shape=event_shape)  # TODO specify percent of global parameters
            ])
        bijections.append(ElementwiseAffine(event_shape=event_shape))
        super().__init__(event_shape, bijections, **kwargs)


class UMNNMAF(AutoregressiveArchitecture):
    def __init__(self, event_shape, n_layers: int = 2, **kwargs):
        bijections = [ElementwiseAffine(event_shape=event_shape)]
        for _ in range(n_layers):
            bijections.extend([
                ReversePermutation(event_shape=event_shape),
                UMNNMaskedAutoregressive(event_shape=event_shape)
            ])
        bijections.append(ElementwiseAffine(event_shape=event_shape))
        super().__init__(event_shape, bijections, **kwargs)
