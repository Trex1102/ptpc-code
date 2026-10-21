"""PTPC deployed calibration method."""

from .pcb_fma import PCBFMA, build_pcb_fma
from .pcb_fma_enhanced import PCBFMAEnhanced, build_pcb_fma_enhanced
from .ptpc import (
    FoundationModelFeatureExtractor,
    PTPCCalibrator,
    PTPCFeatureExtractor,
    PTPCGuard,
    PTPCSupportContext,
    build_fm_only_support,
    build_ptpc,
)

__all__ = [
    "FoundationModelFeatureExtractor",
    "PCBFMA",
    "build_pcb_fma",
    "PCBFMAEnhanced",
    "build_pcb_fma_enhanced",
    "PTPCCalibrator",
    "PTPCFeatureExtractor",
    "PTPCGuard",
    "PTPCSupportContext",
    "build_fm_only_support",
    "build_ptpc",
    "build_ptpc_calibrator",
]


def build_ptpc_calibrator(base_pcb, cfg, variant: str):
    """Build the requested novel-method calibrator."""
    variant_key = variant.lower().replace("-", "_")
    if variant_key == "ptpc":
        return build_ptpc(base_pcb, cfg)
    if variant_key in {"pcb_fma", "fma", "foundation_model"}:
        return build_pcb_fma(base_pcb, cfg)
    if variant_key in {"pcb_fma_enhanced", "pcb_fma_enhanced_noaug", "fma_enhanced"}:
        return build_pcb_fma_enhanced(base_pcb, cfg)
    raise ValueError(
        "Unknown novel method '{}'. Available: ['ptpc', 'pcb-fma', 'pcb-fma-enhanced']".format(variant)
    )
