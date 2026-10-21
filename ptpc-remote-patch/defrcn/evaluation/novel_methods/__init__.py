"""Paper-specific calibration wrappers for the frozen-FM release."""

from .pcb_fma import FoundationModelFeatureExtractor, PCBFMA, build_pcb_fma
from .pcb_fma_enhanced import PCBFMAEnhanced, build_pcb_fma_enhanced
from .negative_proto_guard import (
    NegativeProtoGuard,
    build_neg_proto_guard,
    build_pcb_fma_enhanced_neg,
)
from .pcb_resnet_competitive import (
    PCBResNetCompetitive,
    PCBResNetGuard,
    build_pcb_resnet101,
    build_pcb_resnet101_competitive,
    build_pcb_resnet101_competitive_guard,
    build_pcb_resnet101_guard,
)
from .ptpc import (
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
    "NegativeProtoGuard",
    "build_neg_proto_guard",
    "build_pcb_fma_enhanced_neg",
    "PCBResNetCompetitive",
    "PCBResNetGuard",
    "build_pcb_resnet101",
    "build_pcb_resnet101_competitive",
    "build_pcb_resnet101_competitive_guard",
    "build_pcb_resnet101_guard",
    "PTPCCalibrator",
    "PTPCFeatureExtractor",
    "PTPCGuard",
    "PTPCSupportContext",
    "build_fm_only_support",
    "build_ptpc",
]


def build_novel_method_pcb(base_pcb, cfg, method_name: str):
    """Build one of the supported paper-specific calibration wrappers."""

    method_builders = {
        "pcb_resnet101": build_pcb_resnet101,
        "resnet101": build_pcb_resnet101,
        "pcb_resnet": build_pcb_resnet101,
        "pcb_resnet101_competitive": build_pcb_resnet101_competitive,
        "resnet101_competitive": build_pcb_resnet101_competitive,
        "pcb_resnet101_guarded": build_pcb_resnet101_guard,
        "resnet101_guarded": build_pcb_resnet101_guard,
        "pcb_resnet101_competitive_guarded": build_pcb_resnet101_competitive_guard,
        "resnet101_competitive_guarded": build_pcb_resnet101_competitive_guard,
        "pcb_fma": build_pcb_fma,
        "fma": build_pcb_fma,
        "foundation_model": build_pcb_fma,
        "fm_alignment": build_pcb_fma,
        "pcb_fma_enhanced": build_pcb_fma_enhanced,
        "fma_enhanced": build_pcb_fma_enhanced,
        "enhanced_fma": build_pcb_fma_enhanced,
        "competitive_calibration": build_pcb_fma_enhanced,
        "neg_proto_guard": build_neg_proto_guard,
        "npg": build_neg_proto_guard,
        "negative_guard": build_neg_proto_guard,
        "pcb_fma_enhanced_neg": build_pcb_fma_enhanced_neg,
        "fma_enhanced_neg": build_pcb_fma_enhanced_neg,
        "enhanced_fma_neg": build_pcb_fma_enhanced_neg,
        "competitive_calibration_guarded": build_pcb_fma_enhanced_neg,
        "ptpc": build_ptpc,
    }

    method_name_lower = method_name.lower()
    if method_name_lower not in method_builders:
        raise ValueError(
            f"Unknown novel method: {method_name}. "
            f"Available: {list(method_builders.keys())}"
        )

    return method_builders[method_name_lower](base_pcb, cfg)
