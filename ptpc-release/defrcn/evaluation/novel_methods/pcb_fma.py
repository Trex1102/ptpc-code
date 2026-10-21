"""BMVC-v6 style PCB-FMA calibrator.

PCB-FMA keeps DeFRCN fixed and builds class prototypes from crop-level
foundation-model descriptors only. At inference, it fuses detector confidence
with RoI-to-prototype cosine similarity. Optionally, it can also blend the
original PCB similarity, matching the historical tri-modal implementation.
"""

import cv2
import logging
from typing import Dict, List

import torch
import torch.nn.functional as F

from .ptpc import PTPCFeatureExtractor as FoundationModelFeatureExtractor

logger = logging.getLogger(__name__)


class PCBFMA:
    """Foundation-model RoI prototype matching from the BMVC-v6 implementation."""

    def __init__(self, base_pcb, cfg):
        self.base_pcb = base_pcb
        self.cfg = cfg
        fma_cfg = cfg.NOVEL_METHODS.PCB_FMA

        self.det_weight = float(fma_cfg.DET_WEIGHT)
        self.fm_weight = float(fma_cfg.FM_WEIGHT)
        self.use_original_pcb = bool(fma_cfg.USE_ORIGINAL_PCB)
        self.original_pcb_weight = float(fma_cfg.ORIGINAL_PCB_WEIGHT)

        total = self.det_weight + self.fm_weight + (
            self.original_pcb_weight if self.use_original_pcb else 0.0
        )
        if total > 0:
            self.det_weight /= total
            self.fm_weight /= total
            if self.use_original_pcb:
                self.original_pcb_weight /= total

        self.fm_extractor = FoundationModelFeatureExtractor(
            model_name=str(fma_cfg.FM_MODEL_NAME),
            model_path=str(fma_cfg.FM_MODEL_PATH),
            feat_dim=int(fma_cfg.FM_FEAT_DIM),
            crop_size=int(fma_cfg.ROI_SIZE),
            roi_size=int(fma_cfg.ROI_SIZE),
            batch_size=int(fma_cfg.BATCH_SIZE),
            device=str(cfg.MODEL.DEVICE),
        )

        self.fm_prototypes: Dict[int, torch.Tensor] = {}
        if self.fm_extractor.available:
            self._build_fm_prototypes()
        else:
            logger.warning("PCB-FMA encoder unavailable; falling back to the base PCB path.")

    def _build_fm_prototypes(self):
        logger.info("PCB-FMA: building support prototypes in FM space.")
        class_features: Dict[int, List[torch.Tensor]] = {}
        dataloader = self.base_pcb.dataloader

        for index in range(len(dataloader.dataset)):
            inputs = [dataloader.dataset[index]]
            img = cv2.imread(inputs[0]["file_name"])
            if img is None:
                continue

            inst = inputs[0]["instances"]
            if len(inst) == 0:
                continue

            img_h = img.shape[0]
            ratio = img_h / float(inst.image_size[0])
            gt_boxes = inst.gt_boxes.tensor.clone() * ratio
            labels = inst.gt_classes.cpu()

            fm_features = self.fm_extractor.extract_roi_features(img, gt_boxes)
            if fm_features is None:
                continue

            fm_features = fm_features.detach().cpu()
            for idx, cls_tensor in enumerate(labels):
                cls = int(cls_tensor.item())
                class_features.setdefault(cls, []).append(fm_features[idx])

        for cls, feats in class_features.items():
            proto = torch.stack(feats, dim=0).mean(dim=0)
            self.fm_prototypes[cls] = F.normalize(proto, dim=0)

        logger.info("PCB-FMA: built prototypes for %d classes.", len(self.fm_prototypes))

    def execute_calibration(self, inputs, dts):
        if not self.fm_extractor.available or not self.fm_prototypes:
            return self.base_pcb.execute_calibration(inputs, dts)
        if len(dts) == 0 or len(dts[0]["instances"]) == 0:
            return dts

        img = cv2.imread(inputs[0]["file_name"])
        if img is None:
            return dts

        instances = dts[0]["instances"]
        scores = instances.scores
        ileft = int((scores > self.base_pcb.pcb_upper).sum().item())
        iright = int((scores > self.base_pcb.pcb_lower).sum().item())
        if ileft >= iright:
            return dts

        pred_boxes = instances.pred_boxes[ileft:iright]
        if len(pred_boxes) == 0:
            return dts

        pred_classes = instances.pred_classes
        box_tensor = pred_boxes.tensor
        fm_features = self.fm_extractor.extract_roi_features(img, box_tensor)
        if fm_features is None:
            return self.base_pcb.execute_calibration(inputs, dts)

        pcb_features = None
        area_norm = None
        if self.use_original_pcb:
            img_h, img_w = img.shape[:2]
            boxes = [pred_boxes.to(self.base_pcb.device)]
            pcb_features = self.base_pcb.extract_roi_features(img, boxes)
            area_norm = self.base_pcb._normalized_area(box_tensor, img_h, img_w)

        score_device = scores.device
        score_dtype = scores.dtype
        for index in range(ileft, iright):
            cls = int(pred_classes[index].item())
            if cls in self.base_pcb.exclude_cls:
                continue

            query_index = index - ileft
            det_score = float(scores[index].item())

            fm_sim = 0.0
            if cls in self.fm_prototypes:
                query_fm = F.normalize(fm_features[query_index], dim=0)
                proto_fm = self.fm_prototypes[cls].to(query_fm.device)
                fm_sim = (float(torch.dot(query_fm, proto_fm).item()) + 1.0) / 2.0

            fused = self.det_weight * det_score + self.fm_weight * fm_sim
            if self.use_original_pcb and cls in self.base_pcb.prototypes:
                proto_bank = self.base_pcb._select_proto_bank(
                    cls, float(area_norm[query_index].item())
                )
                if proto_bank is not None:
                    pcb_sim = self.base_pcb._match_similarity(
                        pcb_features[query_index], proto_bank
                    )
                    pcb_sim = (pcb_sim + 1.0) / 2.0
                    fused += self.original_pcb_weight * pcb_sim

            scores[index] = torch.tensor(
                max(0.0, min(1.0, fused)), device=score_device, dtype=score_dtype
            )

        return dts

    def __getattr__(self, name):
        return getattr(self.base_pcb, name)


def build_pcb_fma(base_pcb, cfg):
    """Factory for BMVC-v6 style PCB-FMA."""
    return PCBFMA(base_pcb, cfg)
