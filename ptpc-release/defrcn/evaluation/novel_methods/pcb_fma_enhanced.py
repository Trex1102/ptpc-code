"""BMVC-v6 style PCB-FMA-Enhanced calibrator.

This variant keeps the PCB-FMA RoI crop descriptor pipeline, but replaces the
independent per-class cosine score with temperature-scaled class competition.
Support augmentation is configurable and can be disabled to isolate the scoring
change from additional support views.
"""

import cv2
import logging
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from .pcb_fma import FoundationModelFeatureExtractor

logger = logging.getLogger(__name__)


class PCBFMAEnhanced:
    """PCB-FMA with support-view averaging and competitive similarity."""

    def __init__(self, base_pcb, cfg):
        self.base_pcb = base_pcb
        self.cfg = cfg
        enh_cfg = cfg.NOVEL_METHODS.PCB_FMA_ENHANCED

        self.det_weight = float(enh_cfg.DET_WEIGHT)
        self.fm_weight = float(enh_cfg.FM_WEIGHT)
        self.use_original_pcb = bool(enh_cfg.USE_ORIGINAL_PCB)
        self.original_pcb_weight = float(enh_cfg.ORIGINAL_PCB_WEIGHT)

        self.enable_flip = bool(enh_cfg.AUG_FLIP)
        self.enable_multicrop = bool(enh_cfg.AUG_MULTICROP)
        self.multicrop_scales = list(enh_cfg.AUG_MULTICROP_SCALES)
        self.multicrop_num = int(enh_cfg.AUG_MULTICROP_NUM)

        self.temperature = float(enh_cfg.TEMPERATURE)
        self.competitive_mode = str(enh_cfg.COMPETITIVE_MODE).lower()

        total = self.det_weight + self.fm_weight + (
            self.original_pcb_weight if self.use_original_pcb else 0.0
        )
        if total > 0:
            self.det_weight /= total
            self.fm_weight /= total
            if self.use_original_pcb:
                self.original_pcb_weight /= total

        self.fm_extractor = FoundationModelFeatureExtractor(
            model_name=str(enh_cfg.FM_MODEL_NAME),
            model_path=str(enh_cfg.FM_MODEL_PATH),
            feat_dim=int(enh_cfg.FM_FEAT_DIM),
            crop_size=int(enh_cfg.ROI_SIZE),
            roi_size=int(enh_cfg.ROI_SIZE),
            batch_size=int(enh_cfg.BATCH_SIZE),
            device=str(cfg.MODEL.DEVICE),
        )

        self.fm_prototypes: Dict[int, torch.Tensor] = {}
        if self.fm_extractor.available:
            self._build_augmented_prototypes()
        else:
            logger.warning(
                "PCB-FMA-Enhanced encoder unavailable; falling back to the base PCB path."
            )

    def _augment_crop(self, crop: np.ndarray) -> List[np.ndarray]:
        views = [crop]

        if self.enable_flip:
            views.append(cv2.flip(crop, 1))

        if self.enable_multicrop:
            height, width = crop.shape[:2]
            for scale in self.multicrop_scales:
                crop_h, crop_w = int(height * scale), int(width * scale)
                if crop_h < 2 or crop_w < 2:
                    continue
                y0 = (height - crop_h) // 2
                x0 = (width - crop_w) // 2
                views.append(crop[y0:y0 + crop_h, x0:x0 + crop_w])
                if self.multicrop_num > 1:
                    views.append(crop[0:crop_h, 0:crop_w])
                if self.multicrop_num > 2:
                    views.append(crop[max(0, height - crop_h):height, max(0, width - crop_w):width])

        return views

    def _extract_augmented_features_for_box(
        self, img: np.ndarray, box: np.ndarray
    ) -> Optional[torch.Tensor]:
        img_h, img_w = img.shape[:2]
        x1 = max(0, int(box[0]))
        y1 = max(0, int(box[1]))
        x2 = min(img_w, int(box[2]))
        y2 = min(img_h, int(box[3]))
        if x2 <= x1 or y2 <= y1:
            crop = img[y1:y1 + 1, x1:x1 + 1]
        else:
            crop = img[y1:y2, x1:x2]

        views = self._augment_crop(crop)
        batch_tensor = self.fm_extractor._preprocess_crops(views)
        with torch.no_grad():
            feats = self.fm_extractor.model(batch_tensor)
        return feats.mean(dim=0)

    def _build_augmented_prototypes(self):
        logger.info("PCB-FMA-Enhanced: building support prototypes.")
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
            boxes_np = gt_boxes.cpu().numpy()

            for box_index, cls_tensor in enumerate(labels):
                aug_feat = self._extract_augmented_features_for_box(img, boxes_np[box_index])
                if aug_feat is None:
                    continue
                cls = int(cls_tensor.item())
                class_features.setdefault(cls, []).append(aug_feat.detach().cpu())

        for cls, feats in class_features.items():
            proto = torch.stack(feats, dim=0).mean(dim=0)
            self.fm_prototypes[cls] = F.normalize(proto, dim=0)

        logger.info(
            "PCB-FMA-Enhanced: built prototypes for %d classes.",
            len(self.fm_prototypes),
        )

    def _compute_competitive_similarity(self, query_fm: torch.Tensor, cls: int) -> float:
        if cls not in self.fm_prototypes:
            return 0.0

        class_ids = sorted(self.fm_prototypes.keys())
        if len(class_ids) <= 1:
            proto = self.fm_prototypes[cls].to(query_fm.device)
            sim = float(torch.dot(query_fm, proto).item())
            return (sim + 1.0) / 2.0

        sims = []
        target_index = -1
        for index, proto_cls in enumerate(class_ids):
            proto = self.fm_prototypes[proto_cls].to(query_fm.device)
            sims.append(torch.dot(query_fm, proto))
            if proto_cls == cls:
                target_index = index
        if target_index < 0:
            return 0.0

        sims_tensor = torch.stack(sims, dim=0)
        if self.competitive_mode == "softmax":
            temp = max(self.temperature, 1e-6)
            return float(F.softmax(sims_tensor / temp, dim=0)[target_index].item())
        return (float(sims_tensor[target_index].item()) + 1.0) / 2.0

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
                fm_sim = self._compute_competitive_similarity(query_fm, cls)

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


def build_pcb_fma_enhanced(base_pcb, cfg):
    """Factory for BMVC-v6 style PCB-FMA-Enhanced."""
    return PCBFMAEnhanced(base_pcb, cfg)
