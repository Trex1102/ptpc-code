"""R101 crop-CLS competitive and guarded calibration wrappers.

These wrappers stay entirely in PCB's original ResNet-101 crop-feature space.
They are used for Table 1 rows that vary competition and guarding while keeping
the encoder/pooling fixed to the vanilla PCB branch.
"""

import cv2
import re
import logging
from typing import Dict, List

import torch
import torch.nn.functional as F
from detectron2.structures import Boxes


logger = logging.getLogger(__name__)


class PCBResNetCompetitive:
    """Competitive rescoring in vanilla PCB's crop-CLS feature space."""

    def __init__(self, base_pcb, cfg):
        self.base_pcb = base_pcb
        self.cfg = cfg
        enh_cfg = cfg.NOVEL_METHODS.PCB_FMA_ENHANCED
        self.temperature = max(float(enh_cfg.TEMPERATURE), 1e-6)

    def _competitive_probabilities(self, query_feature, area_norm: float) -> Dict[int, float]:
        class_ids: List[int] = []
        sims: List[float] = []

        for cls in sorted(self.base_pcb.prototypes.keys()):
            if cls in self.base_pcb.exclude_cls:
                continue
            proto_bank = self.base_pcb._select_proto_bank(cls, area_norm)
            if proto_bank is None:
                continue
            sim = self.base_pcb._match_similarity(query_feature, proto_bank)
            class_ids.append(int(cls))
            sims.append(float(sim))

        if not sims:
            return {}

        sim_tensor = torch.tensor(sims, device=query_feature.device, dtype=query_feature.dtype)
        probs = torch.softmax(sim_tensor / self.temperature, dim=0)
        return {cls: float(prob.item()) for cls, prob in zip(class_ids, probs)}

    def execute_calibration(self, inputs, dts, allow_reassign=True):
        del allow_reassign
        if len(dts) == 0 or len(dts[0]["instances"]) == 0:
            return dts

        img = cv2.imread(inputs[0]["file_name"])
        if img is None:
            return dts
        img_h, img_w = img.shape[0], img.shape[1]

        scores = dts[0]["instances"].scores
        ileft = int((scores > self.base_pcb.pcb_upper).sum().item())
        iright = int((scores > self.base_pcb.pcb_lower).sum().item())
        if ileft >= iright:
            return dts

        pred_boxes = dts[0]["instances"].pred_boxes[ileft:iright]
        if len(pred_boxes) == 0:
            return dts

        features = self.base_pcb.extract_roi_features(img, [pred_boxes.to(self.base_pcb.device)])
        pred_classes = dts[0]["instances"].pred_classes
        score_device = scores.device
        score_dtype = scores.dtype
        area_norm = self.base_pcb._normalized_area(pred_boxes.tensor, img_h, img_w)

        for i in range(ileft, iright):
            cls = int(pred_classes[i].item())
            if cls in self.base_pcb.exclude_cls or cls not in self.base_pcb.prototypes:
                continue

            q_idx = i - ileft
            probs = self._competitive_probabilities(features[q_idx], float(area_norm[q_idx].item()))
            if cls not in probs:
                continue

            old_score = float(scores[i].item())
            sim = float(probs[cls])
            alpha = self.base_pcb._effective_alpha(cls, sim)
            fused = old_score * alpha + sim * (1.0 - alpha)
            fused = self.base_pcb._normalize_score(cls, fused)
            scores[i] = torch.tensor(fused, device=score_device, dtype=score_dtype)

        return dts

    def __getattr__(self, name):
        return getattr(self.base_pcb, name)


class PCBResNetGuard:
    """Base-confusability guard in vanilla PCB's crop-CLS feature space."""

    def __init__(self, inner_pcb, cfg):
        self.inner_pcb = inner_pcb
        self.cfg = cfg
        npg_cfg = cfg.NOVEL_METHODS.NEG_PROTO_GUARD
        self.margin = float(npg_cfg.MARGIN)
        self.suppression_factor = float(npg_cfg.SUPPRESSION_FACTOR)
        self.max_base_samples = int(npg_cfg.MAX_BASE_SAMPLES_PER_CLASS)
        self.base_pcb_ref = self._get_base_pcb()
        self.base_cls_ids = self._get_base_class_ids()
        self.base_prototypes: Dict[int, torch.Tensor] = {}

        if self.base_cls_ids:
            self._build_base_prototypes()
        else:
            logger.warning("PCBResNetGuard: no base class IDs resolved, guard disabled.")

    def _get_base_pcb(self):
        pcb = self.inner_pcb
        while hasattr(pcb, "inner_pcb"):
            pcb = pcb.inner_pcb
        while hasattr(pcb, "base_pcb"):
            pcb = pcb.base_pcb
        return pcb

    def _get_base_class_ids(self) -> List[int]:
        pcb = self._get_base_pcb()
        if hasattr(pcb, "exclude_cls") and pcb.exclude_cls:
            return list(pcb.exclude_cls)

        dsname = self.cfg.DATASETS.TEST[0]
        if "voc" in dsname:
            return list(range(0, 15))
        if "coco" in dsname:
            return [
                7, 9, 10, 11, 12, 13, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
                30, 31, 32, 33, 34, 35, 36, 37, 38, 40, 41, 42, 43, 44, 45,
                46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 59, 61, 63, 64, 65,
                66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79,
            ]
        return []

    def _derive_base_dataset_name(self) -> str:
        novel_ds = self.cfg.DATASETS.TRAIN[0]
        match = re.match(r"(voc_\d+_trainval)_novel(\d+)_.*", novel_ds)
        if match:
            return f"{match.group(1)}_base{match.group(2)}"
        match = re.match(r"(coco_trainval)_novel_.*", novel_ds)
        if match:
            return f"{match.group(1)}_base"
        return ""

    def _build_base_prototypes(self):
        logger.info("PCBResNetGuard: Building base-class prototypes...")
        base_ds_name = self._derive_base_dataset_name()
        if not base_ds_name:
            logger.warning("PCBResNetGuard: Could not derive base dataset from '%s'.", self.cfg.DATASETS.TRAIN[0])
            return

        try:
            from defrcn.dataloader import build_detection_test_loader
            base_loader = build_detection_test_loader(self.cfg, base_ds_name)
        except Exception as exc:
            logger.warning("PCBResNetGuard: Could not load base dataset '%s': %s", base_ds_name, exc)
            return

        class_features: Dict[int, List[torch.Tensor]] = {}
        class_counts: Dict[int, int] = {}
        base_set = set(self.base_cls_ids)

        for index in range(len(base_loader.dataset)):
            if all(class_counts.get(c, 0) >= self.max_base_samples for c in base_set):
                break

            inputs = [base_loader.dataset[index]]
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

            keep = torch.zeros(len(labels), dtype=torch.bool)
            for j in range(len(labels)):
                cls = int(labels[j].item())
                if cls in base_set and class_counts.get(cls, 0) < self.max_base_samples:
                    keep[j] = True
            if not keep.any():
                continue

            keep_boxes = gt_boxes[keep]
            keep_labels = labels[keep]
            box_list = [Boxes(keep_boxes).to(self.base_pcb_ref.device)]
            feats = self.base_pcb_ref.extract_roi_features(img, box_list).detach().cpu()

            for j in range(len(keep_labels)):
                cls = int(keep_labels[j].item())
                class_features.setdefault(cls, []).append(feats[j])
                class_counts[cls] = class_counts.get(cls, 0) + 1

        for cls, feats in class_features.items():
            proto = torch.stack(feats, dim=0).mean(dim=0)
            self.base_prototypes[cls] = F.normalize(proto, dim=0)

        logger.info(
            "PCBResNetGuard: Built base prototypes for %d / %d base classes.",
            len(self.base_prototypes),
            len(self.base_cls_ids),
        )

    def _max_novel_similarity(self, query_feature, area_norm: float) -> float:
        best_sim = -1.0
        for cls in sorted(self.base_pcb_ref.prototypes.keys()):
            if cls in self.base_pcb_ref.exclude_cls:
                continue
            proto_bank = self.base_pcb_ref._select_proto_bank(cls, area_norm)
            if proto_bank is None:
                continue
            protos = proto_bank["protos"].to(query_feature.device)
            if int(protos.shape[0]) == 0:
                continue
            sims = self.base_pcb_ref._cosine_to_protos(query_feature.unsqueeze(0), protos).squeeze(0)
            best_sim = max(best_sim, float(torch.max(sims).item()))
        return best_sim

    def _max_base_similarity(self, query_feature) -> float:
        best_sim = -1.0
        for proto in self.base_prototypes.values():
            sim = float(torch.dot(F.normalize(query_feature, dim=0), proto.to(query_feature.device)).item())
            best_sim = max(best_sim, sim)
        return best_sim

    def execute_calibration(self, inputs, dts, allow_reassign=True):
        dts = self.inner_pcb.execute_calibration(inputs, dts, allow_reassign=allow_reassign)
        if not self.base_prototypes:
            return dts
        if len(dts) == 0 or len(dts[0]["instances"]) == 0:
            return dts

        img = cv2.imread(inputs[0]["file_name"])
        if img is None:
            return dts
        img_h, img_w = img.shape[:2]

        instances = dts[0]["instances"]
        scores = instances.scores
        pred_classes = instances.pred_classes
        pred_boxes = instances.pred_boxes

        ileft = int((scores > self.base_pcb_ref.pcb_upper).sum().item())
        iright = int((scores > self.base_pcb_ref.pcb_lower).sum().item())
        if ileft >= iright:
            return dts

        check_boxes = pred_boxes[ileft:iright]
        if len(check_boxes) == 0:
            return dts

        feats = self.base_pcb_ref.extract_roi_features(img, [check_boxes.to(self.base_pcb_ref.device)])
        area_norm = self.base_pcb_ref._normalized_area(check_boxes.tensor, img_h, img_w)
        score_device = scores.device
        score_dtype = scores.dtype

        for i in range(ileft, iright):
            cls = int(pred_classes[i].item())
            if cls in self.base_pcb_ref.exclude_cls:
                continue

            q_idx = i - ileft
            query_feature = feats[q_idx]
            max_novel_sim = self._max_novel_similarity(query_feature, float(area_norm[q_idx].item()))
            max_base_sim = self._max_base_similarity(query_feature)
            if max_base_sim > max_novel_sim + self.margin:
                scores[i] = torch.tensor(
                    float(scores[i].item()) * self.suppression_factor,
                    device=score_device,
                    dtype=score_dtype,
                )

        return dts

    def __getattr__(self, name):
        return getattr(self.inner_pcb, name)


def build_pcb_resnet101(base_pcb, cfg):
    del cfg
    return base_pcb


def build_pcb_resnet101_competitive(base_pcb, cfg):
    return PCBResNetCompetitive(base_pcb, cfg)


def build_pcb_resnet101_guard(base_pcb, cfg):
    return PCBResNetGuard(base_pcb, cfg)


def build_pcb_resnet101_competitive_guard(base_pcb, cfg):
    competitive = PCBResNetCompetitive(base_pcb, cfg)
    return PCBResNetGuard(competitive, cfg)
