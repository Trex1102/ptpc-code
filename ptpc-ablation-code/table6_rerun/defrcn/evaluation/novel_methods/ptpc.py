"""Deployed PTPC calibrator.

PTPC keeps the detector fixed.  It runs DINOv2 once on the full image,
pools contextual patch tokens inside each detector box with center weighting,
blends that RoI descriptor with a crop-level CLS descriptor, then applies a
base-confusability guard.  Detector boxes and labels are preserved; only scores
are recalibrated.
"""

import atexit
import json
import os

import cv2
import logging
import math
import re
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from detectron2.structures import Boxes, pairwise_iou
from defrcn.dataloader import build_detection_test_loader

logger = logging.getLogger(__name__)


class PTPCSupportContext:
    """Minimal PCB-like context for FM-only PTPC evaluation."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device(cfg.MODEL.DEVICE)
        self.pcb_upper = float(cfg.TEST.PCB_UPPER)
        self.pcb_lower = float(cfg.TEST.PCB_LOWER)
        self.dataloader = build_detection_test_loader(cfg, cfg.DATASETS.TRAIN[0])
        self.exclude_cls = self._excluded_class_ids()
        self.proto_to_eval_cls = self._build_proto_to_eval_class_ids()
        self.eval_to_proto_cls = {v: k for k, v in self.proto_to_eval_cls.items()}
        self.prototypes = {}
        self.enable_proto_reassign = False

    def _excluded_class_ids(self):
        dsname = self.cfg.DATASETS.TEST[0]
        if "test_all" not in dsname:
            return []
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

    def _build_proto_to_eval_class_ids(self):
        dsname = self.cfg.DATASETS.TEST[0]
        if "test_all" not in dsname:
            return {}

        try:
            from detectron2.data import MetadataCatalog

            train_classes = list(MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]).thing_classes)
            test_classes = list(MetadataCatalog.get(dsname).thing_classes)
        except Exception as exc:
            logger.warning("PTPC could not build test_all class remap for %s: %s", dsname, exc)
            return {}

        mapping = {}
        for proto_idx, cls_name in enumerate(train_classes):
            if cls_name not in test_classes:
                logger.warning("PTPC missing test_all class '%s' in %s", cls_name, dsname)
                continue
            mapping[proto_idx] = test_classes.index(cls_name)

        if mapping:
            logger.info("PTPC test_all class remap for %s: %s", dsname, mapping)
        return mapping

    def to_eval_class_id(self, cls: int) -> int:
        cls = int(cls)
        return int(self.proto_to_eval_cls.get(cls, cls))

    def to_proto_class_id(self, cls: int) -> int:
        cls = int(cls)
        return int(self.eval_to_proto_cls.get(cls, cls))

    def remap_instances_to_eval_space(self, instances):
        if not self.proto_to_eval_cls or len(instances) == 0:
            return

        pred_classes = instances.pred_classes
        remapped = pred_classes.clone()
        for proto_idx, eval_idx in self.proto_to_eval_cls.items():
            remapped[pred_classes == proto_idx] = int(eval_idx)
        instances.pred_classes = remapped

    def execute_calibration(self, inputs, dts, allow_reassign=True):
        return dts


def build_fm_only_support(cfg):
    return PTPCSupportContext(cfg)


class PTPCFeatureExtractor:
    """DINOv2 full-image patch-grid and crop-CLS feature extractor."""

    def __init__(
        self,
        model_name: str = "dinov2_vitb14",
        model_path: str = "",
        feat_dim: int = 768,
        image_size: int = 518,
        crop_size: int = 224,
        roi_size: Optional[int] = None,
        patch_size: int = 14,
        batch_size: int = 32,
        device: str = "cuda",
    ):
        self.model_name = model_name
        self.feat_dim = int(feat_dim)
        self.image_size = int(image_size)
        self.crop_size = int(roi_size if roi_size is not None else crop_size)
        self.patch_size = int(patch_size)
        self.batch_size = int(batch_size)
        self.device = torch.device(device)
        self.model = None
        self.norm_mean = [0.485, 0.456, 0.406]
        self.norm_std = [0.229, 0.224, 0.225]

        if self.image_size % self.patch_size != 0:
            self.image_size = int(math.ceil(self.image_size / self.patch_size) * self.patch_size)
            logger.warning("Rounded PTPC full-image size to %d", self.image_size)

        self._load_model(model_name, model_path)

    @staticmethod
    def _patch_dinov2_for_py38():
        import glob
        import os
        import sys

        if sys.version_info >= (3, 10):
            return
        for d in glob.glob(os.path.join(torch.hub.get_dir(), "facebookresearch_dinov2*")):
            for root, _, files in os.walk(d):
                for fname in files:
                    if not fname.endswith(".py"):
                        continue
                    fpath = os.path.join(root, fname)
                    try:
                        with open(fpath, "r") as f:
                            content = f.read()
                        if (
                            ("float | None" in content or "int | None" in content or "str | None" in content)
                            and "from __future__ import annotations" not in content
                        ):
                            with open(fpath, "w") as f:
                                f.write("from __future__ import annotations\n" + content)
                    except Exception:
                        pass

    def _load_model(self, model_name: str, model_path: str):
        try:
            if model_path:
                logger.info("Loading PTPC encoder from local path: %s", model_path)
                self.model = torch.load(model_path, map_location="cpu")
            else:
                logger.info("Loading PTPC encoder via torch.hub: %s", model_name)
                self._patch_dinov2_for_py38()
                try:
                    self.model = torch.hub.load("facebookresearch/dinov2", model_name, pretrained=True)
                except TypeError as exc:
                    if "unsupported operand type" not in str(exc):
                        raise
                    self._patch_dinov2_for_py38()
                    import sys as _sys
                    for key in [k for k in _sys.modules if "dinov2" in k]:
                        del _sys.modules[key]
                    self.model = torch.hub.load("facebookresearch/dinov2", model_name, pretrained=True)

            self.model = self.model.to(self.device)
            self.model.eval()
            for param in self.model.parameters():
                param.requires_grad = False

            with torch.no_grad():
                dummy = torch.randn(1, 3, self.image_size, self.image_size, device=self.device)
                patch_tokens, cls_token = self.model.get_intermediate_layers(
                    dummy, n=1, return_class_token=True,
                )[0]
                self.feat_dim = int(cls_token.shape[-1])
                self.grid_h = self.grid_w = int(round(math.sqrt(patch_tokens.shape[1])))
                if self.grid_h * self.grid_w != patch_tokens.shape[1]:
                    raise ValueError("DINOv2 patch token count is not square")

            logger.info(
                "Loaded PTPC encoder %s dim=%d grid=%dx%d",
                model_name, self.feat_dim, self.grid_h, self.grid_w,
            )
        except Exception as exc:
            logger.warning("Failed to load PTPC encoder '%s': %s", model_name, exc)
            self.model = None

    @property
    def available(self) -> bool:
        return self.model is not None

    def _normalizer(self):
        mean = torch.tensor(self.norm_mean).view(3, 1, 1)
        std = torch.tensor(self.norm_std).view(3, 1, 1)
        return mean, std

    def _preprocess_image(self, img: np.ndarray, size: int) -> torch.Tensor:
        mean, std = self._normalizer()
        resized = cv2.resize(img, (size, size))
        tensor = torch.from_numpy(resized[:, :, ::-1].copy()).permute(2, 0, 1).float() / 255.0
        return ((tensor - mean) / std).unsqueeze(0).to(self.device)

    def _preprocess_crops(self, crops: List[np.ndarray]) -> torch.Tensor:
        mean, std = self._normalizer()
        tensors = []
        for crop in crops:
            resized = cv2.resize(crop, (self.crop_size, self.crop_size))
            tensor = torch.from_numpy(resized[:, :, ::-1].copy()).permute(2, 0, 1).float() / 255.0
            tensors.append((tensor - mean) / std)
        return torch.stack(tensors, dim=0).to(self.device)

    def _crop_rois(self, img: np.ndarray, boxes_tensor: torch.Tensor) -> List[np.ndarray]:
        boxes = boxes_tensor.detach().cpu().numpy()
        img_h, img_w = img.shape[:2]
        crops = []
        for box in boxes:
            x1, y1, x2, y2 = box
            x1 = max(0, int(x1))
            y1 = max(0, int(y1))
            x2 = min(img_w, int(x2))
            y2 = min(img_h, int(y2))
            if x2 <= x1 or y2 <= y1:
                crops.append(img[max(0, y1):max(0, y1) + 1, max(0, x1):max(0, x1) + 1])
            else:
                crops.append(img[y1:y2, x1:x2])
        return crops

    def _full_image_tokens(self, img: np.ndarray) -> Optional[torch.Tensor]:
        if not self.available:
            return None
        with torch.no_grad():
            tensor = self._preprocess_image(img, self.image_size)
            patch_tokens, _ = self.model.get_intermediate_layers(
                tensor, n=1, return_class_token=True,
            )[0]
        grid_h = grid_w = int(round(math.sqrt(patch_tokens.shape[1])))
        return patch_tokens[0].view(grid_h, grid_w, -1)

    def _scaled_boxes(self, img: np.ndarray, boxes_tensor: torch.Tensor) -> torch.Tensor:
        img_h, img_w = img.shape[:2]
        boxes = boxes_tensor.detach().to(self.device).float().clone()
        boxes[:, [0, 2]] *= float(self.image_size) / max(float(img_w), 1.0)
        boxes[:, [1, 3]] *= float(self.image_size) / max(float(img_h), 1.0)
        boxes[:, 0::2].clamp_(0.0, float(self.image_size - 1))
        boxes[:, 1::2].clamp_(0.0, float(self.image_size - 1))
        return boxes

    def _center_weighted_pool(self, patch_grid: torch.Tensor, scaled_boxes: torch.Tensor) -> torch.Tensor:
        grid_h, grid_w, feat_dim = patch_grid.shape
        flat = patch_grid.view(grid_h * grid_w, feat_dim)
        ys = (torch.arange(grid_h, device=self.device).float() + 0.5) * (self.image_size / float(grid_h))
        xs = (torch.arange(grid_w, device=self.device).float() + 0.5) * (self.image_size / float(grid_w))
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        centers = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=1)

        pooled = []
        for box in scaled_boxes:
            x1, y1, x2, y2 = box.tolist()
            if x2 < x1:
                x1, x2 = x2, x1
            if y2 < y1:
                y1, y2 = y2, y1
            mask = (
                (centers[:, 0] >= x1) & (centers[:, 0] <= x2) &
                (centers[:, 1] >= y1) & (centers[:, 1] <= y2)
            )
            if not bool(mask.any()):
                cx = 0.5 * (x1 + x2)
                cy = 0.5 * (y1 + y2)
                dist = (centers[:, 0] - cx).pow(2) + (centers[:, 1] - cy).pow(2)
                mask[dist.argmin()] = True

            tokens = F.normalize(flat[mask], dim=1)
            token_centers = centers[mask]
            cx = 0.5 * (x1 + x2)
            cy = 0.5 * (y1 + y2)
            sx = max((x2 - x1) * 0.5, self.image_size / float(grid_w))
            sy = max((y2 - y1) * 0.5, self.image_size / float(grid_h))
            dist = ((token_centers[:, 0] - cx) / sx).pow(2) + ((token_centers[:, 1] - cy) / sy).pow(2)
            weights = torch.exp(-0.5 * dist)
            weights = weights / weights.sum().clamp_min(1e-6)
            vec = (tokens * weights[:, None]).sum(dim=0)
            pooled.append(F.normalize(vec, dim=0))
        return torch.stack(pooled, dim=0)

    def extract_patch_roi_features(self, img: np.ndarray, boxes_tensor: torch.Tensor) -> Optional[torch.Tensor]:
        if not self.available:
            return None
        if boxes_tensor.numel() == 0:
            return torch.empty((0, self.feat_dim), device=self.device)
        patch_grid = self._full_image_tokens(img)
        if patch_grid is None:
            return None
        return self._center_weighted_pool(patch_grid.to(self.device), self._scaled_boxes(img, boxes_tensor))

    def extract_crop_cls_features(self, img: np.ndarray, boxes_tensor: torch.Tensor) -> Optional[torch.Tensor]:
        if not self.available:
            return None
        if boxes_tensor.numel() == 0:
            return torch.empty((0, self.feat_dim), device=self.device)
        crops = self._crop_rois(img, boxes_tensor)
        all_cls = []
        with torch.no_grad():
            for start in range(0, len(crops), self.batch_size):
                batch = self._preprocess_crops(crops[start:start + self.batch_size])
                _, cls_token = self.model.get_intermediate_layers(
                    batch, n=1, return_class_token=True,
                )[0]
                all_cls.append(cls_token)
        return torch.cat(all_cls, dim=0)

    def extract_roi_features(self, img: np.ndarray, boxes_tensor: torch.Tensor) -> Optional[torch.Tensor]:
        """Compatibility path for analysis tools: return crop-CLS RoI features."""
        return self.extract_crop_cls_features(img, boxes_tensor)


FoundationModelFeatureExtractor = PTPCFeatureExtractor


class PTPCCalibrator:
    """Full-image hybrid center-weighted PTPC score calibrator."""

    def __init__(self, base_pcb, cfg):
        self.base_pcb = base_pcb
        self.cfg = cfg
        proto_cfg = cfg.NOVEL_METHODS.PTPC

        self.det_weight = float(proto_cfg.DET_WEIGHT)
        self.fm_weight = float(proto_cfg.FM_WEIGHT)
        self.crop_cls_weight = float(proto_cfg.CROP_CLS_WEIGHT)
        self.temperature = float(proto_cfg.TEMPERATURE)
        self.competitive_mode = str(proto_cfg.COMPETITIVE_MODE).lower()
        total = self.det_weight + self.fm_weight
        if total > 0:
            self.det_weight /= total
            self.fm_weight /= total

        self.fm_extractor = PTPCFeatureExtractor(
            model_name=str(proto_cfg.FM_MODEL_NAME),
            model_path=str(proto_cfg.FM_MODEL_PATH),
            feat_dim=int(proto_cfg.FM_FEAT_DIM),
            image_size=int(proto_cfg.IMAGE_SIZE),
            crop_size=int(proto_cfg.CROP_SIZE),
            patch_size=int(proto_cfg.PATCH_SIZE),
            batch_size=int(proto_cfg.BATCH_SIZE),
            device=str(cfg.MODEL.DEVICE),
        )

        self.patch_prototypes: Dict[int, torch.Tensor] = {}
        self.cls_prototypes: Dict[int, torch.Tensor] = {}
        if self.fm_extractor.available:
            self._build_prototypes()
        else:
            logger.warning("PTPC encoder unavailable; calibration will be skipped.")

    def _build_prototypes(self):
        logger.info("Building PTPC novel support prototypes.")
        class_patch: Dict[int, List[torch.Tensor]] = {}
        class_cls: Dict[int, List[torch.Tensor]] = {}
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

            patch_feats = self.fm_extractor.extract_patch_roi_features(img, gt_boxes)
            cls_feats = self.fm_extractor.extract_crop_cls_features(img, gt_boxes)
            if patch_feats is None or cls_feats is None:
                continue
            patch_feats = patch_feats.detach().cpu()
            cls_feats = cls_feats.detach().cpu()

            for i in range(len(labels)):
                cls = int(labels[i].item())
                class_patch.setdefault(cls, []).append(patch_feats[i])
                class_cls.setdefault(cls, []).append(cls_feats[i])

        for cls, feats in class_patch.items():
            self.patch_prototypes[cls] = F.normalize(torch.stack(feats, dim=0).mean(dim=0), dim=0)
        for cls, feats in class_cls.items():
            self.cls_prototypes[cls] = F.normalize(torch.stack(feats, dim=0).mean(dim=0), dim=0)

        logger.info(
            "Built PTPC prototypes for %d patch classes and %d CLS classes.",
            len(self.patch_prototypes), len(self.cls_prototypes),
        )

    def _competitive_score(self, query_feat: torch.Tensor, prototypes: Dict[int, torch.Tensor], cls: int) -> float:
        if cls not in prototypes:
            return 0.0
        cls_ids = sorted(prototypes.keys())
        if len(cls_ids) <= 1:
            sim = float(torch.dot(query_feat, prototypes[cls].to(query_feat.device)).item())
            return (sim + 1.0) / 2.0

        sims = []
        target_idx = -1
        for idx, proto_cls in enumerate(cls_ids):
            sims.append(torch.dot(query_feat, prototypes[proto_cls].to(query_feat.device)))
            if proto_cls == cls:
                target_idx = idx
        if target_idx < 0:
            return 0.0

        sims_tensor = torch.stack(sims, dim=0)
        if self.competitive_mode == "softmax":
            scores = F.softmax(sims_tensor / max(self.temperature, 1e-6), dim=0)
            return float(scores[target_idx].item())
        return (float(sims_tensor[target_idx].item()) + 1.0) / 2.0

    def _hybrid_score(self, patch_feat, cls_feat, cls: int) -> float:
        patch_score = 0.0
        if patch_feat is not None and cls in self.patch_prototypes:
            patch_score = self._competitive_score(F.normalize(patch_feat, dim=0), self.patch_prototypes, cls)

        cls_score = 0.0
        if cls_feat is not None and cls in self.cls_prototypes:
            cls_score = self._competitive_score(F.normalize(cls_feat, dim=0), self.cls_prototypes, cls)

        w = min(max(self.crop_cls_weight, 0.0), 1.0)
        return (1.0 - w) * patch_score + w * cls_score

    def execute_calibration(self, inputs, dts):
        if not self.fm_extractor.available or not self.patch_prototypes:
            return dts
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

        box_tensor = pred_boxes.tensor
        patch_feats = self.fm_extractor.extract_patch_roi_features(img, box_tensor)
        cls_feats = self.fm_extractor.extract_crop_cls_features(img, box_tensor)
        pred_classes = instances.pred_classes
        score_device = scores.device
        score_dtype = scores.dtype

        for i in range(ileft, iright):
            cls = int(pred_classes[i].item())
            eval_cls = self.base_pcb.to_eval_class_id(cls)
            proto_cls = self.base_pcb.to_proto_class_id(cls)
            if eval_cls in self.base_pcb.exclude_cls:
                continue
            q_idx = i - ileft
            fm_sim = self._hybrid_score(
                patch_feats[q_idx] if patch_feats is not None else None,
                cls_feats[q_idx] if cls_feats is not None else None,
                proto_cls,
            )
            det_score = float(scores[i].item())
            fused = self.det_weight * det_score + self.fm_weight * fm_sim
            scores[i] = torch.tensor(max(0.0, min(1.0, fused)), device=score_device, dtype=score_dtype)
        self.base_pcb.remap_instances_to_eval_space(instances)
        return dts

    def __getattr__(self, name):
        return getattr(self.base_pcb, name)


class PTPCGuard:
    """Base-confusability score suppression for PTPC."""

    DIAG_BUCKETS = (
        "base_gt_overlap",
        "novel_gt_overlap",
        "background_no_gt_overlap",
    )

    def __init__(self, inner_calibrator, cfg, fm_extractor=None):
        self.inner_calibrator = inner_calibrator
        self.cfg = cfg
        guard_cfg = cfg.NOVEL_METHODS.PTPC_GUARD
        self.margin = float(guard_cfg.MARGIN)
        self.suppression_factor = float(guard_cfg.SUPPRESSION_FACTOR)
        self.max_base_samples = int(guard_cfg.MAX_BASE_SAMPLES_PER_CLASS)
        self.base_pcb_ref = self._get_base_pcb()
        self.fm_extractor = fm_extractor if fm_extractor is not None else inner_calibrator.fm_extractor

        self.base_prototypes: Dict[int, torch.Tensor] = {}
        self.novel_prototypes: Dict[int, torch.Tensor] = {}
        if self.fm_extractor.available:
            self._build_base_prototypes()
            self.novel_prototypes = self._resolve_novel_prototypes()
        self.base_class_ids = set(getattr(self.base_pcb_ref, "exclude_cls", []))
        self._init_diagnostics()

    def _get_base_pcb(self):
        pcb = self.inner_calibrator
        while hasattr(pcb, "inner_calibrator"):
            pcb = pcb.inner_calibrator
        while hasattr(pcb, "base_pcb"):
            pcb = pcb.base_pcb
        return pcb

    def _derive_base_dataset_name(self) -> str:
        train_name = self.cfg.DATASETS.TRAIN[0]
        match = re.match(r"(voc_\d+_trainval)_novel(\d+)_.*", train_name)
        if match:
            return f"{match.group(1)}_base{match.group(2)}"
        match = re.match(r"(coco14?_trainval)_novel_.*", train_name)
        if match:
            return f"{match.group(1)}_base"
        return ""

    def _build_base_prototypes(self):
        base_ds_name = self._derive_base_dataset_name()
        if not base_ds_name:
            logger.warning("PTPC guard could not derive base dataset from %s", self.cfg.DATASETS.TRAIN[0])
            return

        try:
            from detectron2.data import MetadataCatalog
            base_loader = build_detection_test_loader(self.cfg, base_ds_name)
            base_set = set(range(len(MetadataCatalog.get(base_ds_name).thing_classes)))
        except Exception as exc:
            logger.warning("PTPC guard could not load base dataset %s: %s", base_ds_name, exc)
            return

        class_features: Dict[int, List[torch.Tensor]] = {}
        class_counts: Dict[int, int] = {}
        for index in range(len(base_loader.dataset)):
            if all(class_counts.get(cls, 0) >= self.max_base_samples for cls in base_set):
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
            boxes = inst.gt_boxes.tensor.clone() * ratio
            labels = inst.gt_classes.cpu()
            keep = torch.zeros(len(labels), dtype=torch.bool)
            for idx in range(len(labels)):
                cls = int(labels[idx].item())
                if cls in base_set and class_counts.get(cls, 0) < self.max_base_samples:
                    keep[idx] = True
            if not keep.any():
                continue

            feats = self.fm_extractor.extract_crop_cls_features(img, boxes[keep])
            if feats is None:
                continue
            feats = feats.detach().cpu()
            for idx, cls_tensor in enumerate(labels[keep]):
                cls = int(cls_tensor.item())
                class_features.setdefault(cls, []).append(feats[idx])
                class_counts[cls] = class_counts.get(cls, 0) + 1

        for cls, feats in class_features.items():
            self.base_prototypes[cls] = F.normalize(torch.stack(feats, dim=0).mean(dim=0), dim=0)
        logger.info("Built PTPC guard prototypes for %d base classes.", len(self.base_prototypes))

    def _resolve_novel_prototypes(self) -> Dict[int, torch.Tensor]:
        if hasattr(self.inner_calibrator, "cls_prototypes") and self.inner_calibrator.cls_prototypes:
            return self.inner_calibrator.cls_prototypes
        return {}

    def _init_diagnostics(self):
        base_path = os.environ.get("PTPC_GUARD_DIAG_SUMMARY_PATH", "").strip()
        self.diag_enabled = bool(base_path)
        self.diag_summary_path = ""
        if self.diag_enabled:
            stem, ext = os.path.splitext(base_path)
            ext = ext or ".json"
            self.diag_summary_path = f"{stem}.pid{os.getpid()}{ext}"
        self._diag_written = False
        self.diag_state = {
            "checked_total": 0,
            "suppressed_total": 0,
            "buckets": {
                name: {"checked": 0, "suppressed": 0, "delta_sum": 0.0}
                for name in self.DIAG_BUCKETS
            },
        }
        if self.diag_enabled:
            logger.info("PTPC guard diagnostics enabled: %s", self.diag_summary_path)
            atexit.register(self._finalize_diag_summary)

    def _original_gt(self, inputs):
        gt_instances = inputs[0].get("instances")
        if gt_instances is None or len(gt_instances) == 0:
            return None, None

        gt_boxes = gt_instances.gt_boxes.tensor.detach().cpu().clone()
        gt_classes = gt_instances.gt_classes.detach().cpu().clone()
        gt_h, gt_w = gt_instances.image_size
        orig_h = int(inputs[0].get("height", gt_h))
        orig_w = int(inputs[0].get("width", gt_w))

        if gt_h > 0 and gt_w > 0 and (gt_h != orig_h or gt_w != orig_w):
            gt_boxes[:, 0::2] *= float(orig_w) / float(gt_w)
            gt_boxes[:, 1::2] *= float(orig_h) / float(gt_h)
        return gt_boxes, gt_classes

    def _bucketize_checked_boxes(self, inputs, check_boxes):
        if not self.diag_enabled:
            return None

        gt_boxes, gt_classes = self._original_gt(inputs)
        if gt_boxes is None or gt_classes is None or gt_boxes.numel() == 0:
            return ["background_no_gt_overlap"] * len(check_boxes)

        overlaps = pairwise_iou(
            Boxes(check_boxes.tensor.detach().cpu()),
            Boxes(gt_boxes),
        )
        max_ious, max_indices = overlaps.max(dim=1)

        buckets = []
        for det_idx in range(len(check_boxes)):
            if float(max_ious[det_idx].item()) < 0.5:
                buckets.append("background_no_gt_overlap")
                continue

            gt_cls = int(gt_classes[int(max_indices[det_idx].item())].item())
            if gt_cls in self.base_class_ids:
                buckets.append("base_gt_overlap")
            else:
                buckets.append("novel_gt_overlap")
        return buckets

    def _record_diag(self, bucket, suppressed, delta):
        if not self.diag_enabled:
            return

        stats = self.diag_state["buckets"][bucket]
        stats["checked"] += 1
        stats["suppressed"] += int(suppressed)
        stats["delta_sum"] += float(delta)
        self.diag_state["checked_total"] += 1
        self.diag_state["suppressed_total"] += int(suppressed)

    def _write_diag_summary(self, final=False):
        if not self.diag_enabled:
            return
        if final and self._diag_written:
            return

        payload = {
            "dataset_test": self.cfg.DATASETS.TEST[0],
            "dataset_train": self.cfg.DATASETS.TRAIN[0],
            "output_dir": self.cfg.OUTPUT_DIR,
            "fm_model_name": self.cfg.NOVEL_METHODS.PTPC.FM_MODEL_NAME,
            "crop_cls_weight": float(self.cfg.NOVEL_METHODS.PTPC.CROP_CLS_WEIGHT),
            "guard_margin": self.margin,
            "suppression_factor": self.suppression_factor,
            "pid": int(os.getpid()),
            "score_band": {
                "upper": float(self.base_pcb_ref.pcb_upper),
                "lower": float(self.base_pcb_ref.pcb_lower),
            },
            "checked_total": int(self.diag_state["checked_total"]),
            "suppressed_total": int(self.diag_state["suppressed_total"]),
            "buckets": {},
        }

        for bucket_name, stats in self.diag_state["buckets"].items():
            checked = int(stats["checked"])
            suppressed = int(stats["suppressed"])
            payload["buckets"][bucket_name] = {
                "checked": checked,
                "suppressed": suppressed,
                "suppression_rate": (suppressed / checked) if checked else 0.0,
                "mean_delta_s": (float(stats["delta_sum"]) / checked) if checked else 0.0,
            }

        os.makedirs(os.path.dirname(self.diag_summary_path), exist_ok=True)
        with open(self.diag_summary_path, "w") as handle:
            json.dump(payload, handle, indent=2)
        if final:
            self._diag_written = True

    def _finalize_diag_summary(self):
        self._write_diag_summary(final=True)

    def execute_calibration(self, inputs, dts):
        dts = self.inner_calibrator.execute_calibration(inputs, dts)
        if not self.fm_extractor.available or not self.base_prototypes or not self.novel_prototypes:
            return dts
        if len(dts) == 0 or len(dts[0]["instances"]) == 0:
            return dts

        img = cv2.imread(inputs[0]["file_name"])
        if img is None:
            return dts

        instances = dts[0]["instances"]
        scores = instances.scores
        pred_classes = instances.pred_classes
        pred_boxes = instances.pred_boxes
        base_pcb = self.base_pcb_ref
        ileft = int((scores > base_pcb.pcb_upper).sum().item())
        iright = int((scores > base_pcb.pcb_lower).sum().item())
        if ileft >= iright:
            return dts

        check_boxes = pred_boxes[ileft:iright]
        if len(check_boxes) == 0:
            return dts
        feats = self.fm_extractor.extract_crop_cls_features(img, check_boxes.tensor)
        if feats is None:
            return dts
        diag_buckets = self._bucketize_checked_boxes(inputs, check_boxes)

        dev = self.fm_extractor.device
        score_device = scores.device
        score_dtype = scores.dtype
        for i in range(ileft, iright):
            cls = int(pred_classes[i].item())
            eval_cls = base_pcb.to_eval_class_id(cls)
            if eval_cls in base_pcb.exclude_cls:
                continue

            q_feat = F.normalize(feats[i - ileft].to(dev), dim=0)
            max_novel = max(float(torch.dot(q_feat, proto.to(dev)).item()) for proto in self.novel_prototypes.values())
            max_base = max(float(torch.dot(q_feat, proto.to(dev)).item()) for proto in self.base_prototypes.values())
            suppressed = False
            delta = 0.0
            if max_base > max_novel + self.margin:
                old_score = float(scores[i].item())
                new_score = old_score * self.suppression_factor
                scores[i] = torch.tensor(new_score, device=score_device, dtype=score_dtype)
                suppressed = True
                delta = new_score - old_score
            if diag_buckets is not None:
                self._record_diag(diag_buckets[i - ileft], suppressed, delta)
        return dts

    def __getattr__(self, name):
        return getattr(self.inner_calibrator, name)


def build_ptpc(base_pcb, cfg):
    """Build the deployed PTPC method."""
    if base_pcb is None:
        base_pcb = build_fm_only_support(cfg)
    calibrator = PTPCCalibrator(base_pcb, cfg)
    if bool(cfg.NOVEL_METHODS.PTPC_GUARD.ENABLE):
        return PTPCGuard(calibrator, cfg, fm_extractor=calibrator.fm_extractor)
    return calibrator
