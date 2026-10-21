import cv2
import torch
import logging
from detectron2.structures import Boxes, ImageList
from detectron2.modeling.poolers import ROIPooler

from defrcn.dataloader import build_detection_test_loader
from defrcn.evaluation.archs import resnet101

logger = logging.getLogger(__name__)


class PrototypicalCalibrationBlock:
    """Vanilla PCB baseline retained for the PTPC submission."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device(cfg.MODEL.DEVICE)

        self.alpha = float(cfg.TEST.PCB_ALPHA)
        self.pcb_upper = float(cfg.TEST.PCB_UPPER)
        self.pcb_lower = float(cfg.TEST.PCB_LOWER)

        self.enable_proto_reassign = bool(cfg.TEST.PCB_PROTO_REASSIGN)
        self.proto_reassign_min_score = float(cfg.TEST.PCB_PROTO_REASSIGN_MIN_SCORE)
        self.proto_reassign_max_score = float(cfg.TEST.PCB_PROTO_REASSIGN_MAX_SCORE)
        self.proto_reassign_min_sim = float(cfg.TEST.PCB_PROTO_REASSIGN_MIN_SIM)
        self.proto_reassign_margin = float(cfg.TEST.PCB_PROTO_REASSIGN_MARGIN)

        self.imagenet_model = self.build_model()
        self.dataloader = build_detection_test_loader(cfg, cfg.DATASETS.TRAIN[0])
        self.roi_pooler = ROIPooler(
            output_size=(1, 1),
            scales=(1 / 32,),
            sampling_ratio=0,
            pooler_type="ROIAlignV2",
        )
        self.prototypes = self.build_prototypes()
        self.exclude_cls = self.clsid_filter()

    def build_model(self):
        logger.info("Loading PCB ImageNet backbone from %s", self.cfg.TEST.PCB_MODELPATH)
        if self.cfg.TEST.PCB_MODELTYPE != "resnet":
            raise NotImplementedError("Only TEST.PCB_MODELTYPE='resnet' is supported.")
        imagenet_model = resnet101()
        state_dict = torch.load(self.cfg.TEST.PCB_MODELPATH)
        imagenet_model.load_state_dict(state_dict)
        imagenet_model = imagenet_model.to(self.device)
        imagenet_model.eval()
        return imagenet_model

    def _normalized_area(self, boxes, img_h, img_w):
        widths = (boxes[:, 2] - boxes[:, 0]).clamp(min=1.0)
        heights = (boxes[:, 3] - boxes[:, 1]).clamp(min=1.0)
        return (widths * heights) / max(float(img_h * img_w), 1.0)

    def build_prototypes(self):
        class_features = {}

        for index in range(len(self.dataloader.dataset)):
            inputs = [self.dataloader.dataset[index]]
            img = cv2.imread(inputs[0]["file_name"])
            if img is None:
                continue

            img_h, img_w = img.shape[:2]
            inst = inputs[0]["instances"]
            if len(inst) == 0:
                continue

            ratio = img_h / float(inst.image_size[0])
            gt_boxes = inst.gt_boxes.tensor.clone() * ratio
            boxes = [Boxes(gt_boxes).to(self.device)]
            features = self.extract_roi_features(img, boxes).detach().cpu()
            labels = inst.gt_classes.cpu()

            for i in range(int(labels.shape[0])):
                cls = int(labels[i].item())
                class_features.setdefault(cls, []).append(features[i])

        prototypes = {}
        for cls, feats in class_features.items():
            proto = torch.mean(torch.stack(feats, dim=0), dim=0)
            prototypes[cls] = {
                "global": {
                    "protos": proto.unsqueeze(0),
                    "weights": torch.tensor([1.0], dtype=proto.dtype),
                }
            }

        logger.info("Built vanilla PCB prototypes for %d classes.", len(prototypes))
        return prototypes

    def extract_roi_features(self, img, boxes):
        mean = torch.tensor([0.406, 0.456, 0.485], dtype=torch.float32).reshape((3, 1, 1)).to(self.device)
        std = torch.tensor([0.225, 0.224, 0.229], dtype=torch.float32).reshape((3, 1, 1)).to(self.device)

        img = img.transpose((2, 0, 1))
        img = torch.from_numpy(img).to(self.device)
        images = [(img / 255.0 - mean) / std]
        images = ImageList.from_tensors(images, 0)
        conv_feature = self.imagenet_model(images.tensor[:, [2, 1, 0]])[1]

        box_features = self.roi_pooler([conv_feature], boxes).squeeze(2).squeeze(2)
        return self.imagenet_model.fc(box_features)

    def _cosine_to_protos(self, feature_vectors, proto_vectors):
        fv = torch.nn.functional.normalize(feature_vectors, dim=1)
        pv = torch.nn.functional.normalize(proto_vectors, dim=1)
        return torch.mm(fv, pv.t())

    def _select_proto_bank(self, cls, area_norm):
        del area_norm
        class_entry = self.prototypes.get(cls)
        if class_entry is None:
            return None
        return class_entry["global"]

    def _match_similarity(self, query_feature, proto_bank):
        protos = proto_bank["protos"].to(query_feature.device)
        if int(protos.shape[0]) == 0:
            return 0.0
        sims = self._cosine_to_protos(query_feature.unsqueeze(0), protos).squeeze(0)
        return float(sims[0].item())

    def _best_proto_match(self, query_feature, area_norm):
        best_cls = None
        best_sim = -1.0
        for cand_cls in sorted(self.prototypes.keys()):
            if cand_cls in self.exclude_cls:
                continue
            proto_bank = self._select_proto_bank(cand_cls, area_norm)
            if proto_bank is None:
                continue
            sim = self._match_similarity(query_feature, proto_bank)
            if best_cls is None or sim > best_sim:
                best_cls = int(cand_cls)
                best_sim = float(sim)
        return best_cls, best_sim

    def execute_calibration(self, inputs, dts, allow_reassign=True):
        if len(dts) == 0 or len(dts[0]["instances"]) == 0:
            return dts

        img = cv2.imread(inputs[0]["file_name"])
        if img is None:
            return dts
        img_h, img_w = img.shape[:2]

        scores = dts[0]["instances"].scores
        ileft = int((scores > self.pcb_upper).sum().item())
        iright = int((scores > self.pcb_lower).sum().item())
        if ileft >= iright:
            return dts

        pred_boxes = dts[0]["instances"].pred_boxes[ileft:iright]
        if len(pred_boxes) == 0:
            return dts

        features = self.extract_roi_features(img, [pred_boxes.to(self.device)])
        pred_classes = dts[0]["instances"].pred_classes
        score_device = scores.device
        score_dtype = scores.dtype
        area_norm = self._normalized_area(pred_boxes.tensor, img_h, img_w)

        for i in range(ileft, iright):
            cls = int(pred_classes[i].item())
            if cls in self.exclude_cls or cls not in self.prototypes:
                continue

            q_idx = i - ileft
            query_feature = features[q_idx]
            proto_bank = self._select_proto_bank(cls, float(area_norm[q_idx].item()))
            if proto_bank is None:
                continue

            sim = self._match_similarity(query_feature, proto_bank)
            det_score = float(scores[i].item())

            if allow_reassign and self.enable_proto_reassign:
                if self.proto_reassign_min_score <= det_score <= self.proto_reassign_max_score:
                    best_cls, best_sim = self._best_proto_match(
                        query_feature,
                        float(area_norm[q_idx].item()),
                    )
                    if (
                        best_cls is not None
                        and best_cls != cls
                        and best_sim >= self.proto_reassign_min_sim
                        and best_sim >= sim + self.proto_reassign_margin
                    ):
                        cls = int(best_cls)
                        pred_classes[i] = cls
                        sim = float(best_sim)

            fused = det_score * self.alpha + sim * (1.0 - self.alpha)
            scores[i] = torch.tensor(fused, device=score_device, dtype=score_dtype)

        return dts

    def clsid_filter(self):
        dsname = self.cfg.DATASETS.TEST[0]
        if "test_all" in dsname:
            if "coco" in dsname:
                return [
                    7, 9, 10, 11, 12, 13, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
                    30, 31, 32, 33, 34, 35, 36, 37, 38, 40, 41, 42, 43, 44, 45,
                    46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 59, 61, 63, 64, 65,
                    66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79,
                ]
            if "voc" in dsname:
                return list(range(0, 15))
            raise NotImplementedError
        return []
