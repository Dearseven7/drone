import torch
import sys
import cv2
import numpy as np
from pathlib import Path

# 添加本地 yolov7 模型目录（路径根据实际情况调整）
sys.path.append(str(Path(__file__).resolve().parent / "yolov7"))

from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_coords
from utils.torch_utils import select_device

class YoloDetector:
    def __init__(self, weights_path, device='cpu', img_size=640, conf_thresh=0.25, iou_thresh=0.45):
        self.device = select_device(device)
        self.model = attempt_load(weights_path, map_location=self.device)
        self.model.eval()

        self.img_size = img_size
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names

    def detect(self, frame):
        img = cv2.resize(frame, (self.img_size, self.img_size))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_tensor = torch.from_numpy(img_rgb).to(self.device).float()
        img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0) / 255.0

        with torch.no_grad():
            pred = self.model(img_tensor)[0]
            detections = non_max_suppression(pred, self.conf_thresh, self.iou_thresh)[0]

        results = []
        if detections is not None and len(detections):
            detections[:, :4] = scale_coords(img_tensor.shape[2:], detections[:, :4], frame.shape).round()
            for *xyxy, conf, cls in detections:
                results.append({
                    "box": [int(x.item()) for x in xyxy],
                    "conf": float(conf.item()),
                    "cls": int(cls.item()),
                    "label": self.names[int(cls.item())]
                })
        return results