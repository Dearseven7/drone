import numpy as np
from deep_sort.deep_sort import DeepSort

class Tracker:
    def __init__(self):
        self.deepsort = DeepSort("deep_sort/deep_sort/model_weights/ckpt.t7")

    def update_tracks(self, bbox_xywh, confidences, frame):
        if len(bbox_xywh) > 0:
            bbox_xywh = np.array(bbox_xywh)
            confidences = np.array(confidences)
        else:
            bbox_xywh = np.empty((0, 4))
            confidences = np.empty((0,))
        
        outputs = self.deepsort.update(bbox_xywh, confidences, frame)
        return outputs