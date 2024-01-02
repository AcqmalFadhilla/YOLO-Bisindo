from ultralytics import YOLO
import numpy as np

class YOLOSEG:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def detect(self, img):
        # get img shape
        # height, width, channel = img.shape
        
        results = self.model.predict(source=img.copy(), save=False, save_txt=False, conf=0.7)
        result = results[0]
        l = len(result)

        segmentation_contours_idx = []
        if len(result) > 0:
            for seg in result.masks.xy:
                # print(seg)
                # seg[:, 0] *= width
                # seg[:, 1] *= height
                segment = np.array(seg, dtype=np.int32)
                segmentation_contours_idx.append(segment)


        bboxes = np.array(result.boxes.xyxy.cpu(), dtype="int")
        class_ids = np.array(result.boxes.cls.cpu(), dtype="int")
        scores = np.array(result.boxes.conf.cpu(), dtype="float")
        return bboxes, class_ids, segmentation_contours_idx, scores


        