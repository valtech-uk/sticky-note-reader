import cv2 as cv
import numpy as np

def non_max_suppression(boxes, scores, threshold):
    assert boxes.shape[0] == scores.shape[0]
    ys1 = boxes[:, 0]
    xs1 = boxes[:, 1]
    ys2 = boxes[:, 2]
    xs2 = boxes[:, 3]
    areas = (ys2 - ys1) * (xs2 - xs1)
    scores_indexes = scores.argsort().tolist()
    boxes_keep_index = []
    while len(scores_indexes):
        index = scores_indexes.pop()
        boxes_keep_index.append(index)
        if not len(scores_indexes):
            break
        ious = compute_iou(boxes[index], boxes[scores_indexes], areas[index],
                           areas[scores_indexes])
        filtered_indexes = set((ious > threshold).nonzero()[0])
        scores_indexes = [
            v for (i, v) in enumerate(scores_indexes)
            if i not in filtered_indexes
        ]
    return np.array(boxes_keep_index)

def compute_iou(box, boxes, box_area, boxes_area):
    assert boxes.shape[0] == boxes_area.shape[0]
    ys1 = np.maximum(box[0], boxes[:, 0])
    xs1 = np.maximum(box[1], boxes[:, 1])
    ys2 = np.minimum(box[2], boxes[:, 2])
    xs2 = np.minimum(box[3], boxes[:, 3])
    intersections = np.maximum(ys2 - ys1, 0) * np.maximum(xs2 - xs1, 0)
    unions = box_area + boxes_area - intersections
    ious = intersections / unions
    return ious

def expand(box, shape):
    box = list(box)
    offset = 3
    box[0] = max(box[0] - offset, 0)
    box[1] = max(box[1] - offset, 0)
    box[2] = min(box[2] + offset, shape[1])
    box[3] = min(box[3] + offset, shape[0])
    return box
    
def word(image, box):
    box = expand(box, image.shape)
    return image[box[1]:box[3], box[0]:box[2]]

class EastDetector:
    def __init__(self, path):
        self.layer_names = [
            "feature_fusion/Conv_7/Sigmoid",
            "feature_fusion/concat_3"]
        self.net = cv.dnn.readNet(path)

    def infer(self, image):
        new_w = int(image.shape[0]/32) * 32
        new_h = int(image.shape[1]/32) * 32
        image = cv.resize(image, (new_w, new_h))
        print(image.shape)
        (H, W) = image.shape[:2]
        blob = cv.dnn.blobFromImage(image, 1.0, (W, H),
            (123.68, 116.78, 103.94), swapRB=True, crop=False)
        self.net.setInput(blob)
        (scores, geometry) = self.net.forward(self.layer_names)
        (numRows, numCols) = scores.shape[2:4]
        rects = []
        confidences = []
        
        # loop over the number of rows
        for y in range(0, numRows):
            # extract the scores (probabilities), followed by the geometrical
            # data used to derive potential bounding box coordinates that
            # surround text
            scoresData = scores[0, 0, y]
            xData0 = geometry[0, 0, y]
            xData1 = geometry[0, 1, y]
            xData2 = geometry[0, 2, y]
            xData3 = geometry[0, 3, y]
            anglesData = geometry[0, 4, y]
            for x in range(0, numCols):
                # if our score does not have sufficient probability, ignore it
                if scoresData[x] < 0.5:
                    continue
        
                # compute the offset factor as our resulting feature maps will
                # be 4x smaller than the input image
                (offsetX, offsetY) = (x * 4.0, y * 4.0)
        
                # extract the rotation angle for the prediction and then
                # compute the sin and cosine
                angle = anglesData[x]
                cos = np.cos(angle)
                sin = np.sin(angle)
        
                # use the geometry volume to derive the width and height of
                # the bounding box
                h = xData0[x] + xData2[x]
                w = xData1[x] + xData3[x]
        
                # compute both the starting and ending (x, y)-coordinates for
                # the text prediction bounding box
                endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
                endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
                startX = int(endX - w)
                startY = int(endY - h)
        
                # add the bounding box coordinates and probability score to
                # our respective lists
                rects.append((startX, startY, endX, endY))
                confidences.append(scoresData[x])

        box_indices = non_max_suppression(np.array(rects), np.array(confidences), 0.5)
        boxes = []
        for i in box_indices:
            boxes.append(rects[i])

        return [word(image, box) for box in boxes]