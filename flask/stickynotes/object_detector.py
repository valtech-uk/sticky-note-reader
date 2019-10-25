import numpy as np
import tensorflow as tf
import cv2 as cv
from collections import defaultdict
from io import StringIO
import os
import sys
import queue
import time
import threading
from matplotlib import pyplot as plt
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

PATH_TO_FROZEN_GRAPH = 'stickynote_inference_graph/frozen_inference_graph.pb'
PATH_TO_LABELS = 'training/stickynote.pbtxt'

def load_model(graph_path, label_path):
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.compat.v1.GraphDef()
        with tf.io.gfile.GFile(graph_path, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    category_index = label_map_util.create_category_index_from_labelmap(label_path, use_display_name=True)
    return detection_graph, category_index


def detector(input_q, output_q, graph, category_index):
    detect = True
    with graph.as_default():
        with tf.compat.v1.Session() as sess:
            # Get handles to input and output tensors
            ops = tf.compat.v1.get_default_graph().get_operations()
            all_tensor_names = {output.name for op in ops for output in op.outputs}
            tensor_dict = {}
            for key in [
                'num_detections', 'detection_boxes', 'detection_scores',
                'detection_classes', 'detection_masks'
            ]:
                tensor_name = key + ':0'
                if tensor_name in all_tensor_names:
                    tensor_dict[key] = tf.compat.v1.get_default_graph().get_tensor_by_name(
                        tensor_name)
            while detect:
                try:
                    image = input_q.get(block = False)
                except queue.Empty:
                    time.sleep(1)
                else:
                    if image is None:
                        detect = False
                        continue
                    
                    image = np.expand_dims(image, axis=0)
                    if 'detection_masks' in tensor_dict:
                        # The following processing is only for single image
                        detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
                        detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
                        # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
                        real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
                        detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
                        detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
                        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                            detection_masks, detection_boxes, image.shape[1], image.shape[2])
                        detection_masks_reframed = tf.cast(
                            tf.greater(detection_masks_reframed, 0.5), tf.uint8)
                        # Follow the convention by adding back the batch dimension
                        tensor_dict['detection_masks'] = tf.expand_dims(
                            detection_masks_reframed, 0)
                    image_tensor = tf.compat.v1.get_default_graph().get_tensor_by_name('image_tensor:0')

                    # Run inference
                    output_dict = sess.run(tensor_dict,
                                            feed_dict={image_tensor: image})

                    # all outputs are float32 numpy arrays, so convert types as appropriate
                    output_dict['num_detections'] = int(output_dict['num_detections'][0])
                    output_dict['detection_classes'] = output_dict[
                        'detection_classes'][0].astype(np.int64)
                    output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
                    output_dict['detection_scores'] = output_dict['detection_scores'][0]
                    if 'detection_masks' in output_dict:
                        output_dict['detection_masks'] = output_dict['detection_masks'][0]
                    output_q.put(output_dict)
                    input_q.task_done()

class Detector:
    def __init__(self, graph_path, label_path, threshold=0.4):
        graph, self.category_index = load_model(graph_path, label_path)
        self.input_q = queue.Queue()
        self.output_q = queue.Queue()
        self.threshold = threshold
        self.detector = threading.Thread(target=detector, args=(self.input_q, self.output_q, graph, self.category_index))
        self.detector.start()

    def infer_boxes(self, image):
        self.input_q.put(image)
        self.input_q.join()
        output_dict = self.output_q.get()
        detections = []
        for box, score in zip(output_dict['detection_boxes'], output_dict['detection_scores']):
            if score > self.threshold:
                detections.append(box)
        return detections

    def box_image(self, box, image):
        xmin = max(0, int(box[0] * image.shape[0]))
        xmax = min(image.shape[0], int(box[2] * image.shape[0]))
        ymin = max(0, int(box[1] * image.shape[1]))
        ymax = min(image.shape[1], int(box[3] * image.shape[1]))
        return image[xmin:xmax, ymin:ymax,:]

    def infer_images(self, image):
        detections = self.infer_boxes(image)
        images = []
        for box in detections:
            images.append(self.box_image(box, image))
        return images

if __name__ == "__main__":
    images = ["/Users/harry.woods/Dropbox (Valtech)/Sticky notes/images/raw/coar_notice_board.jpg", 
        "/Users/harry.woods/Dropbox (Valtech)/Sticky notes/images/raw/full-calendar.jpg", 
        "/Users/harry.woods/Dropbox (Valtech)/Sticky notes/images/raw/Board_Fotor.jpg"]
    input_q = queue.Queue()
    output_q = queue.Queue()
    det = threading.Thread(target=detector, args=(input_q, output_q, graph, category_index))
    det.start()
    counter = 0
    for path in images:
        image = cv.cvtColor(cv.imread(path, 1), cv.COLOR_BGR2RGB)
        input_q.put(image)
        input_q.join()
        output_dict = output_q.get()
        # Visualization of the results of a detection.
        vis_util.visualize_boxes_and_labels_on_image_array(image,
                                                        output_dict['detection_boxes'],
                                                        output_dict['detection_classes'],
                                                        output_dict['detection_scores'],
                                                        category_index,
                                                        instance_masks=output_dict.get('detection_masks'),
                                                        use_normalized_coordinates=True,
                                                        line_thickness=8)
        plt.figure()
        plt.imsave('output_images/image_{}.jpg'.format(counter), image)
        counter += 1
    input_q.put(None)
    det.join()
