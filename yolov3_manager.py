# import the necessary packages
import numpy as np
import argparse
import cv2
import os
import io as StringIO
import base64
from io import BytesIO
import io
import json
from PIL import Image
import base64
try:
    from StringIO import StringIO ## for Python 2
except ImportError:
    from io import StringIO ## for Python 3

import time

confthres = 0.3
nmsthres = 0.1
yolo_path = '.'

import os
DEBUG = bool(int(os.environ.get('DEBUG', 0)))
USE_GPU = bool(int(os.environ.get('USE_GPU', 0)))
PRINT_EXTRA_INFO = bool(int(os.environ.get('PRINT_EXTRA_INFO', 0)))

def head_print(message):
    if PRINT_EXTRA_INFO:
        print(message)

def get_labels(labels_path):
    # load the COCO class labels our YOLO model was trained on
    #labelsPath = os.path.sep.join([yolo_path, "yolo_v3/coco.names"])
    lpath=os.path.sep.join([yolo_path, labels_path])
    LABELS = open(lpath).read().strip().split("\n")
    return LABELS

def get_colors(LABELS):
    # initialize a list of colors to represent each possible class label
    np.random.seed(42)
    COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),dtype="uint8")
    return COLORS

def get_weights(weights_path):
    # derive the paths to the YOLO weights and model configuration
    weightsPath = os.path.sep.join([yolo_path, weights_path])
    return weightsPath

def get_config(config_path):
    configPath = os.path.sep.join([yolo_path, config_path])
    return configPath

def load_model(configpath,weightspath):
    # load our YOLO object detector trained on COCO dataset (80 classes)
    head_print("[INFO] loading YOLO from disk...")
    net = cv2.dnn.readNetFromDarknet(configpath, weightspath)
    if USE_GPU:
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    else:
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    return net


def image_to_byte_array(image:Image):
  imgByteArr = io.BytesIO()
  image.save(imgByteArr, format='PNG')
  imgByteArr = imgByteArr.getvalue()
  return imgByteArr


def get_predection(image,net,LABELS,COLORS):
    (H, W) = image.shape[:2]

    # determine only the *output* layer names that we need from YOLO
    ln = net.getLayerNames()
    try:
        ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    except Exception:
        ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]

    # construct a blob from the input image and then perform a forward
    # pass of the YOLO object detector, giving us our bounding boxes and
    # associated probabilities
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
                                 swapRB=True, crop=False)
    net.setInput(blob)
    start = time.time()
    layerOutputs = net.forward(ln)
    head_print(layerOutputs)
    end = time.time()

    # show timing information on YOLO
    print("[INFO] YOLO took {:.6f} seconds".format(end - start))

    # initialize our lists of detected bounding boxes, confidences, and
    # class IDs, respectively
    boxes = []
    confidences = []
    classIDs = []

    # loop over each of the layer outputs
    for output in layerOutputs:
        # loop over each of the detections
        for detection in output:
            # extract the class ID and confidence (i.e., probability) of
            # the current object detection
            scores = detection[5:]
            # head_print(scores)
            classID = np.argmax(scores)
            # head_print(classID)
            confidence = scores[classID]

            # filter out weak predictions by ensuring the detected
            # probability is greater than the minimum probability
            if confidence > confthres:
                # scale the bounding box coordinates back relative to the
                # size of the image, keeping in mind that YOLO actually
                # returns the center (x, y)-coordinates of the bounding
                # box followed by the boxes' width and height
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                # use the center (x, y)-coordinates to derive the top and
                # and left corner of the bounding box
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                # update our list of bounding box coordinates, confidences,
                # and class IDs
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    # apply non-maxima suppression to suppress weak, overlapping bounding
    # boxes
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, confthres,
                            nmsthres)
    labels_and_boxes = []

    # ensure at least one detection exists
    if len(idxs) > 0:
        # loop over the indexes we are keeping
        for i in idxs.flatten():
            # extract the bounding box coordinates
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

            labels_and_boxes.append([LABELS[classIDs[i]], x, y, w, h])

            # draw a bounding box rectangle and label on the image
            color = [int(c) for c in COLORS[classIDs[i]]]
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
            head_print(boxes)
            head_print(classIDs)
            cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,0.5, color, 2)
    return image, labels_and_boxes

def process_image(image, net,LABELS,COLORS):
    (H, W) = image.shape[:2]

    # determine only the *output* layer names that we need from YOLO
    ln = net.getLayerNames()
    try:
        ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    except Exception:
        ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]

    # construct a blob from the input image and then perform a forward
    # pass of the YOLO object detector, giving us our bounding boxes and
    # associated probabilities
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
                                 swapRB=True, crop=False)
    net.setInput(blob)
    start = time.time()
    outs = net.forward(ln)
    head_print(outs)
    end = time.time()

    # show timing information on YOLO
    head_print("[INFO] YOLO took {:.6f} seconds".format(end - start))

    # initialize our lists of detected bounding boxes, confidences, and
    # class IDs, respectively

    classIDs = []
    confidences = []
    boxes = []
    conf_threshold = 0.1
#        conf_threshold = self.score
    nms_threshold = 0.1

    labels_and_boxes = {}

    ALOT = 1e6
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > conf_threshold:
                label = str(LABELS[class_id])
                #head_print(label)

#                center_x = int(detection[0] * W)
#                center_y = int(detection[1] * H)
#                w = int(detection[2] * W)
#                h = int(detection[3] * H)
                center_x = int(max(min(detection[0] * W, ALOT), -ALOT))
                center_y = int(max(min(detection[1] * H, ALOT), -ALOT))
                w = int(max(min(detection[2] * W, ALOT), -ALOT))
                h = int(max(min(detection[3] * H, ALOT), -ALOT))
                x = center_x - w / 2
                y = center_y - h / 2
                classIDs.append(class_id)
                confidences.append(float(confidence))

                boxes.append([int(x), int(y), int(w), int(h)])

    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

    for i in indices:

        try:
            i = i[0]
        except Exception:
            i = i
        box = boxes[i]
        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]

        label = str(LABELS[classIDs[i]])

        if label not in labels_and_boxes:
            labels_and_boxes[label] = []
        boxes_for_label = labels_and_boxes[label]
        boxes_for_label.append([x, y, x+w, y+h])

        # draw a bounding box rectangle and label on the image
        color = [int(c) for c in COLORS[classIDs[i]]]
        head_print(color)
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
        text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
        head_print(boxes)
        head_print(classIDs)
        cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,0.5, color, 2)


    return image, labels_and_boxes


labelsPath="yolo_v3/obj.names"
cfgpath="yolo_v3/yolov3-tiny.cfg"
wpath="yolo_v3/yolov3-tiny_final.weights"
Lables=get_labels(labelsPath)
CFG=get_config(cfgpath)
Weights=get_weights(wpath)
nets=load_model(CFG,Weights)
Colors=get_colors(Lables)
frame_count=0


# route http posts to this method
def predict_image(image):
    res, labels_and_boxes=process_image(image.copy(),nets,Lables,Colors)
#    print(labels_and_boxes)
    head_print(labels_and_boxes)
    return res, labels_and_boxes
