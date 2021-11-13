import enum
import json
import cv2

import uuid
import glob
import os

ESCAPE_KEY = 27

SAMPLE_FRAMES_PATH = os.environ.get("SAMPLE_FRAMES_PATH", "./sample_video_frames")

extensions = ("*.png","*.jpg","*.jpeg",)
all_files = []
for extension in extensions:
    all_files.extend(glob.iglob(os.path.join(SAMPLE_FRAMES_PATH,extension)))

def process_framed_test_videos():
    import os

    config = {}
    images_count = 0

    i = 0

    for pathAndFilename in all_files:

        title, ext = os.path.splitext(os.path.basename(pathAndFilename))

        image = cv2.imread(pathAndFilename)

        image_with_labels, labels_boxes = YoloV3Manager.predict_image(image)

        if len(labels_boxes) > 0:
            print(labels_boxes)

            save_file = "detected_video_frames/%s.png" % (title)
            cv2.imwrite(save_file, image_with_labels)


YOLOV3_BASE_URL = os.environ.get('YOLOV3_BASE_URL', 'http://localhost:5555/api/')

import requests
class YoloV3Manager():

    @staticmethod
    def predict_image( original_image):
        url = YOLOV3_BASE_URL + 'predict_image'
#        headers = {'content-type': 'multipart/form-data'}

        _, image = cv2.imencode('.jpg', original_image)
        files = {
            'image': ('image.jpg', image, 'image/jpg'),
        }
        print('YoloManager url:', url)

        response = requests.post(url, files=files)#, headers=headers)

        labels_and_boxes = json.loads(response.text)

        # manually add boxes
        image_with_labels = draw_boxes_on_frame(original_image.copy(), labels_and_boxes)

        return image_with_labels, labels_and_boxes


def draw_boxes_on_frame(image, labels_and_boxes):
    # ensure at least one detection exists
    for label_key in labels_and_boxes:
        # loop over the indexes we are keeping
        for box in labels_and_boxes[label_key]:
            # extract the bounding box coordinates
            (x, y) = (box[0], box[1])
            (w, h) = (box[2], box[3])

#            labels_and_boxes.append([LABELS[classIDs[i]], x, y, w, h])

            # draw a bounding box rectangle and label on the image
            color = (255, 0, 0)
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            text = "{}".format(label_key)
            cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,0.5, color, 2)
    return image

process_framed_test_videos()
