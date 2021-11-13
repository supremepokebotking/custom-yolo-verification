# importing libraries
import cv2
import numpy as np
from yolov3_manager import *


def process_framed_test_videos():
    # Create a VideoCapture object and read from input file
    cap = cv2.VideoCapture('Custom Vott Detector Video Clips.mp4')

    # Check if camera opened successfully
    if (cap.isOpened()== False):
      print("Error opening video  file")

    # Read until video is completed
    while(cap.isOpened()):

      # Capture frame-by-frame
      ret, frame = cap.read()
      if ret == True:

        # Display the resulting frame
        image_with_labels, labels_boxes = predict_image(frame)
        cv2.imshow('Frame', image_with_labels)

        # Press Q on keyboard to  exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
          break

      # Break the loop
      else:
        break

    # When everything done, release
    # the video capture object
    cap.release()

    # Closes all the frames
    cv2.destroyAllWindows()

process_framed_test_videos()
