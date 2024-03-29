import cv2
import numpy as np

def extract_frames():
    cap = cv2.VideoCapture("./dotsVid.mp4")
    
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
    print ("num frames expected: ", video_length)

    frame_number = 0
    while True:
        # Read a frame
        success, frame = cap.read()

        # If the frame is read correctly, save it as an image
        if success:
            cv2.imwrite(f'frames/{frame_number:04d}.jpg', frame)
            frame_number += 1
        else:
            break

    cap.release()

extract_frames()