from getMarkerPositions import find_marker_dots
from processFrame import proc_frame, proc_frame_subsections

import os
import glob
import cv2
from PIL import Image

def run_all():
    bg_markers = find_marker_dots("dotsBG.jpg")
    for i in range(325):
        frame_no = f'{"{:04d}"}'.format(i)
        proc_frame(frame_no, bg_markers)
    compile()

def compile():
    image_folder = './finalFrames'
    video_path = 'stabilized.mp4'
    frame_rate = 24

    with Image.open("dotsBG.jpg") as img:
        width, height = img.size

    image_size = (width, height)

    images = glob.glob(os.path.join(image_folder, '*.jpg'))

    images.sort()

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Define the codec
    out = cv2.VideoWriter(video_path, fourcc, frame_rate, image_size)

    for image_filename in images:
        img = cv2.imread(image_filename)
        #img = cv2.resize(img, image_size)  
        out.write(img)

    out.release()

def run_all_subsect():
    bg_markers = find_marker_dots("dotsBG.jpg")
    for i in range(325):
        frame_no = f'{"{:04d}"}'.format(i)
        proc_frame_subsections(frame_no, bg_markers)
    compile_subsect()

def compile_subsect():
    image_folder = './finalFramesSubSect'
    video_path = 'stabilizedSubsections.mp4'
    frame_rate = 24

    with Image.open("dotsBG.jpg") as img:
        width, height = img.size

    image_size = (width, height)

    images = glob.glob(os.path.join(image_folder, '*.jpg'))

    images.sort()

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Define the codec
    out = cv2.VideoWriter(video_path, fourcc, frame_rate, image_size)

    for image_filename in images:
        img = cv2.imread(image_filename)
        #img = cv2.resize(img, image_size)  
        out.write(img)

    out.release()

run_all_subsect()