import cv2
import numpy as np

def interpolate_section(small_image_path, small_anchors, large_anchors, frame_no):
    small_image = cv2.imread(small_image_path)
    large_image = cv2.imread("dotsBG.jpg")

    # Define marker points in the small image
    # These should be [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
    small_points = np.float32(small_anchors)

    # Define corresponding points in the large image
    large_points = np.float32(large_anchors)

    # Compute the perspective transform matrix
    matrix = cv2.getPerspectiveTransform(small_points, large_points)

    # Determine the dimensions of the large image
    height, width, channels = large_image.shape

    # Apply the perspective transformation to the small image
    transformed_small_image = cv2.warpPerspective(small_image, matrix, (width, height))

    # Create a mask from the transformed small image
    mask = np.zeros_like(large_image, dtype=np.uint8)
    mask[transformed_small_image != 0] = 255

    # Overlay the transformed small image onto the large image
    final_image = np.where(mask, transformed_small_image, large_image)

    cv2.imwrite(f'finalFrames/f{frame_no}.jpg', final_image)