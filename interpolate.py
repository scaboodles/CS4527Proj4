import cv2
import numpy as np

def interpolate_section(small_image_path, small_anchors, large_anchors, frame_no):
    small_image = cv2.imread(small_image_path)
    large_image = cv2.imread("dotsBG.jpg")

    small_points = np.float32(small_anchors)

    large_points = np.float32(large_anchors)

    # perspective transform 
    matrix = cv2.getPerspectiveTransform(small_points, large_points)

    height, width, channels = large_image.shape

    # aplly transformation
    transformed_small_image = cv2.warpPerspective(small_image, matrix, (width, height))

    # create mask
    mask = np.zeros_like(large_image, dtype=np.uint8)
    mask[transformed_small_image != 0] = 255

    # overlay
    final_image = np.where(mask, transformed_small_image, large_image)

    cv2.imwrite(f'finalFrames/f{frame_no}.jpg', final_image)

def interpolate_bounding_box(small_image_path, small_anchors, large_anchors, frame_no, bbox):
    small_image = cv2.imread(small_image_path)
    large_image = cv2.imread("dotsBG.jpg")

    small_points = np.float32(small_anchors)
    large_points = np.float32(large_anchors)

    matrix = cv2.getPerspectiveTransform(small_points, large_points)

    height, width, channels = large_image.shape

    # apply the perspective transformation 
    transformed_small_image = cv2.warpPerspective(small_image, matrix, (width, height))

    top_left = np.array([bbox[0], bbox[1], 1])
    top_right = np.array([bbox[0] + bbox[2], bbox[1], 1])
    bottom_right = np.array([bbox[0] + bbox[2], bbox[1] + bbox[3], 1])
    bottom_left = np.array([bbox[0], bbox[1] + bbox[3], 1])

    # put into array for transformation
    bbox_points = np.array([top_left, top_right, bottom_right, bottom_left]).T  

    # apply perspective transformation matrix to the bounding box points
    transformed_points = matrix @ bbox_points  # shape 3x4
    transformed_points = transformed_points / transformed_points[2]  # normalize by the last row

    # define the new bounding box
    min_x, min_y = np.min(transformed_points[:2], axis=1)
    max_x, max_y = np.max(transformed_points[:2], axis=1)

    # calculate the new bounding box
    new_bbox = [min_x, min_y, max_x - min_x, max_y - min_y]
    x_trans, y_trans, w_trans, h_trans = map(int, new_bbox)

    # crop
    cropped_transformed_image = transformed_small_image[y_trans:y_trans+h_trans, x_trans:x_trans+w_trans]

    # mask cropped/transformed section
    mask = np.zeros_like(large_image, dtype=np.uint8)
    mask[y_trans:y_trans+h_trans, x_trans:x_trans+w_trans] = cropped_transformed_image

    # overlay
    final_image = np.where(mask != 0, mask, large_image)

    cv2.imwrite(f'finalFramesSubSect/{frame_no}.jpg', final_image)