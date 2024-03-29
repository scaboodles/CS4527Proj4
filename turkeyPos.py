import cv2
import numpy as np

import cv2
import numpy as np

def get_turkey_centroid(image_path, markers):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    y_threshold = 300
    black_threshold = 50

    # threshold to find black 
    _, binary_image = cv2.threshold(image, black_threshold, 255, cv2.THRESH_BINARY_INV)

    # dilation to connect nearby points into a single group
    kernel = np.ones((5, 5), np.uint8)  
    dilated_image = cv2.dilate(binary_image, kernel, iterations=1)

    # find connected components 
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(dilated_image, connectivity=8)

    offset = 10
    # turkey is between marker points and below y threshold
    closest_label = None
    min_distance_to_center = float('inf')
    for label in range(1, num_labels):  
        centroid_x, centroid_y = centroids[label]
        if centroid_y > y_threshold:  
            if(centroid_x > markers[0][0] + offset  and centroid_x < markers[3][0] - offset):
                closest_label = label

    mask = np.zeros_like(image)

    # mask centermost group below the y_threshold
    if closest_label is not None:
        mask[labels == closest_label] = 255

    #cv2.imwrite("./testFindGroupsMask.jpg", mask)
    return mask

def mask_bbox(image_path, mask, image_no):
    image = cv2.imread(image_path)

    # find contours 
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # dark magic
        largest_contour = max(contours, key=cv2.contourArea)
        
        # get the bounding box of the largest contour
        x, y, w, h = cv2.boundingRect(largest_contour)
        return [x,y,w,h]

    else:
        print("No contours found in the mask.")
        return False