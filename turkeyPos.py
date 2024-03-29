import cv2
import numpy as np

import cv2
import numpy as np

def get_turkey_centroid(image_path, markers):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    y_threshold = 300
    black_threshold = 50

    # Apply threshold to find black or nearly black points
    _, binary_image = cv2.threshold(image, black_threshold, 255, cv2.THRESH_BINARY_INV)

    # Use dilation to connect nearby points into a single group
    kernel = np.ones((5, 5), np.uint8)  # Adjust kernel size as needed
    dilated_image = cv2.dilate(binary_image, kernel, iterations=1)

    # Find connected components or contours to identify groups
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(dilated_image, connectivity=8)

    # Calculate the center of the x-axis of the image
    image_center_x = image.shape[1] / 2

    offset = 10
    # Find the group with the centroid closest to the image center on the x-axis and below the y_threshold
    closest_label = None
    min_distance_to_center = float('inf')
    for label in range(1, num_labels):  # Start from 1 to ignore the background
        centroid_x, centroid_y = centroids[label]
        if centroid_y > y_threshold:  # Check if the centroid is below the y_threshold
            if(centroid_x > markers[0][0] + offset  and centroid_x < markers[3][0] - offset):
                closest_label = label

    # Initialize a blank mask
    mask = np.zeros_like(image)

    # Create a mask for the centermost group below the y_threshold
    if closest_label is not None:
        mask[labels == closest_label] = 255

    cv2.imwrite("./testFindGroupsMask.jpg", mask)
    return mask

def save_subsection_from_mask(image_path, mask, image_no):
    image = cv2.imread(image_path)

    # find contours 
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # assuming the largest contour corresponds to the desired mask
        largest_contour = max(contours, key=cv2.contourArea)
        
        # get the bounding box of the largest contour
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # crop the original image using the bounding box coordinates
        cropped_image = image[y:y+h, x:x+w]
        
        cropped_image_path = f"intermediaries/{image_no}.jpg"

        cv2.imwrite(cropped_image_path, cropped_image)
        
        return cropped_image_path
    else:
        print("No contours found in the mask.")
        return False