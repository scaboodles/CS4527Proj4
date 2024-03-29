import cv2
import numpy as np
    
def find_marker_dots(image_path):
    import cv2
    import numpy as np
    
    # Load the image in color
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    
    # Convert BGR to HSV
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define range for reddish colors in HSV
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([180, 255, 255])

    # Threshold the HSV image to get only reddish colors
    mask1 = cv2.inRange(hsv_image, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv_image, lower_red2, upper_red2)
    reddish_mask = cv2.bitwise_or(mask1, mask2)

    # Optionally, use dilation to connect nearby reddish areas
    kernel = np.ones((5, 5), np.uint8)
    dilated_reddish_mask = cv2.dilate(reddish_mask, kernel, iterations=1)

    # Find connected components or contours to identify groups of reddish points
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(dilated_reddish_mask, connectivity=8)

    # Initialize a blank mask
    mask = np.zeros_like(reddish_mask)

    # Create a mask for each group of reddish points and collect centroids
    positions = []  # To store the x, y positions of each mask
    groups_count = 0  # Counter for the number of groups processed
    for label in range(1, num_labels):  # Start from 1 to ignore the background
        if groups_count >= 4:  # Only process the first four groups
            break
        mask[labels == label] = 255
        # Centroid format in OpenCV is (x, y)
        positions.append((centroids[label][0], centroids[label][1]))
        groups_count += 1  # Increment the counter

    #cv2.imwrite("./testFindGroupsMask.jpg", mask)

    print(positions)
    return sort_markers(positions)


def sort_markers(markers):
# Top-left: minimal sum of coordinates (closest to the origin)
    top_left = min(markers, key=lambda point: point[0] + point[1])
    
    # Top-right: maximal x minus y (furthest to the right but still towards the top)
    top_right = max(markers, key=lambda point: point[0] - point[1])

    # Bottom-left: minimal x minus y (furthest to the left but still towards the bottom)
    bottom_left = min(markers, key=lambda point: point[0] - point[1])
    
    # Bottom-right: maximal sum of coordinates (furthest from the origin)
    bottom_right = max(markers, key=lambda point: point[0] + point[1])

    return [top_left, top_right, bottom_left, bottom_right]

#find_marker_dots("dotsBG.jpg")