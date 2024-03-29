import cv2
import numpy as np
    
def find_marker_dots(image_path):
    import cv2
    import numpy as np
    
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    
    # BGR to HSV
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # range for reddish colors in HSV
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([180, 255, 255])

    # threshold HSV image to get only reddish colors
    mask1 = cv2.inRange(hsv_image, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv_image, lower_red2, upper_red2)
    reddish_mask = cv2.bitwise_or(mask1, mask2)

    kernel = np.ones((5, 5), np.uint8)
    dilated_reddish_mask = cv2.dilate(reddish_mask, kernel, iterations=1)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(dilated_reddish_mask, connectivity=8)

    mask = np.zeros_like(reddish_mask)

    positions = []  # x, y positions of each mask
    groups_count = 0  # number of groups processed
    for label in range(1, num_labels):  
        if groups_count >= 4:  # process the first four groups
            break
        mask[labels == label] = 255
        # (x, y)
        positions.append((centroids[label][0], centroids[label][1]))
        groups_count += 1  

    cv2.imwrite("./markers.jpg", mask)

    print(positions)
    return sort_markers(positions)


def sort_markers(markers):
    # top-left: minimal sum of coordinates (closest to the origin)
    top_left = min(markers, key=lambda point: point[0] + point[1])
    
    # top-right: maximal x minus y (furthest to the right but still towards the top)
    top_right = max(markers, key=lambda point: point[0] - point[1])

    # bottom-left: minimal x minus y (furthest to the left but still towards the bottom)
    bottom_left = min(markers, key=lambda point: point[0] - point[1])
    
    # bottom-right: maximal sum of coordinates (furthest from the origin)
    bottom_right = max(markers, key=lambda point: point[0] + point[1])

    return [top_left, top_right, bottom_left, bottom_right]

#find_marker_dots("dotsBG.jpg")