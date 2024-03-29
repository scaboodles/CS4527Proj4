from getMarkerPositions import find_marker_dots
from turkeyPos import get_turkey_centroid, mask_bbox
from interpolate import interpolate_section, interpolate_bounding_box

import numpy as np

def proc_frame(frame_no, bg_anchors):
    frame_path = f"./frames/{frame_no}.jpg"
    marker_posistions = find_marker_dots(frame_path)
    #turkey_mask = get_turkey_centroid(frame_path, marker_posistions)
    #turkey_subimg = save_subsection_from_mask(frame_path, turkey_mask, frame_no) # path to sub image
    interpolate_section(frame_path, marker_posistions, bg_anchors, frame_no)

def proc_frame_subsections(frame_no, bg_anchors):
    frame_path = f"./frames/{frame_no}.jpg"
    marker_posistions = find_marker_dots(frame_path)
    turkey_mask = get_turkey_centroid(frame_path, marker_posistions)
    turkey_bbox = mask_bbox(frame_path, turkey_mask, frame_no) # bouding box of turkey mask
    if(turkey_bbox):
       interpolate_bounding_box(frame_path, marker_posistions, bg_anchors, frame_no, turkey_bbox) 
    else:
        interpolate_section(frame_path, marker_posistions, bg_anchors, frame_no)

proc_frame_subsections("0243", find_marker_dots("dotsBG.jpg"))
