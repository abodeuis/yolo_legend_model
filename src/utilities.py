import os
import cv2
import logging
import numpy as np
from typing import List

log = logging.getLogger(__name__)

def boundingBox(array) -> List[List[float]]:
    """
    Get the minimum size bounding box that encompasses a set of points

    Args:
        array : 2 Dimensional list or numpy array with shape (:,2)

    Returns:
        List : [[min_x, min_y], [max_x, max_y]]
    """
    min_xy = [min(array, key=lambda x: (x[0]))[0], min(array, key=lambda x: (x[1]))[1]]
    max_xy = [max(array, key=lambda x: (x[0]))[0], max(array, key=lambda x: (x[1]))[1]]
    return [min_xy, max_xy]

def is_intersecting(bbox1, bbox2) -> bool:
    """
    Check if two bounding boxes are intersecting

    Args:
        bbox1 : List or numpy array in the format [x1,y1,x2,y2]
        bbox2 : List or numpy array in the format [x1,y1,x2,y2]

    Returns:
        bool : True if the bbox1 and bbox2 intersect
    """
    return not (bbox1[0] > bbox2[2] or bbox1[2] < bbox2[0] or bbox1[1] > bbox2[3] or bbox1[3] < bbox2[1])

def progress_wrapper(progress, task:int, func, *args, **kwargs):
    """
    Wrapper for progress bar to update progress bar on completion of function

    Args:
        progress : Progress bar object
        task : Task number to update
        func : Function to run
        *args : Arguments to pass to function
    """
    try:
        result = func(*args, **kwargs)
        progress.update(task, advance=1)
        return result
    except Exception as e:
        log.exception(e)
        progress.update(task, advance=1)
        pass
    
def mask(image, contours):
    """
    Mask image so only area inside contours is visable.

    Args:
        image : numpy array of shape (C,H,W)
        contours : list of numpy arrays of shape (N,2) where N is the number of points in the contour

    Returns:
        mask_image : numpy array of shape (C,H,W)
    """
    image = image.transpose(1,2,0)
    area_mask = np.zeros((image.shape[0], image.shape[1], 1), dtype='uint8')
    for area in contours:
        cv2.drawContours(area_mask, [area], -1, 255, -1)
    mask_image = cv2.bitwise_and(image, image, mask=area_mask)
    mask_image = mask_image.transpose(2,0,1)
    return mask_image

def crop(image, contours):
    """
    Crop image to the bounding box of the contours

    Args:
        image : numpy array of shape (C,H,W)
        contours : list of numpy arrays of shape (N,2) where N is the number of points in the contour

    Returns:
        crop_image : numpy array of shape (C,H,W)
        offset : tuple of (x,y) of the cropped images offset from the orignal image's top left corner.
    """
    image = image.transpose(1,2,0)
    min_pt, max_pt = None, None
    for area in contours:
        area_min_pt, area_max_pt = boundingBox(area)
        if min_pt is None:
            min_pt = area_min_pt
            max_pt = area_max_pt
        else:
            min_pt = (min(min_pt[0], area_min_pt[0]), min(min_pt[1], area_min_pt[1]))
            max_pt = (max(max_pt[0], area_max_pt[0]), max(max_pt[1], area_max_pt[1]))

    crop_image = image[:, min_pt[1]:max_pt[1], min_pt[0]:max_pt[0]]
    crop_image = crop_image.transpose(2,0,1)
    return crop_image, min_pt

def mask_and_crop(image, contours):
    """
    Mask image with contours so that only the area inside is visable and crop the image to the bounding box of the
    visable area

    Args:
        image : numpy array of shape (C,H,W)
        contours : list of numpy arrays of shape (N,2) where N is the number of points in the contour
    
    Returns:
        crop_image : numpy array of shape (C,H,W)
        offset : tuple of (x,y) of the cropped images offset from the orignal image's top left corner.
    """
    image = image.transpose(1,2,0)
    min_pt, max_pt = None, None
    area_mask = np.zeros((image.shape[0], image.shape[1], 1), dtype='uint8')
    for area in contours:
        area_min_pt, area_max_pt = boundingBox(area)
        cv2.drawContours(area_mask, [area], -1, 255, -1)
        if min_pt is None:
            min_pt = area_min_pt
            max_pt = area_max_pt
        else:
            min_pt = (min(min_pt[0], area_min_pt[0]), min(min_pt[1], area_min_pt[1]))
            max_pt = (max(max_pt[0], area_max_pt[0]), max(max_pt[1], area_max_pt[1]))

    # Mask non legend areas
    mask_image = cv2.bitwise_and(image, image, mask=area_mask)
    crop_image = mask_image[min_pt[1]:max_pt[1], min_pt[0]:max_pt[0],:]
    crop_image = crop_image.transpose(2,0,1)
    return crop_image, min_pt