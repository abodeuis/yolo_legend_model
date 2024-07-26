import os
import logging
import numpy as np
from PIL import Image
from typing import List
from patchify import patchify

import cmaas_utils.io as io

log = logging.getLogger(__name__)

def boundingBox(array):
    min_xy = [min(array, key=lambda x: (x[0]))[0], min(array, key=lambda x: (x[1]))[1]]
    max_xy = [max(array, key=lambda x: (x[0]))[0], max(array, key=lambda x: (x[1]))[1]]
    return [min_xy, max_xy]

def is_intersecting(bbox1, bbox2):
    return not (bbox1[0] > bbox2[2] or bbox1[2] < bbox2[0] or bbox1[1] > bbox2[3] or bbox1[3] < bbox2[1])

def convert_pixel_box_to_yolo_box(bbox:List[List[float]], row:int, col:int, patch_size:int, patch_overlap:int):
    """
    Convert annotation from CMASS format to YOLO format
    Returns annotation in [x,y,w,h] format
    """
    min_xy, max_xy = boundingBox(bbox)
    patch_step = patch_size-patch_overlap
    if is_intersecting([*min_xy, *max_xy], [row*patch_step, col*patch_step, ((row+1)*patch_step+patch_overlap), ((col+1)*patch_step+patch_overlap)]):
        # Convert full image px coords to patchwise normalized coords (0-1)
        # Convert image-wise coords to patch-wise coords, cropping to just the patch
        ul_x = max(0, (min_xy[0] - row*patch_step))
        ul_y = max(0, (min_xy[1] - col*patch_step))
        lr_x = min(patch_size, (max_xy[0] - row*patch_step))
        lr_y = min(patch_size, (max_xy[1] - col*patch_step))

        w = (lr_x - ul_x) / patch_size
        h = (lr_y - ul_y) / patch_size
        x = (ul_x + (lr_x - ul_x)/2) / patch_size
        y = (ul_y + (lr_y - ul_y)/2) / patch_size

        return [x, y, w, h]
    
# def old_convert_pixel_box_to_yolo_box(bbox, row, col, patch_size, patch_overlap):
#     min_xy, max_xy = boundingBox(bbox)
#     patch_step = patch_size-patch_overlap
    
#     pix_w = (max_xy[0] - min_xy[0])
#     pix_h = (max_xy[1] - min_xy[1])
#     x = (min_xy[0] - row*patch_step + pix_w/2) / patch_size
#     y = (min_xy[1] - col*patch_step + pix_h/2) / patch_size
#     w = pix_w / patch_size
#     h = pix_h / patch_size
#     return [x, y, w, h]

def convert_cmass_file_to_yolo(image_path:str, label_path:str, yolo_image_dir:str, yolo_label_dir:str, classes:List[str], patch_size:int, patch_overlap:int):
    """
    Convert CMASS image and label files to YOLO format
    """
    log.info(f'Converting {os.path.basename(os.path.splitext(label_path)[0])} to YOLO format')
    
    # Load Data
    patch_step = patch_size-patch_overlap
    image, _, _ = io.loadGeoTiff(image_path)
    legend = io.loadLegendJson(label_path)
    os.makedirs(yolo_image_dir, exist_ok=True)
    os.makedirs(yolo_label_dir, exist_ok=True)

    # Reshape maps with 1 channel images (greyscale) to 3 channels for inference
    if image.shape[0] == 1: # This is tmp fix!
        image = np.concatenate([image,image,image], axis=2)   

    # Patchify data
    image_patches = patchify(image, (3, patch_size, patch_size), step=patch_step).squeeze()

    # Generate YOLO labels
    annotations = 0
    for i in range(image_patches.shape[0]):
        for j in range(image_patches.shape[1]):
            annotation_str = ''
            for feature in legend.features:
                yolo_box = convert_pixel_box_to_yolo_box(feature.bounding_box, i, j, patch_size, patch_overlap)
                if yolo_box is not None:
                    annotations += 1
                    annotation_str += f'{classes.index(feature.type)} {" ".join([str(x) for x in yolo_box])}\n'
            
            # Only save labels if there is a annotation
            if annotation_str != '':
                annotation_filename = os.path.join(yolo_label_dir, os.path.basename(image_path).replace('.tif', f'_{i}_{j}.txt'))
                with open(annotation_filename, 'w') as f:
                    f.write(annotation_str)
    
    # Save image patches
    valid_patches = 0 
    for i in range(image_patches.shape[0]):
        for j in range(image_patches.shape[1]):
            # Only save image patches with annotations
            if not os.path.exists(os.path.join(yolo_label_dir, os.path.basename(image_path).replace('.tif', f'_{i}_{j}.txt'))):
                continue
            # Skip if patch already exists
            patch_filename = os.path.join(yolo_image_dir, os.path.basename(image_path).replace('.tif', f'_{i}_{j}.png'))
            if os.path.exists(patch_filename):
                continue
            valid_patches += 1
            pil_img = Image.fromarray(image_patches[j,i].transpose(1,2,0))
            pil_img.save(patch_filename)
    log.info(f'Saved {valid_patches} image patches with {annotations} annotations for {os.path.basename(os.path.splitext(label_path)[0])}')