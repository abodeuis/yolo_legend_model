import os
import logging
import numpy as np
from PIL import Image
from typing import List
from patchify import patchify
from rich.progress import Progress
from concurrent.futures import ThreadPoolExecutor, as_completed

import cmaas_utils.io as io

log = logging.getLogger(__name__)

def progress_wrapper(progress:Progress, task:int, func, *args, **kwargs):
    try:
        result = func(*args, **kwargs)
        progress.update(task, advance=1)
        return result
    except Exception as e:
        log.exception(e)
        progress.update(task, advance=1)
        return False

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
    return True

def convert_cmass_dataset_to_yolo(train_files, val_files, test_files, output_dir, classes, patch_size, patch_overlap):

    # Create Output Directory
    log.info(f'Creating output directories in {output_dir}')
    subfolders = ['train', 'val', 'test']
    subsubfolders = ['images', 'labels']
    for subfolder in subfolders:
        for subsubfolder in subsubfolders:
            os.makedirs(os.path.join(output_dir, subfolder, subsubfolder), exist_ok=True)

    # Convert Data
    log.info('Starting data conversion')
    with Progress() as progress:
        train_task = progress.add_task("Train", total=len(train_files))
        val_task = progress.add_task("Val  ", total=len(val_files))
        test_task = progress.add_task("Test ", total=len(test_files))

        with ThreadPoolExecutor() as p:
            futures = []
            # Train files
            train_image_dir = os.path.join(output_dir, 'train', 'images')
            train_label_dir = os.path.join(output_dir, 'train', 'labels')
            for file in train_files:
                basepath = os.path.splitext(file)[0]
                futures.append(p.submit(
                    progress_wrapper, progress, train_task,
                    convert_cmass_file_to_yolo, f'{basepath}.tif', f'{basepath}.json', train_image_dir, train_label_dir, classes, patch_size, patch_overlap
                ))
            # Val files
            val_image_dir = os.path.join(output_dir, 'val', 'images')
            val_label_dir = os.path.join(output_dir, 'val', 'labels')
            for file in val_files:
                basepath = os.path.splitext(file)[0]
                futures.append(p.submit(
                    progress_wrapper, progress, val_task,
                    convert_cmass_file_to_yolo, f'{basepath}.tif', f'{basepath}.json', val_image_dir, val_label_dir, classes, patch_size, patch_overlap
                ))
            # Test files
            test_image_dir = os.path.join(output_dir, 'test', 'images')
            test_label_dir = os.path.join(output_dir, 'test', 'labels')
            for file in test_files:
                basepath = os.path.splitext(file)[0]
                futures.append(p.submit(
                    progress_wrapper, progress, test_task,
                    convert_cmass_file_to_yolo, f'{basepath}.tif', f'{basepath}.json', test_image_dir, test_label_dir, classes, patch_size, patch_overlap
                ))

            # Wait till files are converted
            successes = 0
            failures = 0
            for future in as_completed(futures):
                result = future.result()
                if result:
                    successes += 1
                else:
                    failures += 1

    log.info(f'Converted {successes} files successfully, {failures} files failed')
    yaml_path = create_yaml_config(output_dir, train_image_dir, val_image_dir, test_image_dir, classes)
    log.info(f'Saved yaml config for dataset at {yaml_path}')
    return yaml_path

def create_yaml_config(output_dir: str, train_dir:str, val_dir:str, test_dir:str, classes:List[str], filename='data.yaml'):
        """
        """
        yaml_contents = '# Data\n'
        yaml_contents += f'path: {os.path.abspath(output_dir)}\n'
        yaml_contents += f'train: {os.path.abspath(train_dir)}\n'
        yaml_contents += f'val: {os.path.abspath(val_dir)}\n'
        yaml_contents += f'test: {os.path.abspath(test_dir)}\n'
        yaml_contents += '\n'
        yaml_contents += '# Classes\n'
        yaml_contents += 'names:\n'
        for i, name in enumerate(classes):
            yaml_contents += f'    {i}: {name}\n'
        
        yaml_path = os.path.join(output_dir, filename)
        with open(yaml_path, 'w') as fh:
            fh.write(yaml_contents)

        return yaml_path
