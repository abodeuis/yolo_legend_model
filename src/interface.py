import os
import logging
import numpy as np
from ultralytics import YOLO
from datetime import datetime
from rich.progress import Progress
from time import sleep
from concurrent.futures import ThreadPoolExecutor, as_completed
from cmaas_utils.logging import start_logger
import cmaas_utils.io as io
from patchify import patchify
from PIL import Image
from typing import List
import yaml

log = logging.getLogger('YoloInterface')

def boundingBox(array):
    min_xy = [min(array, key=lambda x: (x[0]))[0], min(array, key=lambda x: (x[1]))[1]]
    max_xy = [max(array, key=lambda x: (x[0]))[0], max(array, key=lambda x: (x[1]))[1]]
    return [min_xy, max_xy]

def is_intersecting(bbox1, bbox2):
    return not (bbox1[0] > bbox2[2] or bbox1[2] < bbox2[0] or bbox1[1] > bbox2[3] or bbox1[3] < bbox2[1])

def progress_wrapper(progress:Progress, task:int, func, *args, **kwargs):
    try:
        result = func(*args, **kwargs)
        progress.update(task, advance=1)
        return result
    except Exception as e:
        log.exception(e)
        progress.update(task, advance=1)
        pass

def convert_pixel_box_to_yolo_box(bbox, row, col, patch_size, patch_overlap):
    """
    Convert annotation from CMASS format to YOLO format
    Returns annotation in [x,y,w,h] format
    """
    min_xy, max_xy = boundingBox(bbox)
    patch_step = patch_size-patch_overlap
    if is_intersecting([*min_xy, *max_xy], [row*patch_size, col*patch_size, (row+1)*patch_size, (col+1)*patch_size]):
        # Convert full image px coords to patchwise normalized coords (0-1)
        # Convert image-wise coords to patch-wise coords, cropping to just the patch
        ul_x = max(0, (min_xy[0] - row*patch_step))
        ul_y = max(0, (min_xy[1] - col*patch_step))
        br_x = min(1, (max_xy[0] - row*patch_step))
        br_y = min(1, (max_xy[1] - col*patch_step))

        w = (br_x-ul_x) / patch_size
        h = (br_y-ul_y) / patch_size
        x = (ul_x + w/2) / patch_size
        y = (ul_y + h/2) / patch_size
        
        # pix_w = (max_xy[0] - min_xy[0])
        # pix_h = (max_xy[1] - min_xy[1])
        # x = (min_xy[0] - row*patch_size + pix_w/2) 
        # y = (min_xy[1] - col*patch_size + pix_h/2)
        # w = pix_w / patch_size
        # h = pix_h / patch_size
        return [x, y, w, h]

def _convert_cmass_file_to_yolo(image_path, label_path, yolo_image_dir, yolo_label_dir, classes:List, patch_size, patch_overlap):
    """
    Convert a tif image and json label file from CMASS format to YOLO format
    """
    log.debug(f'Converting {os.path.basename(os.path.splitext(label_path)[0])} to YOLO format')
    image, _, _ = io.loadGeoTiff(image_path)
    lgd = io.loadLegendJson(label_path)

    # Reshape maps with 1 channel images (greyscale) to 3 channels for inference
    if image.shape[0] == 1: # This is tmp fix!
        image = np.concatenate([image,image,image], axis=2)   

    patch_step = patch_size-patch_overlap

    image_patches = patchify(image, (3, patch_size, patch_size), step=patch_step).squeeze()
    
    # Save annotations
    for i in range(image_patches.shape[0]):
        for j in range(image_patches.shape[1]):
            outstr = ''
            for feature in lgd.features:
                min_xy, max_xy = boundingBox(feature.bounding_box)
                if is_intersecting([*min_xy, *max_xy], [i*patch_step, j*patch_step, (i+1)*patch_step, (j+1)*patch_step]):
                    # Convert full image px coords to patchwise normalized coords (0-1)
                    feature_class = classes.index(feature.type)
                    pix_w = (max_xy[0] - min_xy[0])
                    pix_h = (max_xy[1] - min_xy[1])
                    x = (min_xy[0] - i*patch_step + pix_w/2) / patch_size
                    y = (min_xy[1] - j*patch_step + pix_h/2) / patch_size
                    w = pix_w / patch_size
                    h = pix_h / patch_size
                    outstr += f'{feature_class} {x} {y} {w} {h}\n'

                if outstr != '':
                    annotation_filename = os.path.join(yolo_label_dir, os.path.basename(image_path).replace('.tif', f'_{i}_{j}.txt'))
                    logging.debug(f'Saving {annotation_filename}')
                    with open(annotation_filename, 'w') as f:
                        f.write(outstr)

    # Save image patches
    for i in range(image_patches.shape[0]):
      for j in range(image_patches.shape[1]):
            # Only save image patches with annotations
            if not os.path.exists(os.path.join(yolo_label_dir, os.path.basename(image_path).replace('.tif', f'_{i}_{j}.txt'))):
                continue
            # Skip if patch already exists
            patch_filename = os.path.join(yolo_image_dir, os.path.basename(image_path).replace('.tif', f'_{i}_{j}.png'))
            if os.path.exists(patch_filename):
                continue
            pil_img = Image.fromarray(image_patches[j,i].transpose(1,2,0))
            pil_img.save(patch_filename)
            logging.debug(f'Saved patches for {patch_filename}')

from cmaas_utils.types import MapUnitType
class YoloInterface():
    classes = [MapUnitType.POINT, MapUnitType.LINE, MapUnitType.POLYGON]

    def __init__(self, checkpoint: str=None, patch_size: int=512, patch_overlap: int=0, unpatch_strategy: str='discard'):
        self.model = None
        if checkpoint is not None:
            self.model = YOLO(checkpoint)

        self.patch_size = patch_size
        self.patch_overlap = patch_overlap
        self.unpatch_strategy = unpatch_strategy

    def load_from_checkpoint(self, checkpoint:str=''):
        """
        """
        self.model = YOLO(checkpoint)

    def train(self, path: str, epochs=5, cache_data: bool=False):
        """

        """
        if self.model is None:
            log.warning("No pretrained model loaded, training from scratch")
            self.model = YOLO()
        
        root_dir = f'yolo_training_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
        data_dir = os.path.join(root_dir, 'data')
        global train_log
        # train_log = start_logger('YoloInterface', os.path.join(root_dir, 'logs','train.log'), logging.DEBUG)
        log.info(f'Starting training with data from {path}')

        # Format data for yolo training
        yaml_config = self.convert_cmass_dataset_to_yolo(path, data_dir)
        
        # Actually train the model
        train_results = self.model.train(data=yaml_config, epochs=epochs, imgsz=self.patch_size)
        val_results = self.model.val()

        success = self.model.export(format='onnx')

        # Clean up data directory after training.
        if not cache_data:
            os.system(f'rm -rf {data_dir}')

    def convert_cmass_dataset_to_yolo(self, usgs_data_path:str, yolo_data_path:str):
        """
        """

        train_labels = os.listdir(os.path.join(usgs_data_path,'train','labels'))
        val_labels = os.listdir(os.path.join(usgs_data_path,'val','labels'))
        test_labels = os.listdir(os.path.join(usgs_data_path,'test','labels'))

        log.info(f'Creating output directories in {yolo_data_path}')
        os.makedirs(os.path.join(yolo_data_path,'train','images'), exist_ok=True)
        os.makedirs(os.path.join(yolo_data_path,'train','labels'), exist_ok=True)
        os.makedirs(os.path.join(yolo_data_path,'val','images'), exist_ok=True)
        os.makedirs(os.path.join(yolo_data_path,'val','labels'), exist_ok=True)
        os.makedirs(os.path.join(yolo_data_path,'test','images'), exist_ok=True)
        os.makedirs(os.path.join(yolo_data_path,'test','labels'), exist_ok=True)

        log.info('Starting data formatting to yolov8 format')
        with Progress() as progress:
            train_task = progress.add_task("Train", total=len(train_labels))
            val_task = progress.add_task("Val  ", total=len(val_labels))
            test_task = progress.add_task("Test ", total=len(test_labels))

            with ThreadPoolExecutor(max_workers=48) as p:
                futures = []
                for label in train_labels:
                    image_path = os.path.join(usgs_data_path,'train','images',label.replace('.json','.tif'))
                    label_path = os.path.join(usgs_data_path,'train','labels',label)
                    yolo_image_dir = os.path.join(yolo_data_path,'train','images')
                    yolo_label_dir = os.path.join(yolo_data_path,'train','labels')
                    futures.append(p.submit(
                        progress_wrapper, progress, train_task,
                        _convert_cmass_file_to_yolo, image_path, label_path, yolo_image_dir, yolo_label_dir, self.classes, self.patch_size, self.patch_overlap))
                for label in val_labels:
                    image_path = os.path.join(usgs_data_path,'val','images',label.replace('.json','.tif'))
                    label_path = os.path.join(usgs_data_path,'val','labels',label)
                    yolo_image_dir = os.path.join(yolo_data_path,'val','images')
                    yolo_label_dir = os.path.join(yolo_data_path,'val','labels')
                    futures.append(p.submit(
                        progress_wrapper, progress, val_task,
                        _convert_cmass_file_to_yolo, image_path, label_path, yolo_image_dir, yolo_label_dir, self.classes, self.patch_size, self.patch_overlap))
                for label in test_labels:
                    image_path = os.path.join(usgs_data_path,'test','images',label.replace('.json','.tif'))
                    label_path = os.path.join(usgs_data_path,'test','labels',label)
                    yolo_image_dir = os.path.join(yolo_data_path,'test','images')
                    yolo_label_dir = os.path.join(yolo_data_path,'test','labels')
                    futures.append(p.submit(
                        progress_wrapper, progress, test_task,
                        _convert_cmass_file_to_yolo, image_path, label_path, yolo_image_dir, yolo_label_dir, self.classes, self.patch_size, self.patch_overlap))

                for future in as_completed(futures):
                    try:
                        result = future.result()
                    except Exception as e:
                        logging.exception(e)
                        continue
            
            # Wait till training data is done.
            while not progress.finished:
                sleep(0.1)

        return self._create_yaml_config(yolo_data_path, 'train/images', 'val/images', 'test/images')

    def _create_yaml_config(self, root_path: str, train_path, val_path, test_path, filename='data.yaml'):
        """
        """
        yaml_contents = '# Data\n'
        yaml_contents += f'path: {os.path.abspath(root_path)}\n'
        yaml_contents += f'train: {os.path.abspath(os.path.join(root_path,train_path))}\n'
        yaml_contents += f'val: {os.path.abspath(os.path.join(root_path,val_path))}\n'
        yaml_contents += f'test: {os.path.abspath(os.path.join(root_path,test_path))}\n'
        yaml_contents += '\n'
        yaml_contents += '# Classes\n'
        yaml_contents += 'names:\n'
        for i, name in enumerate(self.classes):
            yaml_contents += f'    {i}: {name}\n'
        
        yaml_path = os.path.join(root_path, filename)
        with open(yaml_path, 'w') as fh:
            fh.write(yaml_contents)

        return yaml_path


    def inference(self, image: np.ndarray):
        """
        image : 3 dimensional array
        """
        # Type check
        image = self._check_image(image)

    def _check_image(self, image: np.ndarray) -> np.ndarray:
        """
        image : 3 dimensional array in format CHW
        """
        
        if len(image.shape) != 3:
            raise ValueError("Shape of Image must be 3 dimensional array")
        
        if image.shape[0] == 1:
            # Figure out if we should convert to greyscale or rgb
            logging.warning("Image is greyscale, converting to rgb")
            # TODO: convert the image to rgb

        if image.shape[0] not in [1, 3]:
            raise ValueError("Should be in format CHW, Invalid number of channels only supports 1 for greyscale or 3 for rgb")
        
        #check the type of the image pixels
        if image.dtype != np.uint8: 
            logging.warning(f"Image pixel type is {image.dtype} converting to uint8")
            # TODO: convert the image to uint8

        return image
       