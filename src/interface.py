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

def boundingBox(array):
    min_xy = [min(array, key=lambda x: (x[0]))[0], min(array, key=lambda x: (x[1]))[1]]
    max_xy = [max(array, key=lambda x: (x[0]))[0], max(array, key=lambda x: (x[1]))[1]]
    return [min_xy, max_xy]

def is_intersecting(bbox1, bbox2):
    return not (bbox1[0] > bbox2[2] or bbox1[2] < bbox2[0] or bbox1[1] > bbox2[3] or bbox1[3] < bbox2[1])

# def _progress_wrapper(func, *args, progress=None, task=None, **kwargs):
#     out = func(*args, **kwargs)
#     # Increment progress for multi-threading
#     if progress is not None and task is not None:
#         progress.update(task, advance=1)
#     return out

from typing import List
def _format_training_file(usgs_image_path, usgs_label_path, yolo_image_dir, yolo_label_dir, classes:List, patch_size, patch_overlap):
    logging.debug(f'Formatting {usgs_label_path} and {usgs_image_path}')
    image, _, _ = io.loadGeoTiff(usgs_image_path)
    lgd = io.loadLegendJson(usgs_label_path)

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
                    annotation_filename = os.path.join(yolo_label_dir, os.path.basename(usgs_image_path).replace('.tif', f'_{i}_{j}.txt'))
                    logging.debug(f'Saving {annotation_filename}')
                    with open(annotation_filename, 'w') as f:
                        f.write(outstr)
    logging.debug(f'Saved annotations for {annotation_filename}')

    # Save image patches
    for i in range(image_patches.shape[0]):
      for j in range(image_patches.shape[1]):
            # Only save image patches with annotations
            if not os.path.exists(os.path.join(yolo_label_dir, os.path.basename(usgs_image_path).replace('.tif', f'_{i}_{j}.txt'))):
                continue
            # Skip if patch already exists
            patch_filename = os.path.join(yolo_image_dir, os.path.basename(usgs_image_path).replace('.tif', f'_{i}_{j}.png'))
            if os.path.exists(patch_filename):
                continue
            logging.debug(f'Saving {patch_filename}')
            pil_img = Image.fromarray(image_patches[j,i].transpose(1,2,0))
            pil_img.save(patch_filename)
    logging.debug(f'Saved patches for {patch_filename}')

from cmaas_utils.types import MapUnitType
class YoloInterface():
    classes = [MapUnitType.POINT, MapUnitType.LINE, MapUnitType.POLYGON, MapUnitType.UNKNOWN]

    def __init__(self, checkpoint: str=None):
        self.model = None
        if checkpoint is not None:
            self.model = YOLO(checkpoint)

        self.patch_size = 512
        self.patch_overlap = 0
        self.unpatch_strategy = 'discard'

    def load_from_checkpoint(self, checkpoint:str=''):
        """
        """
        self.model = YOLO(checkpoint)

    def train(self, path: str, epochs=5, cache_data: bool=False):
        """

        """
        if self.model is None:
            logging.warning("No pretrained model loaded, training from scratch")
            self.model = YOLO()
        
        root_dir = f'yolo_training_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
        data_dir = os.path.join(root_dir, 'data')
        train_log = start_logger('TRAINING', os.path.join(root_dir, 'logs','train.log'), logging.DEBUG)
        train_log.info(f'Starting training with data from {path}')

        # Format data for yolo training
        yaml_config = self.format_training_data(path, data_dir)
        
        # Actually train the model
        train_results = self.model.train(data=yaml_config, epochs=epochs, imgsz=self.patch_size)
        val_results = self.model.val()

        sucess = self.model.export(format='onnx')

        # Clean up data directory after training.
        if not cache_data:
            os.system(f'rm -rf {data_dir}')

    def format_training_data(self, usgs_data_path:str, yolo_data_path:str):
        """
        """

        train_labels = os.listdir(os.path.join(usgs_data_path,'train','labels'))
        val_labels = os.listdir(os.path.join(usgs_data_path,'val','labels'))
        test_labels = os.listdir(os.path.join(usgs_data_path,'test','labels'))

        logging.info(f'Creating output directories in {yolo_data_path}')
        os.makedirs(os.path.join(yolo_data_path,'train','images'), exist_ok=True)
        os.makedirs(os.path.join(yolo_data_path,'train','labels'), exist_ok=True)
        os.makedirs(os.path.join(yolo_data_path,'val','images'), exist_ok=True)
        os.makedirs(os.path.join(yolo_data_path,'val','labels'), exist_ok=True)
        os.makedirs(os.path.join(yolo_data_path,'test','images'), exist_ok=True)
        os.makedirs(os.path.join(yolo_data_path,'test','labels'), exist_ok=True)

        logging.info('Starting data formatting to yolov8 format')
        with Progress() as progress:
            train_task = progress.add_task("Train", total=len(train_labels))
            val_task = progress.add_task("Val  ", total=len(val_labels))
            test_task = progress.add_task("Test ", total=len(test_labels))

            args = []
            for label in train_labels:
                image_path = os.path.join(usgs_data_path,'train','images',label.replace('.json','.tif'))
                label_path = os.path.join(usgs_data_path,'train','labels',label)
                yolo_image_dir = os.path.join(yolo_data_path,'train','images')
                yolo_label_dir = os.path.join(yolo_data_path,'train','labels')
                args.append(((image_path, label_path, yolo_image_dir, yolo_label_dir, self.classes, self.patch_size, self.patch_overlap), train_task))
            for label in val_labels:
                image_path = os.path.join(usgs_data_path,'val','images',label.replace('.json','.tif'))
                label_path = os.path.join(usgs_data_path,'val','labels',label)
                yolo_image_dir = os.path.join(yolo_data_path,'val','images')
                yolo_label_dir = os.path.join(yolo_data_path,'val','labels')
                args.append(((image_path, label_path, yolo_image_dir, yolo_label_dir, self.classes, self.patch_size, self.patch_overlap), val_task))
            for label in test_labels:
                image_path = os.path.join(usgs_data_path,'test','images',label.replace('.json','.tif'))
                label_path = os.path.join(usgs_data_path,'test','labels',label)
                yolo_image_dir = os.path.join(yolo_data_path,'test','images')
                yolo_label_dir = os.path.join(yolo_data_path,'test','labels')
                args.append(((image_path, label_path, yolo_image_dir, yolo_label_dir, self.classes, self.patch_size, self.patch_overlap),test_task))

            with ThreadPoolExecutor(max_workers=4) as p:
                futures = [(p.submit(_format_training_file, *arg[0]), arg[1]) for arg in args]
                for future, task in futures:
                    result = future.result()
                    progress.update(task, advance=1)
            
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
       