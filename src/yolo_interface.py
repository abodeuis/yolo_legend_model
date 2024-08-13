import torch
import logging
import numpy as np
from patchify import patchify
from ultralytics import YOLO

from src.utilities import is_intersecting, mask_and_crop

log = logging.getLogger('YoloInference')

class YoloInterface:
    def __init__(self, checkpoint='cma_150.pt'):
        self._checkpoint = checkpoint
        self.model = YOLO(self._checkpoint)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = 256
        self.patch_size = 512
        self.patch_overlap = 32

    def my_norm(self, data):
        """
        Normalize data to [0,1] range
        """
        norm_data = data / 255.0
        return norm_data

    def inference(self, image, legend_area=None):
        """
        Perform inference on an image and return the predicted bounding boxes.

        Args:
            image : numpy array of shape (C,H,W)
            legend_area : list of numpy arrays of shape (N,2) where N is the number of points in the contour
                that defines the legend area. If None, the entire image is used.

        Returns:
            list of tuples of the format ([x1,y1,x2,y2], conf, cls)
        """

        # Get the size of the map
        map_channels, map_height, map_width = image.shape

        # Reshape maps with 1 channel images (greyscale) to 3 channels for inference
        if map_channels == 1: # This is tmp fix!    
            image = np.concatenate([image,image,image], axis=0)
            map_channels = 3

        # Cutout legend portion of image
        if legend_area is not None:
            image, crop_offset = mask_and_crop(image, legend_area)
        else:
            crop_offset = [0,0]

        # Generate patches
        # Pad image so we get a size that can be evenly divided into patches.
        patch_step = self.patch_size-self.patch_overlap
        right_pad = self.patch_size - (map_width % self.patch_size)
        bottom_pad = self.patch_size - (map_height % self.patch_size)
        padded_image = np.pad(image, ((0,0), (0, bottom_pad), (0, right_pad)), mode='constant', constant_values=0)
        map_patches = patchify(padded_image, (3, self.patch_size, self.patch_size), step=patch_step)

        rows = map_patches.shape[1]
        cols = map_patches.shape[2]
        log.debug(f'Patches shape: {map_patches.shape}, rows = {rows}, cols = {cols}')

        # Flatten row col dims and normalize map patches to [0,1]
        map_patches = map_patches.reshape(-1, 3, self.patch_size, self.patch_size)
        map_tensor = torch.Tensor(map_patches).to(self.device)
        norm_tensor = self.my_norm(map_tensor)

        # Run inference on patches
        prediction_patches = []
        for i in range(0, len(norm_tensor), self.batch_size):
            batch = norm_tensor[i:i+self.batch_size]
            # Run inference on batch
            gens = self.model.predict(source=batch, stream=True, verbose=False)
            for r in gens:
                prediction_patches.append(r.boxes.data.cpu().numpy())

        # Reconstruct map from patches
        box_margin = 2 # Amount of pixels to contract box borders by before merging 
        raw_boxes = [] # List of unmerged boxes in the format [([x1,y1,x2,y2], conf, cls)]
        for row in range(rows):
            for col in range(cols):
                if len(prediction_patches[row*cols+col]) > 0:
                    for predicted_box in prediction_patches[row*cols+col]:
                        x1,y1,x2,y2,conf,cls = predicted_box
                        x1 = x1 + col*patch_step + box_margin + crop_offset[0]
                        x2 = x2 + col*patch_step - box_margin + crop_offset[0]
                        y1 = y1 + row*patch_step + box_margin + crop_offset[1]
                        y2 = y2 + row*patch_step - box_margin + crop_offset[1]
                        raw_boxes.append(([x1,y1,x2,y2],conf,cls))
        
        # Merge overlapping boxes from different patches
        merged_boxes = []
        finished = False
        last_i, last_j = 0, 0
        # Need to fully restart the loop when items are removed from the list
        while not finished:
            removal = None
            for i in range(last_i, len(raw_boxes)):
                for j in range(last_j, len(raw_boxes)):
                    bbox1 = raw_boxes[i]
                    bbox2 = raw_boxes[j]
                    # Same class and not the exact same box
                    if bbox1 != bbox2 and bbox1[2] == bbox2[2]:
                        if is_intersecting(bbox1[0], bbox2[0]):
                            # Merge boxes
                            bbox1[0][0] = min(bbox1[0][0], bbox2[0][0])
                            bbox1[0][1] = min(bbox1[0][1], bbox2[0][1])
                            bbox1[0][2] = max(bbox1[0][2], bbox2[0][2])
                            bbox1[0][3] = max(bbox1[0][3], bbox2[0][3])
                            # Remove bbox2
                            removal = bbox2
                            break
                # Inner Loop check
                if removal is not None:
                    break
                else:
                    last_j = 0
                    merged_boxes.append(bbox1)
            # Outer Loop check
            if removal is not None:
                last_i = i
                last_j = j
                raw_boxes.remove(removal)
            else:
                finished = True

        # log.info(f'Prediction shape: {len(prediction[0])}, patches = {map_patches.shape[0]}')
        return merged_boxes
    