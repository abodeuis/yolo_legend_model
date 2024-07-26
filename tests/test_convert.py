import os
import cv2

import cmaas_utils.io as io
from cmaas_utils.types import MapUnitType

from tests.utilities import init_test_log
from src.visualization import viz_yolo_annotation
from src.convert import convert_pixel_box_to_yolo_box, convert_cmass_file_to_yolo, convert_cmass_dataset_to_yolo

class TestConvertToYolo:
    def test_convert_pixel_box_to_yolo_box(self):
        log = init_test_log('TestConvertToYolo/test_convert_pixel_box_to_yolo_box')
        log.info('Testing conversion of pixel box to yolo box')
        test_patch = 'tests/data/patches/MN_18_18'

        # Load data
        image = cv2.imread(f'{test_patch}.png')
        full_legend = io.loadLegendJson('tests/data/val/labels/MN.json')
        test_map_units = ['Asc','Aks','Akc','Akv','Ams']
        expected_annotations = []
        with open(f'{test_patch}.txt') as fh:
            for line in fh:
                annotation = [float(x) for x in line.split()]
                annotation[0] = int(annotation[0])
                expected_annotations.append(annotation)

        patch_size = 512
        patch_overlap = 32
        row = 18
        col = 18

        # Test converstion
        for map_unit in full_legend.features:
            if map_unit.label in test_map_units:
                yolo_annotation = convert_pixel_box_to_yolo_box(map_unit.bounding_box, row, col, patch_size, patch_overlap)
                log.info(f'Converted pixel box {map_unit.bounding_box} to yolo box {yolo_annotation}')
                #assert yolo_box in expected_annotations
                viz_yolo_annotation(image, [2,*yolo_annotation], thickness=2)
                
        cv2.imwrite('tests/logs/TestConvertToYolo/test_convert_pixel_box_to_yolo_box.png', image)

        log.info('Test passed successfully')
    
    def test_convert_cmass_file_to_yolo(self):
        log = init_test_log('TestConvertToYolo/test_convert_cmass_file_to_yolo')
        log.info('Testing conversion of cmass file to yolo format')
        test_image = 'tests/data/val/images/MN.tif'
        test_label = 'tests/data/val/labels/MN.json'
        yolo_image_dir = 'tests/logs/TestConvertToYolo/yolo_images'
        yolo_label_dir = 'tests/logs/TestConvertToYolo/yolo_labels'
        patch_size = 512
        patch_overlap = 32
        classes = [MapUnitType.POINT, MapUnitType.LINE, MapUnitType.POLYGON]

        # Test conversion
        convert_cmass_file_to_yolo(test_image, test_label, yolo_image_dir, yolo_label_dir, classes, patch_size, patch_overlap)

        log.info('Test passed successfully')

    def test_convert_cmass_dataset_to_yolo(self):
        log = init_test_log('TestConvertToYolo/test_convert_cmass_dataset_to_yolo')
        log.info('Testing conversion of cmass dataset to yolo format')
        train_dir = 'tests/uncommited_data/cma_sample/training'
        val_dir = 'tests/uncommited_data/cma_sample/validation'
        test_dir = 'tests/uncommited_data/cma_sample/final_evaluation'
        train_files = [os.path.join(train_dir,f) for f in os.listdir(train_dir) if f.endswith('.json')]
        val_files = [os.path.join(val_dir,f) for f in os.listdir(val_dir) if f.endswith('.json')]
        test_files = [os.path.join(test_dir,f) for f in os.listdir(test_dir) if f.endswith('.json')]
        output_dir = 'tests/logs/TestConvertToYolo/cma_sample'
        patch_size = 512
        patch_overlap = 32
        classes = [MapUnitType.POINT, MapUnitType.LINE, MapUnitType.POLYGON]

        # Test conversion
        convert_cmass_dataset_to_yolo(train_files, val_files, test_files, output_dir, classes, patch_size, patch_overlap)

        log.info('Test passed successfully')

