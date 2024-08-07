import os
from src.interface import YoloInterface, _convert_cmass_file_to_yolo, convert_pixel_box_to_yolo_box, is_intersecting
from tests.utilities import init_test_log
from cmaas_utils.types import MapUnitType, MapUnit
import cmaas_utils.io as io
import cv2


class TestYoloInterface:
    def test_constructor(self):
        log = init_test_log('TestYoloInterface/test_constructor')
        yolo_interface = YoloInterface()
        assert yolo_interface

        yolo_interface = YoloInterface('yolov8n.pt')
        assert yolo_interface.model
        log.info("Test passed successfully")

    def test_load_from_checkpoint(self):
        log = init_test_log('TestYoloInterface/test_load_from_checkpoint')
        yolo_interface = YoloInterface()
        yolo_interface.load_from_checkpoint('yolov8n.pt')
        assert yolo_interface.model
        log.info("Test passed successfully")

    def test_create_yaml_config(self):
        log = init_test_log('TestYoloInterface/test_create_yaml_config')
        root_path = 'tests/logs/TestYoloInterface/'
        yolo_interface = YoloInterface('yolov8n.pt')
        yaml_config = yolo_interface._create_yaml_config(root_path, 'train', 'val', 'test', filename='test_config.yaml')
        assert os.path.exists(yaml_config)
        log.info("Test passed successfully")

    def test_format_training_file(self):
        log = init_test_log('TestYoloInterface/test_format_training_file')
        # Parameters
        test_image = 'tests/data/val/images/AR_StJoe.tif'
        test_label = 'tests/data/val/labels/AR_StJoe.json'
        image_dir = 'tests/logs/TestYoloInterface/test_format_training_file/images'
        label_dir = 'tests/logs/TestYoloInterface/test_format_training_file/labels'
        patch_size = 512
        patch_overlap = 32
        classes = [MapUnitType.POINT, MapUnitType.LINE, MapUnitType.POLYGON, MapUnitType.UNKNOWN]

        log.info("Deleting old test files")
        os.system('rm -rf tests/logs/TestYoloInterface/test_format_training_file')
        os.makedirs(image_dir, exist_ok=True)
        os.makedirs(label_dir, exist_ok=True)
        _convert_cmass_file_to_yolo(test_image, test_label, image_dir, label_dir, classes, patch_size, patch_overlap)
        
        log.info("Test passed successfully")
    
    # def test_progress_wrapper(self):
    #     log = init_test_log('TestYoloInterface/test_progress_wrapper')
    #     def foo(a,b=2):
    #         return a+b
    #     out = _progress_wrapper(foo, 1, b=4)
    #     assert out == 5

    #     with Progress() as progress:
    #         task = progress.add_task("Train", total=1)
    #         out2 = _progress_wrapper(foo, 1, b=3, progress=progress, task=task)
    #     assert out2 == 4
    #     log.info("Test passed successfully")

    def test_format_training_data(self):
        log = init_test_log('TestYoloInterface/test_format_training_data')
        yolo_interface = YoloInterface('yolov8n.pt')
        yolo_interface.convert_cmass_dataset_to_yolo('tests/data', 'tests/logs/TestYoloInterface/test_format_training_data')
        log.info("Test passed successfully")

    def test_train(self):
        log = init_test_log('TestYoloInterface/test_train')
        yolo_interface = YoloInterface()
        yolo_interface.train('tests/data', epochs=1, cache_data=True)
        log.info("Test passed successfully")

    # @fixture?
    # patch_size_parameters = [333,512]
    # patch_overlap_parameters = [0,9,32]

#     def test_viz_yolo_annotation(self):
#         log = init_test_log('TestYoloInterface/test_viz_yolo_annotation')
#         image, _, _ = io.loadGeoTiff('tests/data/val/images/MN.tif')
#         legend = io.loadLegendJson('tests/data/val/labels/MN.json')
#         for map_unit in legend.features:
#             viz_yolo_annotation(image, map_unit.bounding_box)
#         log.info("Test passed successfully")

#     def test_convert_pixel_box_to_yolo_box(self, patch_size, patch_overlap):
#         log = init_test_log('TestYoloInterface/test_convert_pixel_box_to_yolo_box')
#         log.info("Running test on 5 random map units from MN json")
#         image, _, _ = io.loadGeoTiff('tests/data/val/images/MN.tif')
#         legend = io.loadLegendJson('tests/data/val/labels/MN.json')

#         # Random sample of 5 map units
#         patch_step = patch_size - patch_overlap
#         for map_unit in sample(legend.features,5):
#             row, col = map_unit.bounding_box[:2] / patch_step
            
#             # Test the bounding box converts without errors
#             yolo_box = convert_pixel_box_to_yolo_box(map_unit.bounding_box, row, col, patch_size, patch_overlap)
            
#             log.info(f"Converted {map_unit.label} - {map_unit.bounding_box} to yolo format: {yolo_box}")
#             patch = image[row*patch_step:row*patch_step+patch_size, col*patch_step:col*patch_step+patch_size]
#             viz_map_unit(patch, map_unit)
#             viz_yolo_annotation(patch, [0, *yolo_box])
#             assert yolo_box is not None

#     def test_convert_annotation_to_yolo(self, patch_size, patch_overlap):
#         log = init_test_log('TestYoloInterface/test_convert_annotation_to_yolo')
#         image, _, _ = io.loadGeoTiff('tests/data/val/images/MN.tif')
#         legend = io.loadLegendJson('tests/data/val/labels/MN.json')
#         centered_box = MapUnit(type=MapUnitType.POLYGON, label='centered_box', bounding_box=[2112])
#         edge_box = MapUnit(type=MapUnitType.POLYGON, label='edge_box', bounding_box=[2112])

        
                

        
#         # test that non-intersecting bounding boxes return nothing
#         result = convert_pixel_box_to_yolo_box(bbox, 5, 5, patch_size, patch_overlap)
#         assert 
#         MN 5,5

#         # test that intersecting bounding boxes return correct annotation
#         MN 15, 18
#         # test that intersecting bounding boxes on the patch edge return the correct annotation
#         MN 18, 18
#         convert_pixel_box_to_yolo_box()
        
#         log.info("Test passed successfully")
# from random import sample