import os
from src.interface import YoloInterface, _format_training_file
from tests.utilities import init_test_log
from cmaas_utils.types import MapUnitType

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
        _format_training_file(test_image, test_label, image_dir, label_dir, classes, patch_size, patch_overlap)
        
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
        yolo_interface.format_training_data('tests/data', 'tests/logs/TestYoloInterface/test_format_training_data')
        log.info("Test passed successfully")

    def test_train(self):
        log = init_test_log('TestYoloInterface/test_train')
        yolo_interface = YoloInterface()
        yolo_interface.train('tests/data', epochs=1, cache_data=True)
        log.info("Test passed successfully")
