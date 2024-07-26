import os
import cv2
from tests.utilities import init_test_log
from src.visualization import viz_map_unit, viz_yolo_annotation
import cmaas_utils.io as io

class TestVizMapUnit:
    def test_viz_map_unit(self):
        log = init_test_log('TestVizMapUnit/test_viz_map_unit')
        test_patch = 'tests/data/patches/MN_18_18'
        log.info(f'Testing visualization of patch {os.path.basename(test_patch)}')

        # Load data
        image = cv2.imread(f'{test_patch}.png')
        legend = io.loadLegendJson(f'{test_patch}.json')

        # Visualize data
        for map_unit in legend.features:
            viz_map_unit(image, map_unit, thickness=3)

        cv2.imwrite('tests/logs/TestVizMapUnit/test_viz_map_unit.png', image)
        log.info('Test passed successfully')

class TestVizYoloAnnotation:
    def test_viz_yolo_annotation(self):
        log = init_test_log('TestVizYoloAnnotation/test_viz_yolo_annotation')
        test_patch = 'tests/data/patches/MN_18_18'
        log.info(f'Testing visualization of patch {os.path.basename(test_patch)}')

        # Load data
        image = cv2.imread(f'{test_patch}.png')
        yolo_annotation = []
        with open(f'{test_patch}.txt') as fh:
            for line in fh:
                annotation = [float(x) for x in line.split()]
                annotation[0] = int(annotation[0])
                yolo_annotation.append(annotation)
                
        # Visualize data
        log.info(f'Visualizing annotation :\n{yolo_annotation}')
        for annotation in yolo_annotation:
            viz_yolo_annotation(image, annotation, thickness=3)
        cv2.imwrite('tests/logs/TestVizYoloAnnotation/test_viz_yolo_annotation.png', image)
        
        log.info('Test passed successfully')