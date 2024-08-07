import os
import logging
import argparse

LOGGER_NAME = 'YoloInference'
FILE_LOG_LEVEL = logging.DEBUG
STREAM_LOG_LEVEL = logging.DEBUG

def parse_command_line():
    from typing import List
    def parse_directory(path : str) -> str:
        """Command line argument parser for directory path. Checks that the path exists and is a valid directory"""
        # Check if it exists
        if not os.path.exists(path):
            msg = f'Invalid path "{path}" specified : Path does not exist\n'
            raise argparse.ArgumentTypeError(msg)
        # Check if its a directory
        if not os.path.isdir(path):
            msg = f'Invalid path "{path}" specified : Path is not a directory\n'
            raise argparse.ArgumentTypeError(msg)
        return path
    
    def parse_data(path: str) -> List[str]:
        """Command line argument parser for data path. Accepts a single file or directory name or a list of files as an input. Will return the list of valid files"""
        # Check if it exists
        if not os.path.exists(path):
            msg = f'Invalid path "{path}" specified : Path does not exist'
            #log.warning(msg)
            return None
            #raise argparse.ArgumentTypeError(msg+'\n')
        # Check if its a directory
        if os.path.isdir(path):
            data_files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.tif')]
            #if len(data_files) == 0:
                #log.warning(f'Invalid path "{path}" specified : Directory does not contain any .tif files')
        if os.path.isfile(path):
            data_files = [path]
        return data_files
    
    def post_parse_data(data):
        data_files = [file for sublist in data if sublist is not None for file in sublist]
        if len(data_files) == 0:
            msg = f'No valid files where given to --data argument. --data should be given a path or paths to file(s) and/or directory(s) containing the data to perform inference on. program will only run on .tif files'
            raise argparse.ArgumentTypeError(msg)
        return data_files
    
    parser = argparse.ArgumentParser(description='', add_help=False)
    # Required Arguments
    required_args = parser.add_argument_group('required arguments', 'These are the arguments the pipeline requires to run, --amqp and --data are used to specify what data source to use and are mutually exclusive.')
    required_args.add_argument('--data',
                            type=parse_data,
                            nargs='+',
                            help='Path to the data to perform inference on. Can be a single file or directory containing multiple files.')
    # Optional Arguments
    optional_args = parser.add_argument_group('optional arguments', '')
    optional_args.add_argument('--layout',
                            type=parse_directory,
                            help='Path to directory containing area_segmentation data.')
    optional_args.add_argument('--output',
                            default='output',
                            help='Path to the output directory to save the results to.')
    optional_args.add_argument('--checkpoint',
                            default='cma_150.pt',
                            help='Path to a pretrained checkpoint file to load the model from.')
    optional_args.add_argument('--log',
                            default='logs/Latest.log',
                            help='Option to set the file logging will output to. Defaults to "logs/Latest.log"')
    # Flags
    flag_group = parser.add_argument_group('Flags', '')
    flag_group.add_argument('-h', '--help',
                            action='help', 
                            help='show this message and exit')

    args = parser.parse_args()
    args.data = post_parse_data(args.data)
    return args


def main(args):
    global log
    log = start_logger(LOGGER_NAME, args.log, log_level=FILE_LOG_LEVEL, console_log_level=STREAM_LOG_LEVEL, writemode='w')

    # Log Run parameters
    log.info(f'Performing inference on {os.uname()[1]} with following parameters:\n' +
            f'\tData : {args.data}\n' +
            f'\tLayout : {args.layout}\n' +
            f'\tCheckpoint : {args.checkpoint}\n' +
            f'\tOutput : {args.output}')
    
    os.makedirs(args.output, exist_ok=True)

    interface = YoloInterface(args.checkpoint)
    classes = [MapUnitType.POINT, MapUnitType.LINE, MapUnitType.POLYGON]
    for file in args.data:
        map_name = os.path.splitext(os.path.basename(file))[0]
        log.info(f'Performing inference on {map_name}')
        image, _, _ = io.loadGeoTiff(file)

        # Load layout
        if args.layout is not None:
            layout_path = os.path.join(args.layout, map_name + '.json')
            if os.path.exists(layout_path):
                layout = io.loadLayoutJson(layout_path)
                legend_areas = [layout.point_legend, layout.line_legend, layout.polygon_legend]
        else:
            legend_areas = None

        # Perform inference
        prediction = interface.inference(image, legend_areas)
        log.info(f'Inference complete on {file}')

        # Visualize the results
        viz_image = image.transpose(1,2,0).copy()
        log.debug(f'Viz image shape : {viz_image.shape}')
        log.info(f'Visualizing the results for {file}')
        for predict in prediction:
            unit = MapUnit(type=classes[int(predict[2])], bounding_box=[predict[0][:2], predict[0][2:]])
            viz_map_unit(viz_image, unit)
        
        # Save the results
        log.info(f'Saving the results for {file}')
        pil_image = Image.fromarray(viz_image)
        filename = os.path.join(args.output, os.path.splitext(os.path.basename(file))[0] + '.png')
        pil_image.save(filename)
        break


if __name__=='__main__':
    args = parse_command_line()
    import cmaas_utils.io as io
    from cmaas_utils.logging import start_logger
    from src.yolo_interface import YoloInterface
    from cmaas_utils.types import MapUnit, MapUnitType
    from src.visualization import viz_map_unit
    from PIL import Image
    main(args)

# if __name__=='__ipython__':
import cmaas_utils.io as io
from src.yolo_interface import YoloInterface
image, _, _ = io.loadGeoTiff('tests/uncommited_data/cma_sample/validation/AR_StJoe.tif')
interface = YoloInterface('cma_150.pt')
layout = io.loadLayoutJson('../data/validation/uncharted_masks/AR_StJoe.json')
legend_areas = [layout.point_legend, layout.line_legend, layout.polygon_legend]
prediction = interface.inference(image, legend_areas)
