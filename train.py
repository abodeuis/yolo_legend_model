import os
import logging
import argparse

LOGGER_NAME = 'YoloTraining'
FILE_LOG_LEVEL = logging.DEBUG
STREAM_LOG_LEVEL = logging.INFO

def parse_command_line():
    from typing import List
    from pathlib import Path
    def parse_usgs_data(path: Path) -> Path:
        """
        Command line argument parser for checking usgs data is correctly formatted. Raises argument error if the path
        does not exist or if it is not a valid directory strucutre. Returns the path if valid.
        """
        # Check path exists
        if not os.path.exists(path):
            msg = f'Invalid path "{path}" specified : Path does not exist'
            raise argparse.ArgumentTypeError(msg)
        # Check path is a directory
        if not os.path.isdir(path):
            msg = f'Invalid path "{path}" specified : Path is not a directory\n'
            raise argparse.ArgumentTypeError(msg)
        return path

    def parse_yolo_data(path: Path) -> Path:
        """
        Command line argument parser for checking yolo data is correctly formatted. Raises argument error if the path
        does not exist or if the yaml file is invalid. Returns the path if valid.
        """
        # Check path exists
        if not os.path.exists(path):
            msg = f'Invalid path "{path}" specified : Path does not exist'
            raise argparse.ArgumentTypeError(msg)
        # Check path is a yaml file
        if not path.endswith('.yaml'):
            msg = f'Invalid file "{path}" specified : File is not a yaml config\n'
            raise argparse.ArgumentTypeError(msg)
        return os.path.abspath(path)

    parser = argparse.ArgumentParser(description='', add_help=False)
    # Required Arguments
    required_args = parser.add_argument_group('required arguments', 'Data formats are mutually exclusive.')
    data_group = required_args.add_mutually_exclusive_group(required=True)
    data_group.add_argument('--usgs_data',
                            type=parse_usgs_data,
                            help='Path to directory containing the data for training formatted in train, val and test directories with images and labels subdirectories.')
    data_group.add_argument('--yolo_data',
                            type=parse_yolo_data,
                            help='Path to a yolo formatted yaml config file.')
    # Optional Arguments
    optional_args = parser.add_argument_group('optional arguments', '')
    optional_args.add_argument('--output',
                            default='datasets/yolo',
                            help='Path to the output directory to save the model to.')
    optional_args.add_argument('--checkpoint',
                            default='yolov8n.pt',
                            help='Path to a pretrained checkpoint file to load the model from.')
    optional_args.add_argument('--patch_size',
                            type=int,
                            default=512,
                            help='Size of the patches to train on.')
    optional_args.add_argument('--patch_overlap',
                            type=int,
                            default=32,
                            help='Overlap of the patches to train on.')
    optional_args.add_argument('--epochs',
                            type=int,
                            default=10,
                            help='Number of epochs to train the model for.')
    optional_args.add_argument('--log',
                            default='logs/Latest.log',
                            help='Option to set the file logging will output to. Defaults to "logs/Latest.log"')
    # Flags
    flag_group = parser.add_argument_group('Flags', '')
    flag_group.add_argument('-h', '--help',
                            action='help', 
                            help='show this message and exit')
    flag_group.add_argument('--cache_data',
                            action='store_true',
                            help='Cache the data for faster training.')
    return parser.parse_args()
    
def main(args):
    global log
    log = start_logger(LOGGER_NAME, args.log, log_level=FILE_LOG_LEVEL, console_log_level=STREAM_LOG_LEVEL, writemode='w')

    # Log Run parameters
    if args.usgs_data:
        log_data_mode = 'USGS'
        log_data_source = f'\tData : {args.usgs_data}\n'
    else:
        log_data_mode = 'YOLO'
        log_data_source = f'\tData : {args.yolo_data}\n'
        
    log.info(f'Training on {os.uname()[1]} with {log_data_mode} data with following parameters:\n' +
            log_data_source +
            f'\tBase Model : {args.checkpoint}\n' + 
            f'\tEpochs : {args.epochs}\n' +
            f'\tPatch Size : {args.patch_size}\n' +
            f'\tPatch Overlap : {args.patch_overlap}\n' +
            f'\tOutput : {args.output}\n')

    # Load a model
    model = YOLO(args.checkpoint)
    classes = [MapUnitType.POINT, MapUnitType.LINE, MapUnitType.POLYGON]

    # Convert data to yolo
    if args.usgs_data:
        train_files = [os.path.join(args.usgs_data, 'train', f) for f in os.listdir(os.path.join(args.usgs_data, 'train')) if f.endswith('.json')]
        val_files = [os.path.join(args.usgs_data, 'val', f) for f in os.listdir(os.path.join(args.usgs_data, 'val')) if f.endswith('.json')]
        test_files = [os.path.join(args.usgs_data, 'test', f) for f in os.listdir(os.path.join(args.usgs_data, 'test')) if f.endswith('.json')]
        yaml_path = convert_cmass_dataset_to_yolo(train_files, val_files, test_files, args.output, classes, args.patch_size, args.patch_overlap)
    if args.yolo_data:
        yaml_path = args.yolo_data

    # Train the model
    train_results = model.train(data=yaml_path, epochs=args.epochs, imgsz=args.patch_size)
    val_results = model.val()
    success = model.export(format='onnx')

if __name__ == '__main__':
    args = parse_command_line()
    from src.convert import convert_cmass_dataset_to_yolo
    from ultralytics import YOLO
    from cmaas_utils.logging import start_logger
    from cmaas_utils.types import MapUnitType
    main(args)
