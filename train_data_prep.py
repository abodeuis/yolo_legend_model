import os
import logging
import argparse
from tqdm import tqdm
from PIL import Image
from patchify import patchify

import src.cmaas_io as io
from src.cmaas_logging import start_logger

LOGGER_NAME = 'train_data_prep'
FILE_LOG_LEVEL = logging.DEBUG
STREAM_LOG_LEVEL = logging.WARNING

def parse_command_line():
    from typing import List
    def parse_directory(path : str) -> str:
        """Command line argument parser for directory path arguments. Raises argument error if the path does not exist
           or if it is not a valid directory. Returns directory path"""
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
        """Command line argument parser for --data. --data should accept a list of file and/or directory paths as an
           input. This function is run called on each individual element of that list and checks if the path is valid
           and if the path is a directory expands it to all the valid files paths inside the dir. Returns a list of 
           valid files. This is intended to be used in conjunction with the post_parse_data function"""
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
    
    def post_parse_data(data : List[List[str]]) -> List[str]:
        """Cleans up the output of parse data from a list of lists to a single list and does validity checks for the 
           data as a whole. Returns a list of valid files. Raises an argument exception if no valid files were given"""
        # Check that there is at least 1 valid map to run on
        data_files = [file for sublist in data if sublist is not None for file in sublist]
        if len(data_files) == 0:
            msg = f'No valid files where given to --data argument. --data should be given a path or paths to file(s) \
                    and/or directory(s) containing the data to perform inference on. program will only run on .tif files'
            raise argparse.ArgumentTypeError(msg)
        return data_files
    parser = argparse.ArgumentParser(description='', add_help=False)
    # Required Arguments
    required_args = parser.add_argument_group('required arguments', 'These are the arguments the pipeline requires to \
                                               run, --amqp and --data are used to specify what data source to use and \
                                               are mutually exclusive.')
    required_args.add_argument('--images', 
                        type=parse_data,
                        nargs='+',
                        help='Path to directory containing the data to convert to yolo_format inference on. The \
                              program will run inference on any .tif files. ')
    required_args.add_argument('--labels',
                        type=parse_directory,
                        help='Path to directory containing the labels to convert to yolo_format inference on. The \
                              program will run inference on any .json files.')         
    # Optional Arguments
    optional_args = parser.add_argument_group('optional arguments', '')
    optional_args.add_argument('--yolo_output',
                        default='yolo_data',
                        help='Directory to write the outputs of inference to')
    optional_args.add_argument('--log',
                        default='logs/Latest.log',
                        help='Option to set the file logging will output to. Defaults to "logs/Latest.log"')
    optional_args.add_argument('--patch_size',
                        default=512,
                        type=int,
                        help='Size in pixels of the patched data. Default is 512px')
    optional_args.add_argument('--overlap',
                        default=0,
                        type=int,
                        help='Overlap in pixels of the patched data. Default is 0px')
    # Flags
    flag_group = parser.add_argument_group('Flags', '')
    flag_group.add_argument('-h', '--help',
                            action='help', 
                            help='show this message and exit')
    flag_group.add_argument('-v', '--verbose',
                            action='store_true',
                            help='Flag to change the logging level from INFO to DEBUG')
    
    args = parser.parse_args()
    args.images = post_parse_data(args.images)
    return args

def boundingBox(array):
    min_xy = [min(array, key=lambda x: (x[0]))[0], min(array, key=lambda x: (x[1]))[1]]
    max_xy = [max(array, key=lambda x: (x[0]))[0], max(array, key=lambda x: (x[1]))[1]]
    return [min_xy, max_xy]

def is_intersecting(bbox1, bbox2):
    return not (bbox1[0] > bbox2[2] or bbox1[2] < bbox2[0] or bbox1[1] > bbox2[3] or bbox1[3] < bbox2[1])

def main():
    args = parse_command_line()

    # Start logger
    if args.verbose:
        global FILE_LOG_LEVEL, STREAM_LOG_LEVEL
        FILE_LOG_LEVEL = logging.DEBUG
        STREAM_LOG_LEVEL = logging.INFO
    global log
    log = start_logger(LOGGER_NAME, args.log, log_level=FILE_LOG_LEVEL, console_log_level=STREAM_LOG_LEVEL, writemode='w')
    log.info(f'Running data prep on {os.uname()[1]} with following parameters:\n' +
            f'\tImages        : {args.images}\n' + 
            f'\tLabels        : {args.labels}\n' +
            f'\tOutput        : {args.yolo_output}\n' +
            f'\tPatch_size    : {args.patch_size}\n' +
            f'\tPatch_overlap : {args.overlap}')
    
    os.makedirs(os.path.join(args.yolo_output, 'images'), exist_ok=True)
    os.makedirs(os.path.join(args.yolo_output, 'labels'), exist_ok=True)
    pbar = tqdm(args.images)
    for file in pbar:
        filename = os.path.basename(file)
        pbar.set_description(f'Processing {filename}')
        pbar.refresh()
        log.debug(f'Processing file: {filename}')
        # Read image geo tif
        image, _, _ = io.loadGeoTiff(file)
        
        if image.shape[0] != 3:
            log.warning(f'Image {filename} does not have 3 channels. Skipping')
            continue
        badfiles = ['46_Coosa_2015_11','Trend_2005','Crystal_2018','CarlinVanadiumNI43']
        skip = False
        for badfile in badfiles:
            if badfile in filename:
                log.warning(f'I hate this file. Skipping')
                skip = True
                continue
        if skip:
            continue
        # read validation json
        lgd = io.loadLegendJson(os.path.join(args.labels, filename.replace('.tif', '.json')))
        # patch image
        patch_step = args.patch_size-args.overlap
        data_patches = patchify(image, (3, args.patch_size, args.patch_size), step=patch_step).squeeze()
        # Save annotations
        for i in range(data_patches.shape[0]):
            for j in range(data_patches.shape[1]):
                outstr = ''
                for feature in lgd.features:
                    min_xy, max_xy = boundingBox(feature.bbox)
                    if is_intersecting([*min_xy, *max_xy], [i*patch_step, j*patch_step, (i+1)*patch_step, (j+1)*patch_step]):
                        # Convert full image px coords to patchwise normalized coords (0-1)
                        feature_class = feature.type
                        x = max(0,(min_xy[0] - i*patch_step)) / args.patch_size
                        y = max(0,(min_xy[1] - j*patch_step)) / args.patch_size
                        w = min((args.patch_size - x), (max_xy[0] - min_xy[0])) / args.patch_size
                        h = min((args.patch_size - y), (max_xy[1] - min_xy[1])) / args.patch_size
                        outstr += f'{feature_class} {x} {y} {w} {h}\n'
                        
                if outstr != '':
                    annotation_filename = os.path.join(args.yolo_output, 'labels', filename.replace('.tif', f'_{i}_{j}.txt'))
                    log.debug(f'Saving {annotation_filename}')
                    with open(annotation_filename, 'w') as f:
                        f.write(outstr)
        log.debug(f'Saved annotations for {filename}')

        # Save image patches
        for i in range(data_patches.shape[0]):
            for j in range(data_patches.shape[1]):
                # Only save image patches with annotations
                if not os.path.exists(os.path.join(args.yolo_output, 'labels', filename.replace('.tif', f'_{i}_{j}.txt'))):
                    continue
                # Skip if patch already exists
                patch_filename = os.path.join(args.yolo_output, 'images', filename.replace('.tif', f'_{i}_{j}.png'))
                if os.path.exists(patch_filename):
                    continue
                log.debug(f'Saving {patch_filename}')
                pil_img = Image.fromarray(data_patches[j,i].transpose(1,2,0))
                pil_img.save(patch_filename)
        log.debug(f'Saved patches for {filename}')
        #break ## DEBUG
    log.info('Data prep complete')

if __name__=='__main__':
    main()