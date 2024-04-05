import os
import json
import rasterio
import multiprocessing
import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from .cmaas_types import CMAAS_Map, CMAAS_MapMetadata, Layout, Legend, GeoReference, MapUnit, MapUnitType, OCRText, TextUnit
from rasterio.crs import CRS
from rasterio.control import GroundControlPoint


#region Legend
def loadLegendJson(filepath:Path, type_filter:MapUnitType=MapUnitType.ALL()) -> Legend:
    with open(filepath, 'r') as fh:
        json_data = json.load(fh)
    if json_data['version'] in ['5.0.1', '5.0.2']:
        legend = _loadUSGSLegendJson(filepath, type_filter)
    else:
        legend = _loadMULELegend(filepath, type_filter)
    return legend


def _loadUSGSLegendJson(filepath:Path, type_filter:MapUnitType=MapUnitType.ALL()) -> Legend:
    with open(filepath, 'r') as fh:
        json_data = json.load(fh)
    legend = Legend(provenance='USGS')
    for m in json_data['shapes']:
        # Filter out unwanted map unit types
        unit_type = MapUnitType.from_str(m['label'].split('_')[-1])
        if unit_type not in type_filter:
            continue
        # Remove type encoding from label
        unit_label = m['label']
        unit_label = ' '.join(unit_label.split('_'))
        if unit_type != MapUnitType.UNKNOWN:
            unit_label = ' '.join(unit_label.split(' ')[:-1])

        legend.features.append(MapUnit(label=unit_label, type=unit_type, bbox=np.array(m['points']).astype(int), provenance='USGS'))
    return legend

def _loadMULELegend(filepath:Path, type_filter:MapUnitType=MapUnitType.ALL()) -> Legend:
    with open(filepath, 'r') as fh:
        json_data = json.load(fh)
    legend = Legend(provenance='MULE')
    # TODO - Implement MULE legend loading
    raise NotImplementedError('MULE legend loading not yet implemented')
    return legend

def saveMULELegend(filepath:Path, legend:Legend):
    raise NotImplementedError('MULE legend saving not yet implemented')

def parallelLoadLegends(filepaths, type_filter:MapUnitType=MapUnitType.ALL(), threads:int=32):
    with ThreadPoolExecutor(max_workers=threads) as executor:
        legends = {}
        for filepath in filepaths:
            map_name = os.path.basename(os.path.splitext(filepath)[0])
            legends[map_name] = executor.submit(loadLegendJson, filepath, type_filter).result()
    return legends
# endregion Legend

# region Layout
def loadLayoutJson(filepath:Path) -> Layout:
    with open(filepath, 'r') as fh:
        layout_version = 1
        try:
            for line in fh:
                json_data = json.loads(line)
                if json_data['name'] == 'segmentation':
                    layout_version = 2
                break
        except:
            layout_version = 1
            pass
        if layout_version == 1:
            layout = _loadUnchartedLayoutv1Json(filepath)
        else:
            layout = _loadUnchartedLayoutv2Json(filepath)
    return layout

def _loadUnchartedLayoutv1Json(filepath:Path) -> Layout:
    layout = Layout()
    layout.provenance = 'Uncharted'
    with open(filepath, 'r') as fh:
        json_data = json.load(fh)

    for section in json_data:
        bounds = np.array(section['bounds']).astype(int)
        if section['name'] == 'legend_points_lines': # Intentionally not using elif so can be overwritten
            layout.point_legend = bounds
            layout.line_legend = bounds
        if section['name'] == 'map':
            layout.map = bounds
        elif section['name'] == 'correlation_diagram':
            layout.correlation_diagram = bounds
        elif section['name'] == 'cross_section':
            layout.cross_section = bounds
        elif section['name'] == 'legend_points':
            layout.point_legend = bounds
        elif section['name'] == 'legend_lines':
            layout.line_legend = bounds
        elif section['name'] == 'legend_polygons':
            layout.polygon_legend = bounds
    return layout

def _loadUnchartedLayoutv2Json(filepath:Path) -> Layout:
    layout = Layout()
    layout.provenance = 'Uncharted'
    with open(filepath, 'r') as fh:
        for line in fh:
            json_data = json.loads(line)
            section_name = json_data['model']['field']
            bounds = np.array(json_data['bounds']).astype(int)
            if section_name == 'legend_points_lines': # Intentionally not using elif so can be overwritten
                layout.point_legend = bounds
                layout.line_legend = bounds
            if section_name == 'map':
                layout.map = bounds
            elif section_name == 'correlation_diagram':
                layout.correlation_diagram = bounds
            elif section_name == 'cross_section':
                layout.cross_section = bounds
            elif section_name == 'legend_points':
                layout.point_legend = bounds
            elif section_name == 'legend_lines':
                layout.line_legend = bounds
            elif section_name == 'legend_polygons':
                layout.polygon_legend = bounds
    return layout

def saveMULELayout(filepath:Path, layout:Layout):
    # TODO
    raise NotImplementedError('MULE layout saving not yet implemented')

def parallelLoadLayouts(filepaths, threads:int=32):
    with ThreadPoolExecutor(max_workers=threads) as executor:
        layouts = {}
        for filepath in filepaths:
            map_name = os.path.basename(os.path.splitext(filepath)[0])
            layouts[map_name] = executor.submit(loadLayoutJson, filepath).result()
    return layouts
# endregion Layout

# region GeoReference
# TODO
def loadGeoReference(filepath:Path) -> GeoReference:
    # TODO
    raise NotImplementedError('GeoReference loading not yet implemented')

def saveMULEGeoReference(filepath:Path, georef:GeoReference):
    # TODO
    raise NotImplementedError('MULE georef saving not yet implemented')
# endregion GeoReference

# region Map Metadata
# TODO
def loadMapMetadata(filepath:Path):
    # TODO
    raise NotImplementedError('Map metadata loading not yet implemented')

def saveMULEMapMetadata(filepath:Path, metadata):
    # TODO
    raise NotImplementedError('MULE metadata saving not yet implemented')
# endregion Map Metadata

# region GeoTiff
def loadGeoTiff(filepath:Path):
    """Load a GeoTiff file. Image is in CHW format. Raises exception if image is not loaded properly. Returns a tuple of the image, crs and transform """
    with rasterio.open(filepath) as fh:
        image = fh.read()
        crs = fh.crs
        transform = fh.transform
    if image is None:
        msg = f'Unknown issue caused "{filepath}" to fail while loading'
        raise Exception(msg)
    
    return image, crs, transform

def saveGeoTiff(filename, image, crs, transform):
    image = np.array(image[...], ndmin=3)
    with rasterio.open(filename, 'w', driver='GTiff', compress='lzw', height=image.shape[1], width=image.shape[2],
                       count=image.shape[0], dtype=image.dtype, crs=crs, transform=transform) as fh:
        fh.write(image)

def parallelLoadGeoTiffs(filepaths, processes=multiprocessing.cpu_count()): # -> list[tuple(image, crs, transfrom)]:
    """Load a list of filenames in parallel with N processes. Returns a list of images"""
    p=multiprocessing.Pool(processes=processes)
    with multiprocessing.Pool(processes) as p:
        images = p.map(loadGeoTiff, filepaths)

    return images
# endregion GeoTiff

# region CMAAS Map IO
def loadCMAASMapFromFiles(image_path:Path, legend_path:Path=None, layout_path:Path=None, georef_path:Path=None, metadata_path:Path=None) -> CMAAS_Map:
    """Loads a CMAAS Map from its individual file components. Returns a CMAAS_Map object."""
    map_name = os.path.basename(os.path.splitext(image_path)[0])

    # Start Threads
    with ThreadPoolExecutor() as executor:
        img_future = executor.submit(loadGeoTiff, image_path)
        if legend_path is not None:
            lgd_future = executor.submit(loadLegendJson, legend_path)
        if layout_path is not None:
            lay_future = executor.submit(loadLayoutJson, layout_path)
        ### Not implemented yet
        # if georef_path is not None:
        #     gr_future = executor.submit(loadGeoReference, georef_path)
        # if metadata_path is not None:
        #     md_future = executor.submit(loadMapMetadata, metadata_path)
        
        image, crs, transform = img_future.result()
        if legend_path is not None:
            legend = lgd_future.result()
        if layout_path is not None:
            layout = lay_future.result()
        ### Not implemented yet
        # if georef_path is not None:
        #     georef = gr_future.result()
        # else:
        georef = GeoReference(crs=crs, transform=transform, provenance='GeoTIFF')
        # if metadata_path is not None:
        #     metadata = md_future.result()
    
    map_data = CMAAS_Map(map_name, image, georef=georef)
    if legend_path is not None:
        map_data.legend = legend
    if layout_path is not None:
        map_data.layout = layout
    ### Not implemented yet
    # if metadata_path is not None:
    #     map_data.metadata = metadata

    return map_data

def _load_metadata_section(json_data):
    # Check section was in json
    if 'metadata' not in json_data:
        return None
    # Load individual fields
    provenance = json_data['metadata']['provenance'] if 'provenance' in json_data['metadata'] else None
    title = json_data['metadata']['title'] if 'title' in json_data['metadata'] else None
    authors = json_data['metadata']['authors'] if 'authors' in json_data['metadata'] else None
    year = int(json_data['metadata']['year']) if 'year' in json_data['metadata'] else None
    publisher = json_data['metadata']['publisher'] if 'publisher' in json_data['metadata'] else None
    organization = json_data['metadata']['organization'] if 'organization' in json_data['metadata'] else None
    source_url = json_data['metadata']['source_url'] if 'source_url' in json_data['metadata'] else None
    url = json_data['metadata']['url'] if 'url' in json_data['metadata'] else None
    scale = json_data['metadata']['scale'] if 'scale' in json_data['metadata'] else None
    map_shape = json_data['metadata']['map_shape'] if 'map_shape' in json_data['metadata'] else None
    map_color = json_data['metadata']['map_color'] if 'map_color' in json_data['metadata'] else None
    physiographic_region = json_data['metadata']['physiographic_region'] if 'physiographic_region' in json_data['metadata'] else None

    return CMAAS_MapMetadata(provenance=provenance, title=title, authors=authors, year=year, publisher=publisher, organization=organization, source_url=source_url, url=url, scale=scale, map_shape=map_shape, map_color=map_color, physiographic_region=physiographic_region)

def _load_layout_section(json_data):
    # Check section was in json
    if 'layout' not in json_data:
        return None
    # Load individual fields
    provenance = json_data['layout']['provenance'] if 'provenance' in json_data['layout'] else None
    map_contour = np.array(json_data['layout']['map']).astype(int) if 'map' in json_data['layout'] else None
    point_legend = np.array(json_data['layout']['point_legend']).astype(int) if 'point_legend' in json_data['layout'] else None
    line_legend = np.array(json_data['layout']['line_legend']).astype(int) if 'line_legend' in json_data['layout'] else None
    polygon_legend = np.array(json_data['layout']['polygon_legend']).astype(int) if 'polygon_legend' in json_data['layout'] else None
    correlation_diagram = np.array(json_data['layout']['correlation_diagram']).astype(int) if 'correlation_diagram' in json_data['layout'] else None
    cross_section = np.array(json_data['layout']['cross_section']).astype(int) if 'cross_section' in json_data['layout'] else None
    return Layout(provenance=provenance, map=map_contour, point_legend=point_legend, line_legend=line_legend, polygon_legend=polygon_legend, correlation_diagram=correlation_diagram, cross_section=cross_section)

def _load_mapunit(json_data):
    unit_type = MapUnitType.from_str(json_data['type']) if 'type' in json_data else MapUnitType.UNKNOWN
    label = json_data['label'] if 'label' in json_data else None
    abbreviation = json_data['abbreviation'] if 'abbreviation' in json_data else None
    description = json_data['description'] if 'description' in json_data else None
    color = json_data['color'] if 'color' in json_data else None
    pattern = bool(json_data['pattern']) if 'pattern' in json_data and json_data['pattern'] is not None else None
    overlay = bool(json_data['overlay']) if 'overlay' in json_data and json_data['overlay'] is not None else None
    bbox = np.array(json_data['bbox']).astype(int) if 'bbox' in json_data else None
    return MapUnit(label=label, type=unit_type, abbreviation=abbreviation, description=description, color=color, pattern=pattern, overlay=overlay, bbox=bbox)

def _load_legend_section(json_data):
    # Check section was in json
    if 'legend' not in json_data or 'features' not in json_data['legend']:
        return None
    # Load individual fields
    provenance = json_data['legend']['provenance'] if 'provenance' in json_data['legend'] else None
    features = []
    for f in json_data['legend']['features']:
        features.append(_load_mapunit(f))
    return Legend(provenance=provenance, features=features)

def _load_georeference_section(json_data):
    # Check section was in json
    if 'georeference' not in json_data:
        return None
    georefs = []
    for gr in json_data['georeference']:
        # Load individual fields
        provenance = gr['provenance'] if 'provenance' in gr else None
        crs = CRS.from_string(gr['crs']) if 'crs' in gr else None
        transform = np.array(gr['transform']).astype(float) if 'transform' in gr else None
        gpcs = []
        for gpc in gr['gcps']:
            gpcs.append(GroundControlPoint(gpc['pixel_x'], gpc['pixel_y'], gpc['ground_lat'], gpc['ground_long']))
        confidence = float(gr['confidence']) if 'confidence' in gr else None
        georefs.append(GeoReference(provenance=provenance, crs=crs, transform=transform, gcps=gpcs, confidence=confidence))
    return georefs

def _load_textunit(json_data):
    label = json_data['label'] if 'label' in json_data else None
    geometry = np.array(json_data['geometry']).astype(int) if 'geometry' in json_data else None
    confidence = json_data['confidence'] if 'confidence' in json_data else None
    return TextUnit(label=label, geometry=geometry, confidence=confidence)

def _load_ocrtext_section(json_data):
    # Check section was in json
    if 'ocrtext' not in json_data:
        return None
    # Load individual fields
    provenance = json_data['ocrtext']['provenance'] if 'provenance' in json_data['ocrtext'] else None
    features = []
    for f in json_data['ocrtext']['features']:
        features.append(_load_textunit(f))
    return OCRText(provenance=provenance, features=features)

def loadCMAASMapJson(filepath:Path):
    """Loads a CMAAS Map from a pre-segmentation json file. Returns a CMAAS_Map object."""
    with open(filepath, 'r') as fh:
        json_data = json.load(fh)
    # Check version number
    valid_json_versions = ['0.1']
    if json_data['version'] not in valid_json_versions:
        raise ValueError(f'Invalid CMAAS Pre-Segmentation Json version: {json_data["version"]}, valid versions are: {valid_json_versions}')
    # Load individual fields
    metadata = _load_metadata_section(json_data)
    layout = _load_layout_section(json_data)
    legend = _load_legend_section(json_data)
    georef = _load_georeference_section(json_data)
    ocrtext = _load_ocrtext_section(json_data)
    return CMAAS_Map(json_data['map_name'], metadata=metadata, layout=layout, legend=legend, georef=georef, ocrtext=ocrtext)

def loadCMAASMap(json_path:Path, image_path:Path=None) -> CMAAS_Map:
    """Loads a CMAAS Map from a pre-segmentation json file. Optionally loads image from a GeoTiff file. If a GeoTiff has georeferencing built in will default to use that instead. Returns a CMAAS_Map object."""
    with ThreadPoolExecutor() as executor:
        if image_path is not None:
            img_future = executor.submit(loadGeoTiff, image_path)
        mule_future = executor.submit(loadCMAASMapJson, json_path)

        map_data = mule_future.result()
        if image_path is not None:
            image, crs, transform = img_future.result()
            map_data.image = image
            # Add GeoTiff Georeferencing at front of list. This will be the default if it exists
            if crs is not None and transform is not None:
                map_data.georef.insert(0, GeoReference(provenance='GeoTIFF', crs=crs, transform=transform, confidence=1.0))
    return map_data

def saveCMASSMap(filepath, map_data:CMAAS_Map):
    if os.path.splitext(filepath)[1] not in ['.json', '.geojson']:
        filepath = f'{filepath}.json'
    with open(filepath, 'w') as fh:
        json.dump(map_data.to_dict(), fh)

def parallelLoadCMASSMapFromFiles(map_files, legend_path=None, layout_path=None, processes : int=multiprocessing.cpu_count()):
    """Load a list of maps in parallel with N processes. Returns a list of CMASS_Map objects"""
    # Build argument list
    map_args = []
    for file in map_files:
        map_name = os.path.basename(os.path.splitext(file)[0])
        lgd_file = None
        if legend_path is not None:
            lgd_file = os.path.join(legend_path, f'{map_name}.json')
            if not os.path.exists(lgd_file):
                lgd_file = None
        lay_file = None
        if layout_path is not None:
            lay_file = os.path.join(layout_path, f'{map_name}.json')
            if not os.path.exists(lay_file):
                lay_file = None
        map_args.append((file, lgd_file, lay_file))

    # Load all files in parallel
    with multiprocessing.Pool(processes) as p:
        results = p.starmap(loadCMAASMapFromFiles, map_args)

    return results
# endregion CMAAS Map IO