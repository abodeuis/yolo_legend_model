import rasterio
from rasterio.crs import CRS
from rasterio.control import GroundControlPoint
from rasterio.transform import from_gcps
from enum import Enum
from typing import List, Tuple
import numpy as np

DEBUG_MODE = True # Turns on debuging why two objects are not equal

class MapUnitType(Enum):
    POINT = 0
    LINE = 1
    POLYGON = 2
    UNKNOWN = 3
    def ALL():
        return [MapUnitType.POINT, MapUnitType.LINE, MapUnitType.POLYGON, MapUnitType.UNKNOWN]
    def ALL_KNOWN():
        return [MapUnitType.POINT, MapUnitType.LINE, MapUnitType.POLYGON]

    def from_str(feature_type:str):
        if feature_type.lower() in ['pt','point']:
            return MapUnitType.POINT
        elif feature_type.lower() in ['line']:
            return MapUnitType.LINE
        elif feature_type.lower() in ['poly','polygon']:
            return MapUnitType.POLYGON
        else:
            return MapUnitType.UNKNOWN
        
    def to_str(self):
        if self == MapUnitType.POINT:
            return '0'
        elif self == MapUnitType.LINE:
            return '1'
        elif self == MapUnitType.POLYGON:
            return '2'
        else:
            return 'unknown'

    def __str__(self) -> str:
        return self.to_str()
    
    def __repr__(self) -> str:
        repr_str = 'MapUnitType.'
        if self == MapUnitType.POINT:
            repr_str += 'POINT'
        elif self == MapUnitType.LINE:
            repr_str += 'LINE'
        elif self == MapUnitType.POLYGON:
            repr_str += 'POLYGON'
        else:
            repr_str += 'Unknown'
        return repr_str

class MapUnit():
    def __init__(self, type=MapUnitType.UNKNOWN, label=None, abbreviation=None, description=None, color=None, pattern=None, overlay=False, bbox=None, provenance=None):
        # MapUnit Legend Information
        self.type = type
        self.label = label
        self.abbreviation = abbreviation
        self.description = description
        self.color = color
        self.pattern = pattern
        self.overlay = overlay
        self.bbox = bbox
        self.provenance = provenance

        # MapUnit Map Segmentation
        self.mask = None
        self.mask_confidence = None
        self.geometry = None
    
    def to_dict(self):
        # We don't save the map segmentation data in the dictionary
        return {
            'type' : self.type.to_str() if self.type is not None else 'unknown',
            'label' : self.label,
            'abbreviation' : self.abbreviation,
            'description' : self.description,
            'color' : self.color,
            'pattern' : self.pattern,
            'overlay' : self.overlay,
            'bounding_box' : self.bbox.to_list() if self.bbox is not None else None
        }
    
    def __eq__(self, __value: object) -> bool:
        if self is None or __value is None:
            if self is None and __value is None:
                return True
            else:
                return False
            
        if self.type != __value.type:
            if DEBUG_MODE:
                print(f'Type Mismatch: {self.type} != {__value.type}')
            return False
        if self.label != __value.label:
            if DEBUG_MODE:
                print(f'Label Mismatch: {self.label} != {__value.label}')
            return False
        if self.abbreviation != __value.abbreviation:
            if DEBUG_MODE:
                print(f'Abbreviation Mismatch: {self.abbreviation} != {__value.abbreviation}')
            return False
        if self.description != __value.description:
            if DEBUG_MODE:
                print(f'Description Mismatch: {self.description} != {__value.description}')
            return False
        if self.color != __value.color:
            if DEBUG_MODE:
                print(f'Color Mismatch: {self.color} != {__value.color}')
            return False
        if self.pattern != __value.pattern:
            if DEBUG_MODE:
                print(f'Pattern Mismatch: {self.pattern} != {__value.pattern}')
            return False
        if self.overlay != __value.overlay:
            if DEBUG_MODE:
                print(f'Overlay Mismatch: {self.overlay} != {__value.overlay}')
            return False
        if isinstance(self.bbox, (np.ndarray, np.generic)) or isinstance(__value.bbox, (np.ndarray, np.generic)):
            if (self.bbox != __value.bbox).any():
                if DEBUG_MODE:
                    print(f'Bounding Box Mismatch: {self.bbox} != {__value.bbox}')
                return False
        else:
            if self.bbox != __value.bbox:
                if DEBUG_MODE:
                    print(f'Bounding Box Mismatch: {self.bbox} != {__value.bbox}')
                return False
        if self.provenance != __value.provenance:
            if DEBUG_MODE:
                print(f'Provenance Mismatch: {self.provenance} != {__value.provenance}')
            return False
        return True

    def __str__(self) -> str:
        out_str = 'MapUnit{\'' + self.label + '\'}'
        return out_str

    def __repr__(self) -> str:
        repr_str = 'MapUnit{'
        repr_str += f'type : \'{self.type}\', '
        repr_str += f'label : \'{self.label}\', '
        repr_str += f'abbreviation : \'{self.abbreviation}\', '
        repr_str += f'description : \'{self.description}\', '
        repr_str += f'color : \'{self.color}\', '
        repr_str += f'pattern : \'{self.pattern}\', '
        #repr_str += f'mask : {self.mask.shape}, ' if self.mask is not None else f'mask : {self.mask}, ',
        repr_str += f'mask_confidence : {self.mask_confidence}'
        return repr_str

class Legend():
    def __init__(self, features=None, provenance=None):
        self.features = features if features is not None else []
        self.provenance = provenance

    def to_dict(self):
        feature_dict = {}
        for map_unit in self.features:
            feature_dict[map_unit.label] = map_unit.to_dict()
        return {
            'features' : feature_dict,
            'provenance' : self.provenance
        }
    
    def map_unit_distr(self):
        dist = {}
        for feature in self.features:
            if feature.type in dist:
                dist[feature.type].append(feature.label)
            else:
                dist[feature.type] = [feature.label]
        return dist

    def __len__(self):
        return len(self.features)
    
    def __eq__(self, __value: object) -> bool:
        if self is None or __value is None:
            if self is None and __value is None:
                return True
            else:
                return False
        if self.provenance != __value.provenance:
            if DEBUG_MODE:
                print(f'Provenance Mismatch: {self.provenance} != {__value.provenance}')
            return False
        for u1 in self.features:
            matched = False
            for u2 in __value.features:
                if u1 == u2:
                    if DEBUG_MODE:
                        print(f'Feature match: {u1} == {u2}')
                    matched = True
                    break
            if not matched:
                if DEBUG_MODE:
                    print(f'Feature Mismatch: {u1} != {u2}')
                return False
        return True

    def __str__(self) -> str:
        out_str = 'Legend{Provenance : ' + f'{self.provenance}, {len(self.features)} Features : {self.features}' + '}'
        return out_str
    
    def __repr__(self) -> str:
        repr_str = 'Legend{Provenance : ' + f'{self.provenance}, {len(self.features)} Features : {self.features}' + '}'
        return repr_str

class Layout():
    def __init__(self, map=None, legend=None, correlation_diagram=None, cross_section=None, point_legend=None, line_legend=None, polygon_legend=None, provenance=None):
        self.provenance = provenance
        self.map = map
        self.point_legend = point_legend
        self.line_legend = line_legend
        self.polygon_legend = polygon_legend
        self.correlation_diagram = correlation_diagram
        self.cross_section = cross_section

    def __str__(self) -> str:
        out_str = 'Layout{'
        out_str += f'map : {self.map}, '
        out_str += f'correlation_diagram : {self.correlation_diagram}, '
        out_str += f'cross_section : {self.cross_section}, '
        out_str += f'point_legend : {self.point_legend}, '
        out_str += f'line_legend : {self.line_legend}, '
        out_str += f'polygon_legend : {self.polygon_legend}, '
        out_str += f'provenance : {self.provenance}'
        out_str += '}'
        return out_str
    
    def __repr__(self) -> str:
        out_str = 'Layout{'
        out_str += f'map : {self.map}, '
        out_str += f'correlation_diagram : {self.correlation_diagram}, '
        out_str += f'cross_section : {self.cross_section}, '
        out_str += f'point_legend : {self.point_legend}, '
        out_str += f'line_legend : {self.line_legend}, '
        out_str += f'polygon_legend : {self.polygon_legend}, '
        out_str += f'provenance : {self.provenance}'
        out_str += '}'
        return out_str

    def __eq__(self, __value: object) -> bool:
        if self is None or __value is None:
            if self is None and __value is None:
                return True
            else:
                return False
        if isinstance(self.map, (np.ndarray, np.generic)) and isinstance(__value.map, (np.ndarray, np.generic)):
            if (self.map != __value.map).any():
                if DEBUG_MODE:
                    print(f'Map Mismatch: {self.map} != {__value.map}')
                return False
        else:
            if self.map != __value.map:
                if DEBUG_MODE:
                    print(f'Map Mismatch: {self.map} != {__value.map}')
                return False
        if isinstance(self.correlation_diagram, (np.ndarray, np.generic)) or isinstance(__value.correlation_diagram, (np.ndarray, np.generic)):
            if (self.correlation_diagram != __value.correlation_diagram).any():
                if DEBUG_MODE:
                    print(f'Correlation Diagram Mismatch: {self.correlation_diagram} != {__value.correlation_diagram}')
                return False
        else:
            if self.correlation_diagram != __value.correlation_diagram:
                if DEBUG_MODE:
                    print(f'Correlation Diagram Mismatch: {self.correlation_diagram} != {__value.correlation_diagram}')
                return False
        if isinstance(self.cross_section, (np.ndarray, np.generic)) or isinstance(__value.cross_section, (np.ndarray, np.generic)):
            if (self.cross_section != __value.cross_section).any():
                if DEBUG_MODE:
                    print(f'Cross Section Mismatch: {self.cross_section} != {__value.cross_section}')
                return False
        else:
            if self.cross_section != __value.cross_section:
                if DEBUG_MODE:
                    print(f'Cross Section Mismatch: {self.cross_section} != {__value.cross_section}')
                return False
        if isinstance(self.point_legend, (np.ndarray, np.generic)) or isinstance(__value.point_legend, (np.ndarray, np.generic)):
            if (self.point_legend != __value.point_legend).any():
                if DEBUG_MODE:
                    print(f'Point Legend Mismatch: {self.point_legend} != {__value.point_legend}')
                return False
        else:
            if self.point_legend != __value.point_legend:
                if DEBUG_MODE:
                    print(f'Point Legend Mismatch: {self.point_legend} != {__value.point_legend}')
                return False
        if isinstance(self.line_legend, (np.ndarray, np.generic)) or isinstance(__value.line_legend, (np.ndarray, np.generic)):
            if (self.line_legend != __value.line_legend).any():
                if DEBUG_MODE:
                    print(f'Line Legend Mismatch: {self.line_legend} != {__value.line_legend}')
                return False
        else:
            if self.line_legend != __value.line_legend:
                if DEBUG_MODE:
                    print(f'Line Legend Mismatch: {self.line_legend} != {__value.line_legend}')
                return False
        if isinstance(self.polygon_legend, (np.ndarray, np.generic)) or isinstance(__value.polygon_legend, (np.ndarray, np.generic)):
            if (self.polygon_legend != __value.polygon_legend).any():
                if DEBUG_MODE:
                    print(f'Polygon Legend Mismatch: {self.polygon_legend} != {__value.polygon_legend}')
                return False
        else:
            if self.polygon_legend != __value.polygon_legend:
                if DEBUG_MODE:
                    print(f'Polygon Legend Mismatch: {self.polygon_legend} != {__value.polygon_legend}')
                return False
        if self.provenance != __value.provenance:
            if DEBUG_MODE:
                print(f'Provenance Mismatch: {self.provenance} != {__value.provenance}')
            return False
        return True
    
    def to_dict(self):
        return {
            'provenance' : self.provenance,
            'map' : self.map,
            'point_legend' : self.point_legend,
            'line_legend' : self.line_legend,
            'polygon_legend' : self.polygon_legend,
            'correlation_diagram' : self.correlation_diagram,
            'cross_section' : self.cross_section,
        }

class GeoReference():
    def __init__(self, crs:CRS=None, transform:rasterio.transform.Affine=None, gcps:List[GroundControlPoint]=None, confidence:float=None, provenance=None):
        self.crs = crs
        self.transform = transform
        self.gcps = gcps
        self.confidence = confidence
        self.provenance = provenance

    def __eq__(self, __value: object) -> bool:
        if self is None or __value is None:
            if self is None and __value is None:
                return True
            else:
                return False
            
        # Mark an object as equal if its crs and transform are equal
        if self.crs is not None and self.transform is not None and __value.crs is not None and __value.transform is not None:
            if self.crs != __value.crs:
                if DEBUG_MODE:
                    print(f'CRS Mismatch: {self.crs} != {__value.crs}')
                return False
            if self.transform != __value.transform:
                if DEBUG_MODE:
                    print(f'Transform Mismatch: {self.transform} != {__value.transform}')
                return False
        # Otherwise test on the gcps and provenance
        else:
            if self.gcps != __value.gcps:
                if DEBUG_MODE:
                    print(f'GCP Mismatch: {self.gcps} != {__value.gcps}')
                return False
            if self.provenance != __value.provenance:
                if DEBUG_MODE:
                    print(f'Provenance Mismatch: {self.provenance} != {__value.provenance}')
                return False
        return True

class TextUnit():
    def __init__(self, label:str, geometry, confidence:float):
        self.label = label
        self.geometry = geometry
        self.confidence = confidence

class OCRText():
    def __init__(self, provenance:str, features:List[TextUnit]=[]):
        self.provenance = provenance
        self.features = features
        
# class MetadataExtraction(BaseModel):
#     model_config = ConfigDict(coerce_numbers_to_str=True)

#     map_id: str
#     title: str
#     authors: List[str]
#     year: str  # should be an int, but there's a chance somethign else is (incorrectly) extracted
#     scale: str  # of the format 1:24000
#     quadrangles: List[str]
#     datum: str
#     vertical_datum: str
#     projection: str
#     coordinate_systems: List[str]
#     base_map: str
#     counties: List[str]
#     population_centres: List[TextExtraction]  # a list of cities, towns, and villages
#     states: List[str]
#     country: str
#     places: List[
#         TextExtraction
#     ]  # a list of places, each place having a name and coordinates

class CMAAS_MapMetadata():
    def __init__(self, provenance:str, title:str=None, authors:List[str]=[], publisher:str=None, url:str=None, source_url:str=None, year:int=None, organization:str=None, map_color:str=None, physiographic_region:str=None, scale:str=None, map_shape:str=None):
        self.provenance = provenance
        self.title = title
        self.authors = authors
        self.publisher = publisher
        self.url = url # What is the diff between url and source url.
        self.source_url = source_url
        
        # Gold standard Validation criteria
        self.year = year
        self.organization = organization # Is this signifgantly difference then publisher?
        self.scale = scale # E.g. 1:24,000 
        self.map_color = map_color # Full Color, Monocolor, Greyscale
        self.map_shape = map_shape # Rectangle, Non-Rectange
        self.physiographic_region = physiographic_region # Would be helpful if theres a link to a resource that can display the possible values for this

    def to_dict(self):
        return {
            'provenance' : self.provenance,
            'title' : self.title,
            'authors' : self.authors,
            'year' : self.year,
            'publisher' : self.publisher,
            'organization' : self.organization,
            'source_url' : self.source_url,
            'url' : self.url,
            'scale' : self.scale,
            'map_shape' : self.map_shape,
            'map_color' : self.map_color,
            'physiographic_region' : self.physiographic_region,
            
        }
    
    def __str__(self) -> str:
        out_str = 'CMASS_MapMetadata{'
        out_str += f'provenance : \'{self.provenance}\', '
        out_str += f'title : \'{self.title}\', '
        out_str += f'authors : {self.authors}, '
        out_str += f'publisher : \'{self.publisher}\', '
        out_str += f'url : \'{self.url}\', '
        out_str += f'source_url : \'{self.source_url}\', '
        out_str += f'year : {self.year}, '
        out_str += f'organization : \'{self.organization}\', '
        out_str += f'scale : \'{self.scale}\', '
        out_str += f'map_color : \'{self.map_color}\', '
        out_str += f'map_shape : \'{self.map_shape}\', '
        out_str += f'physiographic_region : \'{self.physiographic_region}\''
        out_str += '}'
        return out_str
    
    def __repr__(self) -> str:
        repr_str = 'CMASS_MapMetadata{'
        repr_str += f'provenance : \'{self.provenance}\', '
        repr_str += f'title : \'{self.title}\', '
        repr_str += f'authors : {self.authors}, '
        repr_str += f'publisher : \'{self.publisher}\', '
        repr_str += f'url : \'{self.url}\', '
        repr_str += f'source_url : \'{self.source_url}\', '
        repr_str += f'year : {self.year}, '
        repr_str += f'organization : \'{self.organization}\', '
        repr_str += f'scale : \'{self.scale}\', '
        repr_str += f'map_color : \'{self.map_color}\', '
        repr_str += f'map_shape : \'{self.map_shape}\', '
        repr_str += f'physiographic_region : \'{self.physiographic_region}\''
        repr_str += '}'
        return repr_str

    def __eq__(self, __value: object) -> bool:
        if self is None or __value is None:
            if self is None and __value is None:
                return True
            else:
                return False
            
        if self.provenance != __value.provenance:
            if DEBUG_MODE:
                print(f'Provenance Mismatch: {self.provenance} != {__value.provenance}')
            return False
        if self.title != __value.title:
            if DEBUG_MODE:
                print(f'Title Mismatch: {self.title} != {__value.title}')
            return False
        if self.authors != __value.authors:
            if DEBUG_MODE:
                print(f'Authors Mismatch: {self.authors} != {__value.authors}')
            return False
        if self.publisher != __value.publisher:
            if DEBUG_MODE:
                print(f'Publisher Mismatch: {self.publisher} != {__value.publisher}')
            return False
        if self.url != __value.url:
            if DEBUG_MODE:
                print(f'URL Mismatch: {self.url} != {__value.url}')
            return False
        if self.source_url != __value.source_url:
            if DEBUG_MODE:
                print(f'Source URL Mismatch: {self.source_url} != {__value.source_url}')
            return False
        if self.year != __value.year:
            if DEBUG_MODE:
                print(f'Year Mismatch: {self.year} != {__value.year}')
            return False
        if self.organization != __value.organization:
            if DEBUG_MODE:
                print(f'Organization Mismatch: {self.organization} != {__value.organization}')
            return False
        if self.scale != __value.scale:
            if DEBUG_MODE:
                print(f'Scale Mismatch: {self.scale} != {__value.scale}')
            return False
        if self.map_color != __value.map_color:
            if DEBUG_MODE:
                print(f'Map Color Mismatch: {self.map_color} != {__value.map_color}')
            return False
        if self.map_shape != __value.map_shape:
            if DEBUG_MODE:
                print(f'Map Shape Mismatch: {self.map_shape} != {__value.map_shape}')
            return False
        if self.physiographic_region != __value.physiographic_region:
            if DEBUG_MODE:
                print(f'Physiographic Region Mismatch: {self.physiographic_region} != {__value.physiographic_region}')
            return False
        return True

class CMAAS_Map():
    def __init__(self, name:str, image:np.ndarray=None, metadata:CMAAS_MapMetadata=None, layout:Layout=None, legend:Legend=None, georef:GeoReference=None, ocrtext:OCRText=None):
        self.name = name
        self.image = image
        self.metadata = metadata
        self.layout = layout
        self.legend = legend
        self.georef = georef
        self.ocrtext = ocrtext

        # Segmentation mask
        self.mask = None
    
    def __eq__(self, __value: object) -> bool:
        if self is None or __value is None:
            if self is None and __value is None:
                return True
            else:
                return False

        if self.name != __value.name:
            if DEBUG_MODE:
                print(f'Name Mismatch: {self.name} != {__value.name}')
            return False
        if self.image is None or __value.image is None:
            if not (self.image is None and __value.image is None):
                return False
        if self.image.shape != __value.image.shape:
            if DEBUG_MODE:
                print(f'Shape Mismatch: {self.image.shape} != {__value.image.shape}')
            return False
        # if self.georef != __value.georef:
        #     if DEBUG_MODE:
        #         print(f'GeoReference Mismatch: {self.georef} != {__value.georef}')
        #     return False
        if self.legend != __value.legend:
            if DEBUG_MODE:
                print(f'Legend Mismatch: {self.legend} != {__value.legend}')
            return False
        if self.layout != __value.layout:
            if DEBUG_MODE:
                print(f'Layout Mismatch: {self.layout} != {__value.layout}')
            return False
        if self.metadata != __value.metadata:
            if DEBUG_MODE:
                print(f'Metadata Mismatch: {self.metadata} != {__value.metadata}')
            return False
        return True

    def __str__(self) -> str:
        out_str = 'CMASS_Map{'
        out_str += f'name : \'{self.name}\', '
        if self.image is not None:
            out_str += f'image : {self.image.shape}, '
        else:
            out_str += f'image : {self.image}, '
        out_str += f'georef : {self.georef}, '
        out_str += f'legend : {self.legend}, '
        out_str += f'layout : {self.layout}, '
        out_str += f'metadata : {self.metadata}'
        out_str += '}'
        return out_str

    def __repr__(self) -> str:
        repr_str = 'CMASS_Map{'
        repr_str += f'name : \'{self.name}\', '
        if self.image is not None:
            repr_str += f'image : {self.image.shape}, '
        else:
            repr_str += f'image : {self.image}, '
        repr_str += f'georef : {self.georef}, '
        repr_str += f'legend : {self.legend}, '
        repr_str += f'layout : {self.layout}, '
        repr_str += f'metadata : {self.metadata}'
        repr_str += '}'
        return repr_str
    
    def to_dict(self):
        return {
            'version' : '0.1',
            'map_name' : self.name,
            'metadata' : self.metadata.to_dict() if self.metadata is not None else None,
            'layout' : self.layout.to_dict() if self.layout is not None else None,
            'legend' : self.legend.to_dict() if self.legend is not None else None,
            'georef' : [g.to_dict() for g in self.georef] if self.georef is not None else None,
            'ocrtext' : self.ocrtext.to_dict() if self.ocrtext is not None else None
        }

