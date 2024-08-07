import cv2
import logging

def viz_map_unit(image, map_unit, color=(0,255,0), thickness=5):
    for point in map_unit.bounding_box:
        for i,x in enumerate(point):
            point[i] = int(x)
    center = [int(map_unit.bounding_box[0][0] + (map_unit.bounding_box[1][0]-map_unit.bounding_box[0][0])/2), int(map_unit.bounding_box[0][1] +(map_unit.bounding_box[1][1]-map_unit.bounding_box[0][1])/2)]
    # logging.debug(f'Viz yolo annotation min_xy: {map_unit.bounding_box[0]} max_xy: {map_unit.bounding_box[1]}')
    cv2.rectangle(image, map_unit.bounding_box[0], map_unit.bounding_box[1], color, thickness)
    cv2.putText(image, map_unit.label, center, cv2.FONT_HERSHEY_SIMPLEX, 1, color=color, thickness=2)
    return image

def viz_yolo_annotation(image, annotation, color=(255,0,255), thickness=5):
    h,w,c = image.shape
    # logging.debug(f'Viz yolo annotation c: {c} w: {w} h: {h}')
    center = [int(annotation[1]*w), int(annotation[2]*h)]
    min_xy = [int((annotation[1]-(annotation[3]/2))*w), int((annotation[2]-(annotation[4]/2))*h)]
    max_xy = [int((annotation[1]+(annotation[3]/2))*w), int((annotation[2]+(annotation[4]/2))*h)]
    logging.debug(f'Viz yolo annotation min_xy: {min_xy} max_xy: {max_xy}')
    cv2.rectangle(image, min_xy, max_xy, color, thickness)
    cv2.putText(image, str(annotation[0]), center, cv2.FONT_HERSHEY_SIMPLEX, 1, color=color, thickness=2)
    return image