import numpy as np
from imantics import Polygons, Mask, BBox
from shapely.geometry import Polygon
import skimage.io as io
import os
join = os.path.join
import argparse
import pandas as pd
import json
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
import cv2
import icecream as ic

def get_anns_per_image(image_id: int, gt_path):
    image = Image.open(gt_path)
    output = []
    gt_data_ori = np.uint8(io.imread(gt_path))
    label_ids = np.unique(gt_data_ori)[1:]
    mask = np.uint8(gt_data_ori == 255)
    
    #get annotation per segmentation in binary mask
    try: 
        output = []
        anns = Mask(image)
        polygons = anns.polygons()
        segmentations, points = polygons.segmentation, polygons.points
        
        for i in range(len(segmentations)):
            s = segmentations[i]
            #find bbox 
            x_coords = [s[i] for i in range(len(s)) if i%2==0]
            y_coords = [s[i] for i in range(len(s)) if i%2==1]
            xmin, ymin, xmax, ymax = min(x_coords), min(y_coords), max(x_coords), max(y_coords)
            coco_bbox = [xmin, ymin, xmax-xmin, ymax-ymin]
            #find area
            area = Polygon(points[i]).area
            segmentation_dict = {"segmentation": [s],
                                "bbox": coco_bbox,
                                "area": area}
            output.append(segmentation_dict)
            
    except SystemError:
        output = [{"segmentation": [list(np.zeros(8))],
                    "bbox": [0, 0, 0, 0],
                    "area": 0}]
    
    # anns_lst = get_segmentation(image)
    return output

def get_data(img_path, gt_path, bbox_path, json_filename):
    img_name_suffix = ".jpg"
    gt_name_suffix = ".png"
    
    info = {'description': 'DFU2022 Dataset',
            'url': '',
            'version': '0.1.0',
            'year': 2022,
            'date_created': ""}
    
    licenses = [{'id': 1,
  'name': 'Attribution-NonCommercial-ShareAlike License',
  'url': 'http://creativecommons.org/licenses/by-nc-sa/2.0/'}]
    
    categories = [{'id': 0, 
                   'name': 'diabetic foot ulcer wound', 
                   'supercategory': 'N/A'}]
    images = []
    annotations = []
    gt_files = sorted(os.listdir(gt_path))
    #id, (w, h)
    annotation_count = 0
    for gt_file in tqdm(gt_files):
        # gt_file = gt_files[i]
        filename = os.path.basename(gt_file).split(".")[0]
        id = int(filename)
        gt_filename = filename+gt_name_suffix
        gt_dir = join(gt_path, str(filename)+gt_name_suffix)
        image_dir = join(img_path, str(filename)+img_name_suffix) 
        
        #get annotations data
        if os.path.isfile(image_dir) and os.path.isfile(gt_dir):
            #get image data
            img = Image.open(image_dir)
            images_dict = {"id": id,
                        "file_name": filename+img_name_suffix,
                        "width": img.size[0],
                        "height": img.size[1]}
            images.append(images_dict)
            
            image_name = filename + gt_name_suffix
            anns_lst_per_image = get_anns_per_image(id, gt_dir)
            i = 0
            for anns_dict in anns_lst_per_image:
                anns_dict["id"] = int(str(id)+str(i))
                anns_dict["image_id"] = id
                anns_dict["category_id"] = 0
                anns_dict["iscrowd"] = 0
                i+= 1
                annotation_count += 1
            annotations.extend(anns_lst_per_image)
            
        if annotation_count > 50:
            break
        
    output = {"info": info,
              "licenses": licenses,
              "categories": categories,
              "images": images,
              "annotations": annotations}
    with open(json_filename, "w") as fp:
        json.dump(output, fp) 
                
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--img_path",
        type=str,
        default='data/DFUC2022_train_val/validation/images/',
        help="path to training image files",
    )
    parser.add_argument(
        "--gt_path",
        type=str,
        default="data/DFUC2022_train_val/validation/labels/",
        help="path to training groundtruth masks",
    )
    parser.add_argument(
        "--json_filename",
        type=str,
        default="test_convert.json",
        help="preferred .json output filename",
    )
    args = parser.parse_args()
    get_data(**vars(args))

