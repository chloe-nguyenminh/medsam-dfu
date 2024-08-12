#%% import packages
import numpy as np
import os
join = os.path.join 
from skimage import io, transform

from tqdm import tqdm
import pandas as pd
import argparse
import random
from PIL import Image
from torchvision.transforms.functional import hflip, vflip
import albumentations as A
import cv2
import matplotlib.pyplot as plt

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([251 / 255, 252 / 255, 30 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def get_bbox(gt2D, bbox_shift):
    y_indices, x_indices = np.where(gt2D > 0)
    x_min, x_max = np.min(x_indices), np.max(x_indices)
    y_min, y_max = np.min(y_indices), np.max(y_indices)
    H, W = gt2D.shape
    x_min = max(0, x_min - random.randint(0, bbox_shift))
    x_max = min(W, x_max + random.randint(0, bbox_shift))
    y_min = max(0, y_min - random.randint(0, bbox_shift))
    y_max = min(H, y_max + random.randint(0, bbox_shift))
    bboxes = np.array([x_min, y_min, x_max, y_max])
    return bboxes

def random_crop(img, mask, width, height, bbox_shift):
    assert img.shape[0] >= height
    assert img.shape[1] >= width
    assert img.shape[0] == mask.shape[0]
    assert img.shape[1] == mask.shape[1]
    label_ids = np.unique(mask)[1:]
    mask = np.uint8(mask == 255)
    cropped_mask = np.zeros((width, height))
    while np.max(cropped_mask)!= 1:
        x = random.randint(0, img.shape[1] - width)
        y = random.randint(0, img.shape[0] - height)
        cropped_img = img[y:y+height, x:x+width]
        cropped_mask = mask[y:y+height, x:x+width] 
    return cropped_img, cropped_mask

def preprocess(img_path: str, 
                gt_path: str,
                bbox_shift: int,
                size: int):
    gt_data_ori = np.uint8(io.imread(gt_path))
    # crop the ground truth with non-zero slices
    img = io.imread(img_path)
    if np.max(img) > 255.0:
        img = np.uint8((img-img.min()) / (np.max(img)-np.min(img))*255.0)
    if len(img.shape) == 2:
        img = np.repeat(np.expand_dims(img, -1), 3, -1)
    assert len(img.shape) == 3, 'image data is not three channels: img shape:' + str(img.shape) + image_name
    # convert three channel to one channel
    if img.shape[-1] > 3:
        img = img[:,:,:3]
    label_ids = np.unique(gt_data_ori)[1:]
    mask = np.uint8(gt_data_ori == 255)
    assert img.shape[0] == mask.shape[0]
    assert img.shape[1] == mask.shape[1]

    resized_img, resized_mask = resize(img, mask, width=mask.shape[0], height=mask.shape[1], size=size)
    transformed_img, transformed_mask = transform(resized_img, resized_mask)
    #Normalize images to ImageNet mean and std
    normalization = A.Compose(
        [A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    normalized = normalization(image=transformed_img)
    normalized_img = normalized['image']
         
    #SANITY CHECK
    # _, axs = plt.subplots(1, 2, figsize=(10, 10))
    # axs[0].imshow(transformed_img01)
    # show_mask(transformed_mask, axs[0])
    # axs[0].axis("off")

    # axs[1].imshow(transformed_mask)
    # axs[1].axis("off")
    # plt.subplots_adjust(wspace=0.01, hspace=0)
    # plt.savefig("test_augmentation.png", bbox_inches="tight", dpi=300)
    # plt.close()
    # print("done sanity check ") 

    return normalized_img, transformed_mask

def resize(image, mask, width, height, size):
    resize = A.Compose([
        A.OneOf([
            A.LongestMaxSize(max_size=width, interpolation=1),
            A.LongestMaxSize(max_size=height, interpolation=1),
        ], p=0.5),
        A.OneOf([
            A.PadIfNeeded(min_height=min(width, height), min_width=min(width, height), border_mode=0, value=(0,0,0)),
            A.PadIfNeeded(min_height=min(width, height), min_width=min(width, height), border_mode=cv2.BORDER_REFLECT_101, value=(0,0,0)),
        ]),
        A.OneOf([
            A.RandomCrop(height=size, width=size),
            A.CenterCrop(height=size, width=size),
        ], p=1),
    ])
    resized = resize(image=image, mask=mask)
    resized_image = resized['image']
    resized_mask = resized['mask']
    return resized_image, resized_mask

def transform(image, mask):
    transform = A.Compose([
        # Affine transforms
        A.OneOf([
            A.RandomRotate90(),
            A.HorizontalFlip(),
            A.VerticalFlip(),
            A.Transpose(),
        ], p=0.5),
        
        # Perspective transforms
        A.OneOf([A.Perspective(scale=(0.05, 0.1), p=0.5),
        ], p=0.5),
        
        # Brightness/contrast/colors manipulations
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5)
        ], p=0.5),
        
        # Gaussian noise
        A.OneOf([
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
        ]),
    ])
    
    # Image blurring and sharpening
    blurr = A.OneOf([
            A.Blur(blur_limit=3),
            A.GaussianBlur(blur_limit=3),
            A.MotionBlur(blur_limit=3),
            A.Sharpen(alpha=0.5, lightness=0.5),
        ], p=1)

    transformed = transform(image=image, mask=mask)
    transformed_image = transformed['image']
    transformed_mask = transformed['mask']

    blurred = blurr(image=transformed_image, mask=transformed_mask)
    blurred_image = blurred['image']

    return blurred_image, transformed_mask

# preprocess(img_path="/lustre06/project/6086937/chaunm/MedSAM/data/ManchesterDFU/validation/101027.jpg", 
#             gt_path="/lustre06/project/6086937/chaunm/MedSAM/data/ManchesterDFU_pixelmasks/101027.png", 
#             bbox_shift=10)