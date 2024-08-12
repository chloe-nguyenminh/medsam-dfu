import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import io
import cv2
join = os.path.join
from math import prod
from PIL import Image
import torch
import torchvision
from segment_anything import SamAutomaticMaskGenerator
from segment_anything.build_sam_lora import sam_lora_model_registry
from sam_lora_image_encoder import LoRA_Sam
from skimage import io, transform
import monai
import torch.nn as nn
import torch.nn.functional as F
import argparse
from math import ceil, floor
import random
from tqdm import tqdm
from matplotlib.image import imsave as imsave
from sklearn import metrics
from sklearn.metrics import confusion_matrix, f1_score, jaccard_score
from importlib import import_module
from sam_lora_utils import test_single_volume

torch.cuda.empty_cache()

def show_mask(mask, random_color=False):
    ax = plt.gca()
    ax.set_autoscale_on(False)
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([251 / 255, 252 / 255, 30 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['predicted_iou']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for i in range(len(sorted_anns)):
        m = sorted_anns[i]['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
        # if i > 50:
        #   break
    ax.imshow(img)

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(
        plt.Rectangle((x0, y0), w, h, edgecolor="blue", facecolor=(0, 0, 0, 0), lw=2)
    )

class_to_name = {1: "dfu"}

@torch.no_grad()


def sam_lora_eval(gt_path: str,
                 img_path: str,
                 single_outpath: str,
                 multi_outpath: str,
                 checkpoint: str,
                 df_filename: str,
                 model_type: str,
                 ratio: float,
                 lora_ckpt: str):
    count = 0
    img_suffix = '.jpg' 
    gt_suffix = '.png' 
    eval_dict = {"Name": [],
                 "Jaccard coefficient or IoU": [],
                "F1 score or DSC": []} 
    if single_outpath:
        os.makedirs(single_outpath, exist_ok=True)
    if multi_outpath:
        os.makedirs(multi_outpath, exist_ok=True)
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    sam, _ = sam_lora_model_registry[model_type](checkpoint=checkpoint, 
                                                image_size=256,
                                                num_classes=8,
                                                pixel_mean=[0.485, 0.456, 0.406],
                                                pixel_std=[0.229, 0.224, 0.225])
    
    pkg = import_module('sam_lora_image_encoder')
    sam_model = pkg.LoRA_Sam(sam, 4).cuda()
    sam_model.load_lora_parameters(lora_ckpt)
    sam_model_posttrain = sam_model
    sam_model_posttrain.to(device=DEVICE)
    sam_model_posttrain.eval()
    posttrain_predictor = SamAutomaticMaskGenerator(model=sam_model_posttrain,
                                                    points_per_side=32,
                                                    pred_iou_thresh=0,
                                                    stability_score_thresh=0)

    img_lst = os.listdir(img_path)
    img_lst = random.sample(img_lst, ceil(len(img_lst)*ratio))
    # print("Number of testing samples: ", len_lst)
    count = 0
    for img_dir in tqdm(img_lst):
        str_name = img_dir.split("/")[-1].split(img_suffix)[0]
        gt_ori = np.uint8(io.imread(join(gt_path, str_name + gt_suffix)))
        label_ids = np.unique(gt_ori)[1:]
        groundtruth_mask = np.uint8(gt_ori == 255)
        reshaped_gt_mask = groundtruth_mask.flatten()
        reshaped_gt_mask = reshaped_gt_mask.astype(bool)

        image = cv2.imread(img_path+str_name+img_suffix)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[0], image.shape[1]
        if np.max(image) > 255.0:
            image = np.uint8((image-image.min()) / (np.max(image)-np.min(image))*255.0)
        if len(image.shape) == 2:
            image = np.repeat(np.expand_dims(image, -1), 3, -1)
        assert len(image.shape) == 3, 'image data is not three channels: img shape:' + str(img.shape) + image_name

        # metric, pred_mask = test_single_volume(image, gt_ori, sam_model, classes=1, multimask_output=False,
        #                                        patch_size=[256, 256], input_size=[h, w],
        #                                        test_save_path=single_outpath, case=str_name, z_spacing=1)
        
        
        posttrain_masks = posttrain_predictor.generate(image)
        if len(posttrain_masks) > 0:
            final_masks_lst = [posttrain_masks[i]['segmentation'].astype(int) for i in range(len(posttrain_masks))]
            final_mask = np.zeros((h, w), dtype=int)
            for i in range(len(final_masks_lst)):
                if final_masks_lst[i].shape == (h, w):
                    final_mask += final_masks_lst[i]
            final_mask_bin = np.where(final_mask>0, 1, 0)
        else:
            final_mask_bin = np.zeros((h,w))
        reshaped_mask = final_mask_bin.flatten()
        reshaped_mask = reshaped_mask.astype(bool)

        final_mask_bin = final_mask_bin.astype(bool)
        final_mask_img = Image.fromarray(np.uint8(final_mask_bin))
        
        if single_outpath and multi_outpath:
            plt.figure(figsize=(5,5))
            plt.imshow(image)
            show_anns(posttrain_masks)
            plt.axis("off")
            plt.savefig(multi_outpath+str_name+".png", bbox_inches="tight", dpi=300)
            plt.close()

            _, axs = plt.subplots(1, 2, figsize=(10, 10))
            axs[0].imshow(groundtruth_mask)
            axs[0].axis("off")
            axs[0].set_title("groundtruth")

            axs[1].imshow(final_mask_img)
            axs[1].axis("off")
            axs[1].set_title("predicted")
            plt.subplots_adjust(wspace=0.01, hspace=0)
            plt.savefig(single_outpath+str_name+".png", bbox_inches="tight", dpi=300)
            plt.close()
    
        jaccard_score = metrics.jaccard_score(y_true=reshaped_gt_mask, y_pred=reshaped_mask, average="macro")
        unweighted_f1score = f1_score(y_true=reshaped_gt_mask, y_pred=reshaped_mask, average="macro")
        
        final_mask_bin = final_mask_bin.astype(int)
        groundtruth_mask = groundtruth_mask.astype(int)
        # pred = torch.tensor(final_mask_bin).float()
        # gt_mask = torch.tensor(groundtruth_mask).float()

        eval_dict["Name"].append(str_name)
        eval_dict["Jaccard coefficient or IoU"].append(jaccard_score) 
        eval_dict["F1 score or DSC"].append(unweighted_f1score) 

        count += 1
        if count > 10:
            break 
    eval_df = pd.DataFrame.from_dict(eval_dict)
    eval_df = eval_df.fillna(0)
    if df_filename:
        eval_df.to_csv(df_filename)
    
    mean_dsc = sum(eval_dict["F1 score or DSC"])/len(eval_dict["F1 score or DSC"])
    mean_actual_iou = sum(eval_dict["Jaccard coefficient or IoU"])/len(eval_dict["Jaccard coefficient or IoU"])

    print("mean DSC: ", mean_dsc)
    print("mean Jaccard coefficient or IoU: ", mean_actual_iou)

    return mean_actual_iou, mean_dsc

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="run sam inference on validation set"
    )
    parser.add_argument(
        "--gt_path",
        type=str,
        default="data/DFUC2022_test_split/masks/",
        help="path to the groundtruth pixel mask folder",
    )
    parser.add_argument(
        "--img_path",
        type=str,
        default="data/DFUC2022_test_split/images/",
        help="path to the images for validation folder",
    )
    parser.add_argument(
        "--single_outpath",
        type=str,
        default="output/sam_lora_pretrain_single_mask/",
        help="path to output folder for single mask per image",
    )
    parser.add_argument(
        "--multi_outpath",
        type=str,
        default="output/sam_lora_pretrain_multimask/",
        help="path to output folder to multiple masks per image",
    )
    parser.add_argument(
        "--checkpoint",
        default="work_dir/SAM-ViT-H-20240723-1333/sam_model_best_val_accuracy.pth",
        type=str,
        help="path to the trained model",
    )
    parser.add_argument(
        "--lora_ckpt",
        default="/lustre06/project/6086937/chaunm/MedSAM/work_dir/SAM-ViT-H-LORA-20240806-2324/sam_lora_ckpt_latest.pth",
        type=str,
        help="path to the trained model",
    )
    parser.add_argument(
        "--df_filename",
        type=str,
        default="sam_lora_pretrain_scores.csv",
        help="preferred name of validation score .csv file"
        )
    parser.add_argument(
        "--model_type",
        type=str,
        default="vit_h"
    )
    parser.add_argument(
        "--ratio",
        type=float,
        default=1.0,
        help="proportion of testing images preferred"
    )
    args = parser.parse_args()
    sam_lora_eval(**vars(args))
    

