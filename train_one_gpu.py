# -*- coding: utf-8 -*-
"""
train the image encoder and mask decoder
freeze prompt image encoder
"""

# %% setup environment
import numpy as np
import os
join = os.path.join
from tqdm import tqdm
from skimage import transform
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import monai
from segment_anything_new.build_sam import sam_model_registry as _sam_model_registry
# from sam_LoRa import LoRA_Sam
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
import argparse
import random
from datetime import datetime
import shutil
import glob
import numpy as np
import matplotlib.pyplot as pltS
from pre_grey_rgb import *
from sam_inference import *
from ckpt_convert import convert
from random import sample

# set seeds
torch.manual_seed(2023)
torch.cuda.empty_cache()
torch.autograd.set_detect_anomaly(True)

# torch.distributed.init_process_group(backend="gloo")

os.environ["OMP_NUM_THREADS"] = "4"  # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "4"  # export OPENBLAS_NUM_THREADS=4
os.environ["MKL_NUM_THREADS"] = "6"  # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"  # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "6"  # export NUMEXPR_NUM_THREADS=6

img_name_suffix = '.jpg' 
gt_name_suffix = '.png'

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([251 / 255, 252 / 255, 30 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_anns(anns, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([255 / 255, 255 / 255, 30 / 255, 0.6])
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(
        plt.Rectangle((x0, y0), w, h, edgecolor="blue", facecolor=(0, 0, 0, 0), lw=2)
    )

class NpyDataset(Dataset):
    def __init__(self, img_path, gt_path, bbox_shift, img_size):
        self.gt_path = gt_path
        self.img_path = img_path
        self.gt_files = sorted(os.listdir(gt_path))
        self.img_size = img_size
        # for small scale training
        self.gt_files = sample(self.gt_files, len(self.gt_files)//1)
        self.bbox_shift = bbox_shift
        self.gt_path_files = []
        img_name_suffix = ".jpg"
        gt_name_suffix = ".png"
        for gt_file in self.gt_files:
            filename = os.path.basename(gt_file).split(".")[0]
            gt_dir = join(self.gt_path, str(filename)+gt_name_suffix)
            image_dir = join(self.img_path, str(filename)+img_name_suffix) 
            if os.path.isfile(image_dir) and os.path.isfile(gt_dir):
                self.gt_path_files.append(gt_dir)
        print(f"number of images: {len(self.gt_path_files)}")

    def __len__(self):
        return len(self.gt_path_files)

    def __getitem__(self, index):
        # load npy image (256, 256, 3), [0,1]
        gt_dir = self.gt_path_files[index]
        img_name = os.path.basename(gt_dir).split(".")[0]
        image_dir = join(self.img_path, str(img_name)+img_name_suffix) 

        img_256, gt2D = preprocess(image_dir, gt_dir, self.bbox_shift, size=self.img_size)
        img_256 = np.transpose(img_256, (2, 0, 1))
        # print("np.max(img_256)", np.max(img_256))
        # print("np.min(img_256)", np.min(img_256))
        # assert (
        #     np.max(img_256) <= 1.0 and np.min(img_256) >= 0.0
        # ), "image should be normalized to [0, 1]"
        
        bboxes = [0, 0, self.img_size, self.img_size]
        return (
            torch.tensor(img_256).float(),
            torch.tensor(gt2D[None, :, :]),
            torch.tensor(bboxes).float(),
            img_name,
        )

# %% set up parser
parser = argparse.ArgumentParser()
parser.add_argument(
    "--img_path",
    type=str,
    default="data/DFUC2022_train_val/train/DFUC2022_train_images/",
    help="path to training image files",
)
parser.add_argument(
    "--gt_path",
    type=str,
    default="data/DFUC2022_train_val/train/DFUC2022_train_masks/",
    help="path to training groundtruth masks",
)
parser.add_argument(
    "--val_gt_path",
    type=str,
    default="data/DFUC2022_train_val/validation/labels/",
    help="path to validation groundtruth masks",
)
parser.add_argument(
    "--val_img_path",
    type=str,
    default="data/DFUC2022_train_val/validation/images/",
    help="path to validation images",
)

parser.add_argument("-task_name", type=str, default="SAM-ViT-H")
parser.add_argument("-model_type", type=str, default="vit_h")
parser.add_argument("-img_size", type=int, default=256)
parser.add_argument(
    "-checkpoint", type=str, default="work_dir/SAM-ViT-H-20240807-0536/sam_model_best_val_accuracy.pth"
)
parser.add_argument(
    "--load_pretrain", type=bool, default=False, help="use wandb to monitor training"
)
parser.add_argument("-work_dir", type=str, default="./work_dir")
# train
parser.add_argument("-num_epochs", type=int, default=200)
parser.add_argument("-batch_size", type=int, default=4)
parser.add_argument("-num_workers", type=int, default=8)
# Optimizer parameters
parser.add_argument(
    "-weight_decay", type=float, default=1e-5, help="weight decay (default: 0.1)"
)
parser.add_argument(
    "-lr", type=float, default=1e-4, metavar="LR", help="learning rate (absolute lr)"
)
parser.add_argument("-use_amp", action="store_true", default=False, help="use amp")
parser.add_argument(
    "--resume", type=str, default="", help="Resuming training from checkpoint"
)
parser.add_argument("--device", type=str, default="cuda:0")
parser.add_argument("-output_name", type=str, default="sam_ckpt_latest.pth", help="preferred name for finished checkpoint")

args = parser.parse_args()
# %% sanity test of dataset class
# tr_dataset = NpyDataset(args.img_path, args.gt_path)
# tr_dataloader = DataLoader(tr_dataset, batch_size=8, shuffle=True)
# for step, (image, gt, bboxes, names_temp) in enumerate(tr_dataloader):
#     print("Sanity check", image.shape, gt.shape, bboxes.shape)
#     # show the example
#     for idx in range(8):
#         _, axs = plt.subplots(1, 2, figsize=(10, 10))
#         axs[0].imshow(image[idx].cpu().permute(1, 2, 0).numpy())
#         show_mask(gt[idx].cpu().numpy(), axs[0])
#         show_box(bboxes[idx].numpy(), axs[0])
#         axs[0].axis("off")
#         axs[0].set_title(names_temp[idx])

#         axs[1].imshow(gt[idx].cpu().permute(1, 2, 0).numpy())
#         axs[1].axis("off")
#         axs[1].set_title(names_temp[idx])
#         plt.subplots_adjust(wspace=0.01, hspace=0)
#         plt.savefig("./data_sanitycheck{i}.png".format(i=str(idx)), bbox_inches="tight", dpi=300)
#         plt.close()
#         print("done sanity check ", str(idx)) 
#         break  
#     break

# %% set up model for training
# device = args.device
run_id = datetime.now().strftime("%Y%m%d-%H%M")
model_save_path = join(args.work_dir, args.task_name + "-" + run_id)
device = torch.device(args.device)
# %% set up model
class sam(nn.Module):
    def __init__(
        self,
        image_encoder,
        mask_decoder,
        prompt_encoder,
    ):
        super().__init__()
        self.image_encoder = image_encoder
        self.mask_decoder = mask_decoder
        self.prompt_encoder = prompt_encoder
        # freeze prompt encoder
        for param in self.prompt_encoder.parameters():
            param.requires_grad = False

    def forward(self, image, box):
        image = image.clone().detach().requires_grad_(True)
        image_embedding = self.image_encoder(image) 
        # image_embedding = self.image_encoder(torch.tensor(image))  
        
        # do not compute gradients for prompt encoder
        with torch.no_grad():
            box_torch = torch.as_tensor(box, dtype=torch.float32, device=image.device)
            if len(box_torch.shape) == 2:
                box_torch = box_torch[:, None, :]  # (B, 1, 4)

            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=None,
                boxes=box_torch,
                masks=None,
            )
        low_res_masks, _ = self.mask_decoder(
            image_embeddings=image_embedding,  # (B, 256, 64, 64)
            image_pe=self.prompt_encoder.get_dense_pe(),  # (1, 256, 64, 64)
            sparse_prompt_embeddings=sparse_embeddings,  # (B, 2, 256)
            dense_prompt_embeddings=dense_embeddings,  # (B, 256, 64, 64)
            multimask_output=False,
        )
        ori_res_masks = F.interpolate(
            low_res_masks,
            size=(image.shape[2], image.shape[3]),
            mode="bilinear",
            align_corners=False,
        )
        return ori_res_masks


def main():
    os.makedirs(model_save_path, exist_ok=True)
    shutil.copyfile(
        __file__, join(model_save_path, run_id + "_" + os.path.basename(__file__))
    )
    
    sam_model = _sam_model_registry[args.model_type](checkpoint=args.checkpoint, 
                                                    custom_img_size=args.img_size)
    
    sam_model = sam(
        image_encoder=sam_model.image_encoder,
        mask_decoder=sam_model.mask_decoder,
        prompt_encoder=sam_model.prompt_encoder,
    ).to(device)
    sam_model.train()

    print(
        "Number of total parameters: ",
        sum(p.numel() for p in sam_model.parameters()),
    )  
    print(
        "Number of trainable parameters: ",
        sum(p.numel() for p in sam_model.parameters() if p.requires_grad),
    )  

    img_mask_encdec_params = list(sam_model.image_encoder.parameters()) + list(
        sam_model.mask_decoder.parameters()
    )
    
    optimizer = torch.optim.AdamW(
        img_mask_encdec_params, lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = ReduceLROnPlateau(optimizer, 
                                  mode="min", 
                                  factor=0.1, 
                                  patience=10, 
                                  threshold=0.0001, 
                                  threshold_mode="rel",
                                  eps=1e-30)
    print(
        "Number of image encoder and mask decoder parameters: ",
        sum(p.numel() for p in img_mask_encdec_params if p.requires_grad),
    )  
    dice_loss = monai.losses.DiceLoss(sigmoid=True, squared_pred=True, reduction="mean")
    ce_loss = nn.BCEWithLogitsLoss(reduction="mean")
    # %% train
    num_epochs = args.num_epochs
    validation_losses = []
    training_losses = []
    best_validation_loss = 1e10
    learning_rates = []
    mean_IoU_lst = []
    best_meanIoU = 0
    best_dsc = 0
    dsc_lst = []
    
    
    train_dataset = NpyDataset(args.img_path, args.gt_path, bbox_shift=5, img_size=args.img_size)
    val_dataset = NpyDataset(args.val_img_path, args.val_gt_path, bbox_shift=5, img_size=args.img_size)

    print("Number of training samples: ", len(train_dataset))
    print("Number of validation samples: ", len(val_dataset))
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
            pin_memory=True,
        )
    start_epoch = 0
    if args.resume is not None:
        if os.path.isfile(args.resume):
            ## Map model to be loaded to specified single GPU
            checkpoint = torch.load(args.resume, map_location=device)
            print("epoch of checkpoint: ", checkpoint["epoch"] + 1)
            start_epoch = checkpoint["epoch"] + 1
            sam_model.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])
    if args.use_amp:
        scaler = torch.cuda.amp.GradScaler()

    for epoch in range(start_epoch, start_epoch+num_epochs):
        print("epoch: ", epoch)
        val_loss = 0
        epoch_loss = 0
        for step, (image, gt2D, boxes, _) in enumerate(tqdm(train_dataloader)):
            optimizer.zero_grad()
            boxes_np = boxes.detach().cpu().numpy()
            image, gt2D = image.to(device), gt2D.to(device)
            sam_model.train()
            if args.use_amp:
                ## AMP
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    sam_pred = sam_model(image, boxes_np)
                    loss = (dice_loss(sam_pred, gt2D) + ce_loss(sam_pred, gt2D.float()))/2
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                optimizer.step()
            else:
                sam_pred = sam_model(image, boxes_np)
                loss = (dice_loss(sam_pred, gt2D) + ce_loss(sam_pred, gt2D.float()))/2
                loss.backward()
                optimizer.step()

            epoch_loss += loss.item()
        epoch_loss /= step+1
        training_losses.append(epoch_loss)
        
        for step, (image, gt2D, boxes, _) in enumerate(tqdm(val_dataloader)):
            sam_model.eval()
            sam_model.zero_grad(set_to_none=True)
            boxes_np = boxes.detach().cpu().numpy()
            image, gt2D = image.to(device, non_blocking=True), gt2D.to(device, non_blocking=True)
            with torch.inference_mode():
                sam_pred = sam_model(image, boxes_np)
            val_epoch_loss = (dice_loss(sam_pred, gt2D) + ce_loss(sam_pred, gt2D.float()))/2
            val_loss += val_epoch_loss.item()
        #step=num_batches-1
        val_loss /= step+1
        validation_losses.append(val_loss)         
        print("val loss: ", val_loss)
        lr = scheduler.get_last_lr()
        learning_rates.append(lr)
        print("learning rate: ", lr)
        scheduler.step(val_loss)
  
        # save the latest model
        checkpoint = {
            "model": sam_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
        }
        checkpoint_path = join(model_save_path, args.output_name)
        torch.save(checkpoint, checkpoint_path)
        
        posttrain_model_path = convert(ckpt_path=checkpoint_path, 
                                            sam_path=args.checkpoint, 
                                            save_path=model_save_path+"/sam_model_latest.pth", 
                                            multi_gpu_ckpt=False)
        
        # Validation accuracy
        mean_actual_iou, mean_dsc = sam_eval(
                gt_path=args.val_gt_path,
                img_path=args.val_img_path,
                single_outpath="",
                multi_outpath="",
                checkpoint=posttrain_model_path,
                df_filename="latest_val_scores.csv",
                model_type=args.model_type,
                ratio=0.1)
        
        mean_IoU_lst.append(mean_actual_iou)
        dsc_lst.append(mean_dsc)
        if mean_dsc > best_dsc or (mean_dsc == best_dsc and mean_actual_iou > best_meanIoU):
            best_dsc = mean_dsc
            if mean_actual_iou > best_meanIoU:
                best_meanIoU = mean_actual_iou
            posttrain_model_path = convert(ckpt_path=checkpoint_path, 
                                            sam_path=args.checkpoint, 
                                            save_path=model_save_path+"/sam_model_best_val_accuracy.pth", 
                                            multi_gpu_ckpt=False)
        
        if val_loss < best_validation_loss:
            best_validation_loss = val_loss
            best_checkpoint_path = join(model_save_path, "sam_ckpt_best_val_loss.pth")
            torch.save(checkpoint, best_checkpoint_path)

        # %% plot learning rates, training and validation
        plt.yscale("log")
        plt.plot(learning_rates)
        plt.title("Learning Rates")
        plt.xlabel("Epoch")
        plt.ylabel("Learning Rates")
        plt.savefig(join(model_save_path, args.task_name + "_learning_rates.png"))
        plt.close()

        plt.plot(training_losses)
        plt.title("Train Weighted Average Dice + Cross Entropy Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Train Loss")
        plt.savefig(join(model_save_path, args.task_name + "_train_loss.png"))
        plt.close()
        
        plt.plot(validation_losses)
        plt.title("Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Val Loss")
        plt.savefig(join(model_save_path, args.task_name + "_validation_loss.png"))
        plt.close()
        
        plt.plot(dsc_lst)
        plt.title("Validation DSC Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Validation Dice Accuracy")
        plt.savefig(join(model_save_path, args.task_name + "_validation_dsc_accuracy.png"))
        plt.close()

        plt.plot(mean_IoU_lst)
        plt.title("Validation meanIoU Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Validation meanIoU Accuracy")
        plt.savefig(join(model_save_path, args.task_name + "_validation_meanIoU_accuracy.png"))
        plt.close()


if __name__ == "__main__":
    main()
