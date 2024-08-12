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
from segment_anything.build_sam_lora import sam_lora_model_registry
from sam_lora_image_encoder import LoRA_Sam
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
from sam_lora_inference import *
from ckpt_convert import convert
from random import sample
import json
from importlib import import_module
import torch.backends.cudnn as cudnn
from sklearn import metrics
from sklearn.metrics import confusion_matrix, f1_score, jaccard_score

# set seeds
torch.manual_seed(2024)
torch.cuda.manual_seed(2024)
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

#%% Utils function
def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([251 / 255, 252 / 255, 30 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

dice_loss_fn = monai.losses.DiceLoss(sigmoid=True, squared_pred=True, reduction="mean")
ce_loss_fn = nn.BCEWithLogitsLoss(reduction="mean")

def weighted_loss_fn(outputs, labels, dice_weight:float=0.8):
    dice_loss_fn = monai.losses.DiceLoss(sigmoid=True, squared_pred=True, reduction="mean")
    ce_loss_fn = nn.BCEWithLogitsLoss(reduction="mean")
    logits = outputs['low_res_logits']
    ce_loss = ce_loss_fn(logits, labels)
    dice_loss = dice_loss_fn(logits, labels)
    weighted_loss = (1 - dice_weight) * ce_loss + dice_weight * dice_loss
    return weighted_loss, ce_loss, dice_loss

#%% Define Dataset class
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
    def __len__(self):
        return len(self.gt_path_files)

    def __getitem__(self, index):
        # load npy image (256, 256, 3), [0,1]
        gt_dir = self.gt_path_files[index]
        img_name = os.path.basename(gt_dir).split(".")[0]
        image_dir = join(self.img_path, str(img_name)+img_name_suffix) 

        img_256, gt2D = preprocess(image_dir, gt_dir, self.bbox_shift, size=self.img_size)
        img_256 = np.transpose(img_256, (2, 0, 1))
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
parser.add_argument('--module', type=str, default='sam_lora_image_encoder')

parser.add_argument("-task_name", type=str, default="SAM-ViT-H-LORA")
parser.add_argument("-model_type", type=str, default="vit_h")
parser.add_argument("-img_size", type=int, default=256)
parser.add_argument(
    "-checkpoint", type=str, default="work_dir/SAM-ViT-H-20240723-1333/sam_model_best_val_accuracy.pth"
)
parser.add_argument(
    "--load_pretrain", type=bool, default=False, help="use wandb to monitor training"
)
parser.add_argument("-pretrain_model_path", type=str, default="")
parser.add_argument("-work_dir", type=str, default="./work_dir")
# train
parser.add_argument("-num_epochs", type=int, default=400)
parser.add_argument("-batch_size", type=int, default=16)
parser.add_argument("-num_workers", type=int, default=8)
# Optimizer parameters
parser.add_argument(
    "-weight_decay", type=float, default=1e-4, help="weight decay (default: 0.1)"
)
parser.add_argument(
    "-lr_exp", type=float, default=7, help="learning rate decay during warmup"
)
parser.add_argument(
    "-lr", type=float, default=1e-4, metavar="LR", help="learning rate (absolute lr)"
)
parser.add_argument("-use_amp", action="store_true", default=True, help="use amp")
parser.add_argument(
    "--resume", type=str, default="", help="Resuming training from checkpoint"
)
parser.add_argument("--device", type=str, default="cuda:0")
parser.add_argument("-output_name", type=str, default="sam_lora_ckpt_latest.pth", help="preferred name for finished checkpoint")

# new args from sam_lora
parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
parser.add_argument('--deterministic', type=int, default=1,
                    help='whether use deterministic training')
parser.add_argument('--tf32', action='store_true', help='If activated, use tf32 to accelerate the training process')
parser.add_argument('--compile', action='store_true', help='If activated, compile the training model for acceleration')
parser.add_argument('--use_amp', action='store_true', help='If activated, adopt mixed precision for acceleration')
parser.add_argument('--num_classes', type=int,
                    default=1)
parser.add_argument('--rank', type=int, default=4, help='Rank for LoRA adaptation')
parser.add_argument('--max_iterations', type=int,
                    default=30000, help='maximum  number to train')
parser.add_argument('--max_epochs', type=int,
                    default=400, help='maximum epoch number to train')
parser.add_argument('--stop_epoch', type=int,
                    default=300, help='maximum epoch number to train')
parser.add_argument('--warmup', action='store_true', help='If activated, warp up the learning from a lower lr to the lr')
parser.add_argument('--warmup_period', type=int, default=250,
                    help='Warp up iterations, only valid when warmup is activated')
parser.add_argument('--lora_ckpt', type=str, default=None, help='Finetuned lora checkpoint')

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
run_id = datetime.now().strftime("%Y%m%d-%H%M")
model_save_path = join(args.work_dir, args.task_name + "-" + run_id)
device = torch.device(args.device)
#%% Main Training Loop
def main():
    if args.tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True
        
    os.makedirs(model_save_path, exist_ok=True)
    shutil.copyfile(
        __file__, join(model_save_path, run_id + "_" + os.path.basename(__file__))
    )
    # Sam_Lora
    sam, _ = sam_lora_model_registry[args.model_type](checkpoint=args.checkpoint, 
                                                image_size=args.img_size,
                                                num_classes=args.num_classes,
                                                pixel_mean=[0.485, 0.456, 0.406],
                                                pixel_std=[0.229, 0.224, 0.225])
    
    pkg = import_module(args.module)
    sam_model = pkg.LoRA_Sam(sam, args.rank).cuda()
    if args.compile:
        mdel = torch.compile(net)
    if args.lora_ckpt is not None:
        sam_model.load_lora_parameters(args.lora_ckpt)

    if args.num_classes > 1:
        multimask_output = True
    else:
        multimask_output = False
    
    # Save model config for reference
    config_file = os.path.join(model_save_path, 'config.txt')
    config_items = []
    for key, value in args.__dict__.items():
        config_items.append(f'{key}: {value}\n')

    with open(config_file, 'w') as f:
        f.writelines(config_items)
        
    if args.n_gpu > 1:
        sam_model = nn.DataParallel(sam_model)
        args.batch_size *= args.n_gpu

    sam_model.train()

    print(
        "Number of total parameters: ",
        sum(p.numel() for p in sam_model.parameters()),
    )  
    trainable_params = [p for p in sam_model.parameters() if p.requires_grad==True]
    print(
        "Number of trainable parameters: ", sum(p.numel() for p in trainable_params if p.requires_grad)
        )
    
    if args.warmup:
        lr = args.lr / args.warmup_period
    else:
        lr = args.lr
        
    if args.use_amp:
        scaler = torch.cuda.amp.GradScaler()
        
    max_epoch, stop_epoch, max_iterations = args.max_epochs, args.stop_epoch, args.max_iterations
    

    optimizer = torch.optim.AdamW(trainable_params, 
                                  lr=lr, 
                                  betas=(0.9, 0.999),
                                  weight_decay=0.1
    )
    scheduler = ReduceLROnPlateau(optimizer, 
                                  mode="min", 
                                  factor=0.01, 
                                  patience=10, 
                                  threshold=1e-4, 
                                  threshold_mode="rel",
                                  eps=1e-30
    )
    
    # %% train
    num_epochs = args.num_epochs
    validation_losses = []
    training_losses = []
    best_val_loss = 1e10
    learning_rates = []
    total_iou_lst = []
    best_meanIoU = 0
    best_dice = 0
    total_dice_lst = []
    
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
    
    iter_num = 0

    for epoch in range(start_epoch, start_epoch+num_epochs):
        print("epoch: ", epoch)
        val_loss = 0
        epoch_loss = 0
        sam_model.train()
        for step, (image, gt2D, boxes, _) in enumerate(tqdm(train_dataloader)):
            boxes_np = boxes.detach().cpu().numpy()
            image, gt2D = image.to(device), gt2D.to(device)
            gt2D = gt2D.float()
            if args.use_amp:
                ## AMP
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=args.use_amp):
                    sam_pred = sam_model(image, multimask_output, args.img_size)
                    loss = (dice_loss_fn(sam_pred['low_res_logits'], gt2D) + ce_loss_fn(sam_pred['low_res_logits'], gt2D.float()))/2
                    weighted_loss, ce_loss, dice_loss = weighted_loss_fn(sam_pred, gt2D)
                    loss = weighted_loss
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    optimizer.step()
            else:
                sam_pred = sam_model(image, boxes_np)
                loss = (dice_loss(sam_pred, gt2D) + ce_loss(sam_pred, gt2D.float()))/2
                loss.backward()
                optimizer.step()
            epoch_loss += loss.item()
                
            if args.warmup and iter_num < args.warmup_perod:
                lr_ = base_lr * ((iter_num + 1)/ args.warmup_period)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_
            else:
                if args.warmup:
                    shift_iter = iter_num - args.warmup_period
                    assert shift_iter >= 0, f'Shift iter is {shift_iter}, smaller than zero'
                else:
                    shift_iter = iter_num
                lr_ = lr * (1.0 - shift_iter / max_iterations) ** args.lr_exp
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_
            # break

        iter_num += 1
        epoch_loss /= step+1
        training_losses.append(epoch_loss)
        
        sam_model.eval()
        iou_per_epoch, dice_per_epoch = 0, 0
        for step, (image, gt2D, boxes, _) in enumerate(tqdm(val_dataloader)):
            sam_model.zero_grad(set_to_none=True)
            boxes_np = boxes.detach().cpu().numpy()
            image, gt2D = image.to(device, non_blocking=True), gt2D.to(device, non_blocking=True)
            gt2D = gt2D.float()
            
            with torch.inference_mode():
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=args.use_amp):
                    sam_pred = sam_model(image, multimask_output, args.img_size)
                    weighted_loss, ce_loss, dice_loss = weighted_loss_fn(sam_pred, gt2D)
                    val_epoch_loss = weighted_loss
                    val_loss += val_epoch_loss.item()

        val_loss /= (step+1)
        validation_losses.append(val_loss)         
        print("val loss: ", val_loss)
        lr = scheduler.get_last_lr()[0]
        learning_rates.append(lr)
        print("learning rate: ", lr)
        scheduler.step(val_loss)
        
        checkpoint_path = join(model_save_path, args.output_name)
        try:
            sam_model.save_lora_parameters(checkpoint_path)
        except:
            sam_model.module.save_lora_parameters(checkpoint_path)
            
        dice_per_epoch, iou_per_epoch = sam_lora_eval(
                                                    gt_path=args.val_gt_path,
                                                    img_path=args.val_img_path,
                                                    single_outpath="",
                                                    multi_outpath="",
                                                    checkpoint=args.checkpoint,
                                                    df_filename="latest_val_scores.csv",
                                                    lora_ckpt = checkpoint_path,
                                                    model_type=args.model_type,
                                                    ratio=0.1)
        total_dice_lst.append(dice_per_epoch)
        total_iou_lst.append(iou_per_epoch)
        
        if dice_per_epoch > best_dice or (dice_per_epoch == best_dice and iou_per_epoch > best_meanIoU):
            best_dice = dice_per_epoch
            if iou_per_epoch > best_meanIoU:
                best_meanIoU = iou_per_epoch
                checkpoint_path = join(model_save_path, "sam_lora_best_acc.pth")
                try:
                    sam_model.save_lora_parameters(checkpoint_path)
                except:
                    sam_model.module.save_lora_parameters(checkpoint_path)
        
            
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_checkpoint_path = join(model_save_path, "sam_lora_best_val_loss.pth")
            try:
                sam_model.save_lora_parameters(checkpoint_path)
            except:
                sam_model.module.save_lora_parameters(checkpoint_path)
        
        train_log_dir = join(model_save_path, "training_log.json")
        train_log = {"lr": learning_rates,
                             "training_losses": training_losses,
                             "validation_losses": validation_losses,
                             "validation_dice": total_dice_lst,
                             "validation_mean_iou": total_iou_lst}
        
        with open(train_log_dir, "w") as file:
            json.dump(train_log, file)
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
        
        plt.plot(total_dice_lst)
        plt.title("Validation Dice")
        plt.xlabel("Epoch")
        plt.ylabel("Validation Dice")
        plt.savefig(join(model_save_path, args.task_name + "_validation_dice_accuracy.png"))
        plt.close()

        plt.plot(total_iou_lst)
        plt.title("Validation meanIoU")
        plt.xlabel("Epoch")
        plt.ylabel("Validation meanIoU")
        plt.savefig(join(model_save_path, args.task_name + "_validation_meanIoU_accuracy.png"))
        plt.close()


if __name__ == "__main__":
    main()
