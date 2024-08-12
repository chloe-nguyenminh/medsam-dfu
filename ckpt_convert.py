# -*- coding: utf-8 -*-
import torch
import argparse


# %% convert medsam model checkpoint to sam checkpoint format for convenient inference

def convert(
    ckpt_path: str,
    sam_path: str,
    save_path: str,
    multi_gpu_ckpt: bool): 
    sam_ckpt = torch.load(sam_path)
    medsam_ckpt = torch.load(ckpt_path)
    sam_keys = sam_ckpt.keys()
    for key in sam_keys:
        if not multi_gpu_ckpt:
            sam_ckpt[key] = medsam_ckpt["model"][key]
        else:
            sam_ckpt[key] = medsam_ckpt["model"]["module." + key]

    torch.save(sam_ckpt, save_path)
    return save_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="convert MedSAM checkpoint to SAM checkpoint format"
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        required=True,
        help="path to the trained MedSAM checkpoint",
    )
    parser.add_argument(
        "--sam_path",
        type=str,
        default="work_dir/sam_vit_b_01ec64.pth",
        help="path to SAM model",
    )
    parser.add_argument(
        "--save_path",
        default="work_dir/sam_model_latest.pth",
        help="path to save converted sam checkpoint",
    )
    parser.add_argument(
        "--multi_gpu_ckpt",
        type=bool,
        default=False,
        help="True if the model is trained with multiple gpus"
    )
    args = parser.parse_args()
    convert(args.ckpt_path, args.sam_path, args.save_path, args.multi_gpu_ckpt)
