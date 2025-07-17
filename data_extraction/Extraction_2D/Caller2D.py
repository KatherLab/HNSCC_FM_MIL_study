#%%
import torch
import torchvision
from getcuda import get_free_gpu_indices
from Extractor2D import extract_features_2D_mask
import glob as glob
import os
from pathlib import Path
from open_clip import create_model_from_pretrained, get_tokenizer
import argparse
#########################################
#Parser
#########################################

parser = argparse.ArgumentParser(description="Inputs for the extractors. Make sure all images and segmentations follow LAS orientation!")
parser.add_argument('--pth', type=str, required=True, help="Path to where the NIFTI images with segmentations are, each folder in the path should be one patient")
parser.add_argument('--pth_out', type=str, required=True, help = "Output directory where you want the features")
parser.add_argument("--window", type=int, default=120, required=True, help="Window range for the HU clipping")
parser.add_argument("--level", type=int, default=50, required=True, help="level for the middle point of the HU windowing")
parser.add_argument("--image",  type=str, required=True, default="image", help="name of the CT NIFTI, must be the same for all patients")
parser.add_argument("--mask",  type=str, required=True, default="GTV", help="Name of the segmentation in NIFTI format that contains the region of interest")
parser.add_argument("--body", type=str, required=True, default="body", help="Name of the segmentation containing the body of the patient to mask the patient bed")
parser.add_argument("--crop_size", type=int, required=False, default=224, help="Cropping that will be done around the CoM of the region of interest to then input the image into the extractor")
parser.add_argument("--model",type=str, default="biomedclip", required=True, help="model name to choose an extractor; allowed ones are: swin-imagenet,vit-imagenet,resnet50-imagenet,biomedclip, defualt is biomedclip")

args= parser.parse_args()

model= args.model
pth= args.pth
outdir= args.pth_out
window= args.window
level= args.level
image=args.image
mask = args.mask
body= args.body 
if args.crop_size:
    cropping= args.crop_size
else:
    cropping=224
feat_names = os.listdir(pth)
pth = [os.path.join(pth,j) for j in os.listdir(pth)]

#########################################
#Caller definition
##########################################

def extract_features_2D_(slide_tile_paths, model_name, **kwargs):
    assert model_name in ["swin-imagenet","vit-imagenet","resnet50-imagenet","biomedclip"], "Available models are: ViT_b_32 (vit-imagenet),ResNet50 (resnet50-imagenet),BioMedClip (biomedclip) and swin-vit"
    if model_name == "swin-imagenet":
        model = torchvision.models.swin_b(weights="IMAGENET1K_V1") 
        tokenizer=None
    if model_name == "vit-imagenet":
        model = torchvision.models.vit_b_32(weights="IMAGENET1K_V1")  
        tokenizer=None
    if model_name == "resnet50-imagenet":
        model = torchvision.models.resnet50(weights="IMAGENET1K_V2") 
        tokenizer=None
    if model_name== "biomedclip":
        model , pre = create_model_from_pretrained("hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224")
        del pre 
        tokenizer = get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
    model.fc = torch.nn.Identity()
    device_ids = get_free_gpu_indices()
    
    if len(device_ids)==0:
        device = 'cpu'
    else:
        device='cuda:'+str(device_ids[0])
    print(device)
    model = model.eval().to(device) #device is needed for the BioMedClip features
   
    return extract_features_2D_mask(slide_tile_paths=slide_tile_paths, 
                                    seg_paths=slide_tile_paths, device=device,
                                    model=model, model_name=model_name,
                                    toknzer=tokenizer, **kwargs)
    





extract_features_2D_(slide_tile_paths=pth,
                           model_name= model,    
                                       img_name= image,
                                       mask_name = mask,
                                       outdir=outdir,
                                       augmented_repetitions=1,
                                       body_name=body,
                                       level=level,
                                       window= window,
                                       crop_size=cropping, 
                                       h5_names=feat_names)

