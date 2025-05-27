
import torch
from Extractor3D import extract_3Dfeatures_
from getcuda import get_free_gpu_indices
import glob as glob
import os
from pathlib import Path
from fmcib.models import fmcib_model
from typing import  Sequence
from inflated_ResNet50 import run_inflater
import argparse

parser = argparse.ArgumentParser(description="Inputs for the extractors. Make sure all images and segmentations follow LAS orientation!")
parser.add_argument('--pth', type=str, required=True, help="Path to where the NIFTI images with segmentations are, each folder in the path should be one patient")
parser.add_argument('--pth_out', type=str, required=True, help = "Output directory where you want the features")
parser.add_argument("--min", type=int, default=-1024, required=True, help="Minimum value for the HU windowing")
parser.add_argument("--max", type=int, default=2048, required=True, help="Maximum value for the HU windowing")
parser.add_argument("--image",  type=str, required=True, default="image", help="name of the CT NIFTI, must be the same for all patients, must have .nii.gz termination and the termination must NOT be passed")
parser.add_argument("--mask",  type=str, required=True, default="GTV", help="Name of the segmentation in NIFTI format that contains the region of interest, must have .nii.gz termination and the termination must NOT be passed")
parser.add_argument("--body", type=str, required=True, default="body", help="Name of the segmentation containing the body of the patient to mask the patient bed, must have .nii.gz termination and the termination must NOT be passed")
parser.add_argument("--volume", type=int, required=False, default=50, help="shape of the cubic volumes extracted to pass into the models; default is 50 (so we extract volumes of 50x50x50 voxels)")
parser.add_argument("--model",type=str, default="foundation", required=True, help="model name to choose an extractor; allowed ones are: foundation or resnet50 , defualt is foundation")
parser.add_argument("--sampling_reps",type=int, default=100, required=True, help="Number of volumes sampled around the CoM of the region of interest, default is 100")
parser.add_argument("--sampling_var",type=int, default=16, required=True, help="variance for the multivariate normal distribution for subvolume sampling, default is 16")

args= parser.parse_args()

model_name= args.model
pth= args.pth
outdir= args.pth_out
min= args.min
max= args.max
image=args.image
mask = args.mask
body= args.body 
variance = args.sampling_var
reps_sampling = args.sampling_reps

if args.volume:
    vols= args.volume
else:
    vols=50
feat_names = os.listdir(pth)
pth = [Path(os.path.join(pth,j)) for j in os.listdir(pth)]


def extract_3D_features_(paths:Sequence[Path], model_name:str="foundation", out_dir= Path, **kwargs):
    """
    Extracts features into .h5 files and saves them into a directory; this is done with 3D models.
    args:
    paths: sequence of Path elements that contain the path to the folder containing the image and mask NIFTIs
    model_name: string indicating the type of model to use as an extractor. Options are foundation and MedNet
    out_dir: Path specifying the directory to save the .h5 files
    **kwargs: keyword arguments passed into the function extract_3D_features
    """
    if model_name =="foundation":
        model = fmcib_model()
    if model_name == "resnet-50":
        model=run_inflater(ResNet_dim=50) #This sets the type of resnet
    else:
        ValueError("Please select a model type between foundation model (foundation) or Inflated ResNet50 (resnet-50).")
    model.fc = torch.nn.Identity()
    #device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device_ids = get_free_gpu_indices()
    
    if len(device_ids)==0:
        device = 'cpu'
    else:
        device='cuda:'+str(device_ids[0])

    print(device)
    model = model.eval().to(device)

    return extract_3Dfeatures_(paths=paths, 
                               model=model, 
                               model_name=model_name, 
                               outdir= out_dir, 
                               **kwargs)



extract_3D_features_(paths=pth,
                            model_name= model_name,
                            out_dir=outdir,
                            augmented_repetitions=1, #One round of augmented features :)
                            sampling_repetitions= reps_sampling,
                            sampling_variance = variance,
                            augmentations= True,
                            min  = min, 
                            max = max,
                            h5_names=feat_names,
                            mask_name =mask,
                            body_name=body
                            )

