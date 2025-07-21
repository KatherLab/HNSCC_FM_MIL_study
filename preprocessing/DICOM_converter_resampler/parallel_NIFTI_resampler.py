import SimpleITK as sitk
import numpy as np
import multiprocessing.pool as mp
from functools import partial
from pathlib import Path
import os
import argparse

################################################################################
#Parser
################################################################################
parser = argparse.ArgumentParser(description="Inputs for the Resampler fucntions. Make sure all images and segmentations follow LAS orientation!")
parser.add_argument('--pth', type=str, required=True, help="Path to the directory where the patient folders with NIFTI images are.")
parser.add_argument('--pth_out', type=str, required=True, help = "Output directory where you want the resampled_images")
parser.add_argument('--img_names', nargs='+', type=str,required=True, help='List of names with .nii.gz termination you want to resample e.g image.nii.gz GTV.nii.gz')
parser.add_argument("--n_workers", type=int, default=20, required=False, help="Input the number of CPUs that will be used, default:20")
parser.add_argument("--outdim", type= int, required=False, default=[1,1,1], help="3-element list with the spacing desired in LAS orientation in the outputted NIFTIs, defualt: [1,1,1]")
args= parser.parse_args()

pth= Path(args.pth_csv)
outdir= Path(args.pth_out)
im_names= args.im_names
n_works= args.n_workers
dims= args.outdim


################################################################################
#Function definition
################################################################################

def Resampler(img: sitk.Image, out_spacing: list, order: int=2):
    """
    This function changes the dimensions of the image to fit into a new voxel spacing

    The image is loaded through simpleITK

    Inputs:

    img: sitk Image containing the loaded scan from a NIFTI file

    out_spacing: list with the [x,y,z] resampling desired; make sure you are aware of the orientation you want! 
    """
    original_spacing = img.GetSpacing()
    print(original_spacing)
    original_size= img.GetSize()
    print(original_size)
    out_size = [
        int(np.round(original_size[0] * (original_spacing[0] / out_spacing[0]))),
        int(np.round(original_size[1] * (original_spacing[1] / out_spacing[1]))),
        int(np.round(original_size[2] * (original_spacing[2] / out_spacing[2])))]
    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(out_spacing)
    resample.SetSize(out_size)
    resample.SetOutputDirection(img.GetDirection())
    resample.SetOutputOrigin(img.GetOrigin())
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(img.GetPixelIDValue())
    if order==1:
        resample.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resample.SetInterpolator(sitk.sitkBSpline)
    return resample.Execute(img)

def image_and_mask_interpolator(pth:Path, out:Path,names: list, out_space= [1,1,1]):
    outer= os.path.join(out,Path(pth.stem))
    os.makedirs(name=outer, exist_ok=True)
    print(f"Transforming patient {pth.stem}")
    for name in names:
        print(name)
        try:
            im= sitk.ReadImage(os.path.join(pth,name))
            print("Image read")
            im= Resampler(img=im, out_spacing=out_space, order =2)
            sitk.WriteImage(image=im, fileName=os.path.join(outer,name))

        except:
            print(f"Could not reconvert {name} for patient {pth.stem}")


def paralleliser(in_path: Path,outdir: Path, names:list[str], num_workers: int=30, space_out:list[int]=[1,1,1]):
    """
    Paralleliser helper function to run resampling of a set of similarly-named NIFTI images into  resampled images of space_out voxel dimensions with bilinear interpolation.
    
    Inputs: 
    
    in_path: Path where the folders containing the nifti images re located; we assume that they will be immediately after each folder eg; in_path/PATIENT/image.nii.gz
    
    outdir: Path, the absolute path where the resampled images should be put, it will create a structure parallel to the input folder
    
    space_out: list[int], list of three elements that are the voxel dimeniosn of the resampled images; default is [1,1,1]. We assume LAS orientation
    """
    print(f"The output space will be {space_out}")

    if isinstance(in_path,Path):
        out= [Path(os.path.join(in_path, i)) for i in os.listdir(in_path)]
    os.makedirs(outdir,exist_ok=True)
    with mp.Pool(num_workers) as pl:
        try:
            pl.map(partial(image_and_mask_interpolator, out=outdir, names=names,out_space=space_out), out)
        except KeyboardInterrupt:
            print("Interrupted the conversion. Terminating workers")
    return "Finished resampling to model physical size :)"      


paralleliser(in_path=pth, outdir= outdir, num_workers=n_works, space_out=dims)
