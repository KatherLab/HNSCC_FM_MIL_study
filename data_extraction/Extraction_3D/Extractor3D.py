#%%
import os
import json
from typing import Sequence
import torch
from torch.utils.data import ConcatDataset
from pathlib import Path
from torchvision import transforms
import numpy as np
from tqdm import tqdm
import h5py
import nibabel as nib
from datasets import CubeFoundation_NiftiDataset
from transforms_3D import RotateAroundZ, AddGaussianNoise, Grayscale_Norm3d
           
def extract_3Dfeatures_(
        *,
        model, 
        model_name="foundation", 
        paths: Sequence[Path],
        img_name : str ="image",
        mask_name : str ="mask_GTVp",
        body_name : str="body",
        h5_names:Sequence[str], 
        outdir: Path, 
        sampling_repetitions: int = 100,
        augmented_repetitions: int= 1,
        augmentations: bool = True, 
        min :int = -1024, 
        max: int = 2048,
        out_vol : int =50,
        sampling_variance: int=16
        ) -> None:
    """
    Extracts features from cubes samples from the Centre of Mass (CoM) of a segmentation mask.
    Args:
        model: model instance from torch from the RadImageNet.py script
        model_name: str, Name of the model used. Either foundation (Aerts) or resnet-50. They have different preprocessing
        paths: Sequence[Paths] A list of paths containing the path to where the features will be deposited.
        img_name: str, name of the NIFTI image containing the CT (the .nii.gz termination is not necessary)
        mask_name: str, name of the NIFTI image containing the ROI segmentation (the .nii.gz termination is not necessary)
        body_name: str, name of the NIFTI image containing the segmentation of the body (the .nii.gz termination is not necessary)
        h5_names:Sequence[str], names of the patients in a list from Caller3D
        outdir: Path, Path to save the features to.
        sampling repetitions: int, Number of times to sample for subvolumes around the CoM of the provided mask
        augmented_repetitions: int, How many additional iterations over the 
        dataset with augmentation should be performed (default=1).
        augmentations: bool, whether to perform augmentation at all (default=True).
        min: int, minimum for the HU window of the CTs
        max: int, maximum for the HU window of the CTs
        out_vol: int, volume size fed into the models, default 50 (50x50x50 voxels), others are untested for the foundation model!
        sampling variance: int, variance to sample the centre of the new points from a multivariate normal in the Dataset, default: 16
    """
    np.random.seed(42) #Set seed for reproducibility of all numpy distributions and thus sampling
    assert min<max, "The minimum range value should be smaller than the maximum range value"

    #Set parameters according to model specifications; the FM needs that specific range as it is the one it was pretrained on
    #ImageNet models are more flexible as they can clip to arbitrary ranges to then make those "grayscale" and fool the model
    #Transforms depend on the model; augmentations are always to rotate around the axial dimension (the first one in nibabel)
    #Then to ad noise and blurring.
    #Models are then trained by putting either augmented or non-augmented features each batch.

    if model_name=="foundation":
        if min!= 1024 or maxi!= 2048:
            print("For the foundation model we need to follow pretraining specificatuions so the minimum and maximum HU are set to -1024 and 2048")
        mini=-1024
        maxi=2048
        non_aug_transform= transforms.Compose([transforms.ToTensor()])
        if augmentations:
                    augmenting_transform = transforms.Compose([transforms.ToTensor(),
                                                RotateAroundZ(10),
                                                transforms.RandomApply([transforms.GaussianBlur(3)], p=.5),
                                                AddGaussianNoise(0., 1.0)]) 
    if model_name=="resnet-50":
        mini=min #We use the same HU window as the foundation model
        maxi=max
        mean=[0.485, 0.456, 0.406]
        std=[0.229, 0.224, 0.225]
        #Extra transforms are applied for the Inflated ResNet and preprocessing as it is not a CT-native model
        non_aug_transform= transforms.Compose([transforms.ToTensor(),
                                               Grayscale_Norm3d(mean=mean,std= std),
                                              ])
        if augmentations:
            augmenting_transform = transforms.Compose([transforms.ToTensor(),
                                                      RotateAroundZ(10),
                                                      Grayscale_Norm3d(mean=mean, std=std),
                                                      transforms.RandomApply([transforms.GaussianBlur(3)], p=.5),
                                                      AddGaussianNoise(0., 1.0)]) 
    outdir = Path(outdir)
    outdir.mkdir(exist_ok=True, parents=True)

    #Dump info of the extraction into a .json folder
    with open(outdir/'info.json', 'w') as f:
        json.dump({'extractor': model_name,
                   "augmentations" : augmentations,
                   "minimum_HU" : mini,
                   "maximum_HU" : maxi,
                   "sampled_repetitions" : sampling_repetitions,
                    "Augmentations": augmentations,
                  "augmented_repetitions": augmented_repetitions,
                  "sample_variance": sampling_variance}, f)

    for gen_path, folder_name in zip(paths,h5_names):
        # check if h5 for slide already exists / slide_tile_path path contains tiles
        gen_path=Path(gen_path)
        if (h5outpath := outdir/f'features_{folder_name}.h5').exists():
            print(f'{h5outpath} already exists.  Skipping...')
            continue
        if not next(gen_path.glob('*.nii.gz'), False):
            print(f'No tiles in {gen_path}.  Skipping...')
            continue
       
        #Load the image and the mask
        #Remember to deallocate the memory!
        im= np.asarray(nib.load(os.path.join(gen_path, img_name + ".nii.gz")).dataobj) #This just gives us the array, but instead does not store pointers in memory, mitigating
        msk = np.asarray(nib.load(os.path.join(gen_path, mask_name + ".nii.gz")).dataobj) #This just gives us the array, but instead does not store pointers in memory, mitigating  
        body = np.asarray(nib.load(os.path.join(gen_path, body_name + ".nii.gz")).dataobj) #This just gives us the array, but instead does not store pointers in memory, mitigating  
        #Mask out-of-body elements and remove body mask:
        im[body==0] = -2048
        del body
        #The last slide of the image and slide are usually faulty after resampling to isotropic voxel spacing, substitute by the minimum -2048 and 0:
        im[:,:,-1] = -2048
        msk[:,:,-1] = 0

        #Get the indices (LAS oriented) of the cubes to put them into the h5:
        #As we are always using the same seed, the samplings should be reproducible
        CoM =  np.mean(np.argwhere(msk==1), axis=0)
        sample_centres = [np.random.normal(0,np.sqrt(sampling_variance), sampling_repetitions) + CoM[a] for a in np.arange(3)]
        sample_centres = list(zip(*sample_centres)) #This creates a list of (nreps) with each being the centre of the new cube to sample for the model
        sample_centres.append(CoM)
        #Also append the Centre of Mass of the lesion itself :)
        
        ######################################################
        unaugmented_ds = CubeFoundation_NiftiDataset(img=im,
                                                     mask= msk,model_name=model_name,sample_variance= sampling_variance,
                                                     transform=non_aug_transform,
                                                     repetitions=sampling_repetitions,
                                                     cube_size=out_vol, min=mini, max=maxi)

        if augmentations:
            augmented_ds = CubeFoundation_NiftiDataset(img=im,
                                                     mask= msk, model_name=model_name,
                                                     transform=augmenting_transform, sample_variance= sampling_variance,
                                                     repetitions=sampling_repetitions,
                                                     cube_size=out_vol, min=mini, max=maxi)
            
            ds = ConcatDataset([unaugmented_ds, augmented_ds])
        
        else:
            ds= unaugmented_ds

        dl = torch.utils.data.DataLoader(
            ds, batch_size=1, shuffle=False, num_workers=os.cpu_count(), drop_last=False)

        feats = []
        for batch in tqdm(dl, leave=False):
            feats.append(
                model(batch.type_as(next(model.parameters()))).half().cpu().detach())
        #Delete non-useful variables at this point to deallocate memory#
        del im
        del msk 
        del ds

        with h5py.File(h5outpath, 'w') as f:
            if augmentations:
                try:     
                    f['location'] = [idxs for idxs in sample_centres] + [idxs for idxs in sample_centres]
                    f['feats'] = torch.concat(feats).cpu().numpy()
                    f['augmented'] = np.repeat(
                    [False, True], [len(unaugmented_ds), len(augmented_ds)])
                    f.attrs['extractor'] = model_name
                    print(f.attrs['extractor'])
                except:
                    print('Error with file naming, no location given')
            else:
                try:        
                    f['location'] = [idxs for idxs in sample_centres]
                    f['feats'] = torch.concat(feats).cpu().numpy()
                    f['augmented'] = np.repeat(
                    [False], [len(unaugmented_ds)])
                    f.attrs['extractor'] = model_name
                except:
                    print('Error with file naming, no location given')
        del CoM
        del sample_centres
        del unaugmented_ds 
        del augmented_ds