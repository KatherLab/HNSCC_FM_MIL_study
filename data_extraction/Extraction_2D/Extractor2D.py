import os
import json
from typing import Sequence
import torch
from torch.utils.data import Dataset, ConcatDataset
from pathlib import Path
import nibabel as nib 
from torchvision import transforms
import numpy as np
from tqdm import tqdm
import h5py
from PIL import Image
from datetime import datetime
from scipy.ndimage import center_of_mass
import torchvision.transforms.functional as TF
from transforms_2D import AddGaussianNoise, TriChanneler, CenterMassCrop
from datasets import CT_slice_Grayscale_NIFTIDataset_mask, CT_slice_Grayscale_NIFTIDataset_mask_BioMedClip

         
def extract_features_2D_mask(
        *,
        model, 
        model_name, 
        slide_tile_paths: Sequence[Path], 
        seg_paths: Sequence[Path],
        img_name: str,
        mask_name: str,
        body_name: str,
        device: str,
        h5_names:Sequence[str], 
        outdir: Path, 
        augmented_repetitions: int = 1, 
        crop_size: int=224,
        level: int=50,
        window: int= 120,
        toknzer= None
) -> None:
    """Extracts features from CT slices for all three views of the image.

    Args
        model: model object passed for extraction
        model_name: str, name of the model, passed from the Caller
        seg_pths: str, paths of the segmentations of the tumor and the body
        img_name: str,name of the NIFTIs containing the CT (termination of .nii.gz not needed)
        mask_name: str
        device: name of the device used, passed from the Caller
        h5_names:  A list of paths containing the names of the outputed folders, one per slide.
        outdir:  Path to save the features to.
        augmented_repetitions:  How many additional iterations over the
            dataset with augmentation should be performed.  0 means that
            only one, non-augmentation iteration will be done.
        crop_sizte: int, the size that the CoM-centered and cropped image will be resized to.
        level: int, medium point of the HU scale used for windowing defualt: 50
        window: int, total range of the HU window, default: 120, so HU will be windowed between [-10,170]
        toknzer: used only for BioMedClip extraction, passed from Caller2D
    """
    with torch.no_grad():
        if model_name.endswith('-imagenet'):
            t_mean = [0.485, 0.456, 0.406]
            t_sd = [0.229, 0.224, 0.225]
        if model_name=="biomedclip":
            t_mean = [0.48145466, 0.4578275, 0.40821073] #Taken from biomedclip preprocessing
            t_sd = [0.26862954, 0.26130258, 0.27577711]
        #defining transforms
        normal_transform = transforms.Compose([
            transforms.Resize(224),
            TriChanneler(3),
            transforms.Normalize(t_mean,t_sd)])
        #transforms for the augmented features
        augmenting_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.RandomRotation(10),    
        TriChanneler(3),
        transforms.Normalize(t_mean,t_sd),
        transforms.RandomApply([transforms.GaussianBlur(3)], p=.5),
        AddGaussianNoise(0., 1.)])

        #äcreate output directory and dump information
        outdir = Path(outdir)
        outdir.mkdir(exist_ok=True, parents=True)
        with open(outdir/'info.json', 'w') as f:
            json.dump({'extractor': str(model_name),
                       'date' : str(datetime.now()),
                    'augmented_repetitions': augmented_repetitions,
                    'crop_size': crop_size,
                    'resize_size': 224,
                    'window':window,
                    'level':level}, f)

        for slide_tile_path, seg_path, folder_name in zip(slide_tile_paths,seg_paths,h5_names):
            if not next(Path(slide_tile_path).glob('*.nii.gz'), False):
                    print(f'No tiles in {slide_tile_path}.  Skipping...')
                    continue
            # check if h5 for slide already exists / slide_tile_path path contains tiles
            slide_tile_path=Path(slide_tile_path)
            im= np.asarray(nib.load(os.path.join(slide_tile_path, img_name + ".nii.gz")).dataobj) #This just gives us the array, but instead does not store pointers in memory, mitigating
            msk = np.asarray(nib.load(os.path.join(seg_path, mask_name + ".nii.gz")).dataobj) #This just gives us the array, but instead does not store pointers in memory, mitigating  
            bod=  np.asarray(nib.load(os.path.join(seg_path, body_name + ".nii.gz")).dataobj)
            #######################
            #Mask out non-body elements
            ######################
            im[:,:,-1] = -2048
            im[bod==0] = -2048 
            del bod
            #######################
            #Isotropic reconstructions have issues at the last (superior) slide :)
            #######################
            nonzero_indices = np.nonzero(msk)
            for i in np.unique(nonzero_indices[-1]):
                if (msk[:,:,i]==1).all() :
                    msk[:,:,i] =0
            nonzero_indices = np.nonzero(msk)

            for view in ["axial","coronal","sagittal"]: #extract for each view
                if (h5outpath := outdir/f'features_{folder_name}_{view+".h5"}').exists():
                    print(f'{h5outpath} already exists.  Skipping...')
                    continue
                if view =="axial":
                    non_zero = np.unique(nonzero_indices[2])
                    CoM= []
                    for i in non_zero:
                        CoM.append(center_of_mass(msk[:,:,i]))
                if view =="sagittal":
                    non_zero = np.unique(nonzero_indices[0])
                    CoM= []
                    for i in non_zero:
                        CoM.append(center_of_mass(msk[i,:,:]))
                if view =="coronal":
                    non_zero = np.unique(nonzero_indices[1])
                    CoM= []
                    for i in non_zero:
                        CoM.append(center_of_mass(msk[:,i,:]))
                
                if model_name !="biomedclip":
                    unaugmented_ds = CT_slice_Grayscale_NIFTIDataset_mask(
                                                                          im_array=im,
                                                                          CoM=CoM,
                                                                          msk_array=non_zero, 
                                                                          view=view, 
                                                                          transform=normal_transform, 
                                                                          window=window, 
                                                                          level=level,
                                                                          crop_size=crop_size)
                    augmented_ds = CT_slice_Grayscale_NIFTIDataset_mask( 
                                                                        im_array=im,
                                                                        CoM=CoM,
                                                                        msk_array=non_zero, 
                                                                        view=view, 
                                                                        transform=augmenting_transform,
                                                                        window=window, 
                                                                        level=level,
                                                                        crop_size=crop_size)
                    
                    ds= ConcatDataset([unaugmented_ds, augmented_ds])
                    dl = torch.utils.data.DataLoader(
                        ds, batch_size=1, shuffle=False, drop_last=False)
                    feats = []
                    for batch in tqdm(dl, leave=False):
                        feats.append(
                            model(batch.type_as(next(model.parameters()))).half().cpu().detach())
                
                else:
                    unaugmented_ds = CT_slice_Grayscale_NIFTIDataset_mask_BioMedClip(
                                                                                     im_array=im,
                                                                                     CoM=CoM,
                                                                                     msk_array=non_zero, 
                                                                                     view=view, 
                                                                                     transform=normal_transform, 
                                                                                     window=window, 
                                                                                     level=level,
                                                                                     tokenizer=toknzer,
                                                                                     crop_size=crop_size)
                    
                    augmented_ds = CT_slice_Grayscale_NIFTIDataset_mask_BioMedClip( 
                                                                                   im_array=im,
                                                                                   CoM=CoM,
                                                                                   msk_array=non_zero, 
                                                                                   view=view, 
                                                                                   transform=augmenting_transform, 
                                                                                   window=window, 
                                                                                   level=level,
                                                                                   tokenizer=toknzer,
                                                                                   crop_size=crop_size)
                    
                    ds= ConcatDataset([unaugmented_ds, augmented_ds])
                    dl = torch.utils.data.DataLoader(
                        ds, batch_size=1, shuffle=False, drop_last=False)
                    feats = []
                    for batch in tqdm(dl, leave=False):
                        im_f = batch[0].to(device)
                        txt_f = batch[1].to(device)
                        feat_im, feat_txt, log_sc = model(im_f, txt_f)
                        feat_im = feat_im.half().cpu().detach()
                        del feat_txt
                        del txt_f
                        del log_sc
                        del im_f
                        feats.append(feat_im)
                with h5py.File(h5outpath, 'w') as f:
                    try:        
                        f['location'] = [n_zero for n_zero in non_zero] + [n_zero for n_zero in non_zero]
                        #print('22222222222222')
                        f['feats'] = torch.concat(feats).cpu().numpy()
                        #print('========================')
                        #print(f['feats'].shape)
                        f['augmented'] = np.repeat(
                        [False, True], [len(unaugmented_ds), len(augmented_ds)])
                        f.attrs['extractor'] = model_name
                        print(f.attrs['extractor'])
                    except:
                        print('Error with file naming, no location given')

            del im
            del msk