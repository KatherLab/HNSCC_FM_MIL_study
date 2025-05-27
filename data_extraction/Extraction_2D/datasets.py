from torch.utils.data import Dataset
import numpy as np
from PIL import Image
from torchvision import transforms
from transforms_2D import CenterMassCrop
import torch 

class CT_slice_Grayscale_NIFTIDataset_mask(Dataset):
    def __init__(self, 
                 im_array: np.array,
                 CoM:list[tuple], 
                 msk_array: np.array, 
                 view: str, 
                 transform=None,
                 level: int = 50,
                 window: int=120,
                 crop_size: int=224) -> None:
        """
        Dataset for CT slice extraction based on CoMs and locations of a mask per view. Input image is assumed to be LAS-oriented [sagittal,coronal,axial]
        Args:
        im_array: np.array, array containing the 3D image itself, which is then sliced based on the view and mask indices
        CoM: tupple with the centre of mass, these should match the view we want to obtain e.g in axial the indices represent sagittal and coronal location of the CoM of a mask
        msk_array: np.array, contains a list of indices that are extracted from the image for a specific view
        view: str, 2D-projection we will extract and preprocess images with; can be one of axial, sagittal or coronal
        transform: Object, series of torhcvision transforms applied to the image after obtan9ng the 2D projection
        repetitions: if augmentations are performed, number of times we iterate over the dataset to use more transforms
        level: int, middle point of the interval for HU extraction
        window: int; width of HU interval for clipping of intensities in CT
        crop_size: int, size of crop performed around centre of the tumor,afterwards it is resized to 224x224 for input into the CNNs/ViTs
        
        Output:
        Tensor, preprocessed to dims (3,224,224)
        """
        assert view in ["coronal","sagittal","axial"], "The image view needs to be one of coronal, sagittal, axial"
        
        self.img = im_array
        self.crop_size= crop_size
        self.transform = transform
        self.clip_HU = [abs(level) - (abs(window)//2),level + (abs(window)//2)] #get clipping range right
        self.view= view
        self.two_dim_mask_indices = msk_array
        self.CoM = CoM
    
    def __len__(self):
        return len(self.two_dim_mask_indices)

    def __getitem__(self, i):
        CoM_im= self.CoM[i]
        #Returns pixel array of the stored dicom data
        
        if self.view =="axial":
            image = self.img[:,:,self.two_dim_mask_indices[i]]
        if self.view =="coronal":
            image = self.img[:,self.two_dim_mask_indices[i],:]    
        if self.view =="sagittal":
            image = self.img[self.two_dim_mask_indices[i],:,:]
        #clip image data to specific HU range:
        np.clip(image,self.clip_HU[0],self.clip_HU[-1],image) #clip image in-place (faster)
        if np.min(image) == self.clip_HU[-1]:
            #If some images are having all bright or all dark spots, we can just set them to 1 or 0
            image[:] = 1.
            image.astype(np.uint8)
        if np.max(image) == self.clip_HU[0]:
            image[:] = 0.
            image.astype(np.uint8)
        else:
            image = ((image - np.min(image)) / (np.max(image) - np.min(image)) * 255).astype(np.uint8) #Transform to grayscale values

        image= Image.fromarray(image, mode="L") #Transform to grayscale type
        image=transforms.ToTensor()(image) #reduces intensity range between 0 and 1
        image=CenterMassCrop(self.crop_size)(image, CoM_im) #crop image to CoM of the mask in the CT slice
        if self.transform:
            image = self.transform(image)
        return image

class CT_slice_Grayscale_NIFTIDataset_mask_BioMedClip(Dataset):
    def __init__(self, 
                 im_array: np.array,
                 CoM:list[tuple], 
                 msk_array: np.array, 
                 view: str, 
                 transform=None,
                 level: int = 50,
                 window: int=120,
                 crop_size: int=224,
                 tokenizer= None) -> None:
        """
        Dataset for CT slice extraction based on CoMs and locations of a mask per view. Input image is assumed to be LAS-oriented [sagittal,coronal,axial]
        
        Args:
        im_array: np.array, array containing the 3D image itself, which is then sliced based on the view and mask indices
        CoM: tupple with the centre of mass, these should match the view we want to obtain e.g in axial the indices represent sagittal and coronal location of the CoM of a mask
        msk_array: np.array, contains a list of indices that are extracted from the image for a specific view
        view: str, 2D-projection we will extract and preprocess images with; can be one of axial, sagittal or coronal
        transform: Object, series of torhcvision transforms applied to the image after obtan9ng the 2D projection
        repetitions: if augmentations are performed, number of times we iterate over the dataset to use more transforms
        level: int, middle point of the interval for HU extraction
        window: int; width of HU interval for clipping of intensities in CT
        crop_size: int, size of crop performed around centre of the tumor,afterwards it is resized to 224x224 for input into the CNNs/ViTs
        
        Output:
        Tensor, preprocessed to dims (3,224,224)
        """
        assert view in ["coronal","sagittal","axial"], "The image view needs to be one of coronal, sagittal, axial"
        
        self.img = im_array
        self.crop_size= crop_size
        self.transform = transform
        self.clip_HU = [abs(level) - (abs(window)//2),level + (abs(window)//2)] #get clipping range right
        self.view= view
        self.two_dim_mask_indices = msk_array
        self.CoM = CoM
        self.tokenizer = tokenizer
        self.caption="UNHhhh"
    def __len__(self):
        return len(self.two_dim_mask_indices)

    def __getitem__(self, i):
        CoM_im= self.CoM[i]
        #Returns pixel array of the stored dicom data
        
        if self.view =="axial":
            image = self.img[:,:,self.two_dim_mask_indices[i]]
        if self.view =="coronal":
            image = self.img[:,self.two_dim_mask_indices[i],:]    
        if self.view =="sagittal":
            image = self.img[self.two_dim_mask_indices[i],:,:]
        #clip image data to specific HU range:
        np.clip(image,self.clip_HU[0],self.clip_HU[-1],image) #clip image in-place (faster)
        if np.min(image) == self.clip_HU[-1]:
            #If some images are having all bright or all dark spots, we can just set them to 1 or 0
            image[:] = 1.
            image.astype(np.uint8)
        if np.max(image) == self.clip_HU[0]:
            image[:] = 0.
            image.astype(np.uint8)
        else:
            image = ((image - np.min(image)) / (np.max(image) - np.min(image)) * 255).astype(np.uint8) #Transform to grayscale values

        image= Image.fromarray(image, mode="L") #Transform to grayscale type
        image=transforms.ToTensor()(image) #reduces intensity range between 0 and 1
        image=CenterMassCrop(self.crop_size)(image, CoM_im) #crop image to CoM of the mask in the CT slice

        if self.transform:
            image = self.transform(image)
        return image, torch.squeeze(self.tokenizer(self.caption))
       