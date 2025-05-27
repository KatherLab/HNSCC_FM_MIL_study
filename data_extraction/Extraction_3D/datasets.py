import numpy as np
from torch.utils.data import Dataset
import math
from torchvision.transforms import transforms
class CubeFoundation_NiftiDataset(Dataset):
    def __init__(self, img: np.array,model_name: str , mask: np.array, 
                 cube_size: int =50,
                 repetitions: int=50, 
                 sample_variance: int=16,
                 min: int = -1024,
                 max: int = 2048,
                 transform= None) -> None:
        """
        Dataset to extract cubes from a 3D image which are preprocessed to model specifications and centered on the CoM of a lesion
        """
        self.model_name = model_name
        self.cube_size= cube_size
        assert cube_size>0, "Cube size for sampling has to be an int bigger than 0"
        self.img = img
        self.mask = mask
        self.repetitions=repetitions
        self.CoM = np.mean(np.argwhere(mask==1), axis=0) #get centre of mass of the image mask
        self.sample_variance = sample_variance
        self.sample_centres = [np.random.normal(0,np.sqrt(self.sample_variance), repetitions) + self.CoM[a] for a in np.arange(3)]
        self.sample_centres = list(zip(*self.sample_centres))
        self.sample_centres.append(tuple(self.CoM)) #This creates a list of (nreps) with each being the centre of the new cube to sample for the model
        self.out_shape = [self.cube_size] *3 #create desired output shape
        self.transform= transform
        self.max= max #parameters for minmax normalisation for the foundation model
        self.min = min
    def __len__(self):
        return(self.repetitions)
    
    def __getitem__(self, index):
        idx_x = [math.floor(self.sample_centres[index][0]) - self.cube_size//2, math.floor(self.sample_centres[index][0]) + self.cube_size//2]
        idx_y = [math.floor(self.sample_centres[index][1]) - self.cube_size//2, math.floor(self.sample_centres[index][1]) + self.cube_size//2]
        idx_z = [math.floor(self.sample_centres[index][2]) - self.cube_size//2, math.floor(self.sample_centres[index][2]) + self.cube_size//2]
        ref_x, ref_y, ref_z = self.img.shape 
        #Adjust in case we have out-of.bound indices#########
        if idx_x[0]<0:
            idx_x[0] =0
        if idx_y[0]<0:
            idx_y[0] =0
        if idx_z[0]<0:
            idx_z[0] =0

        if idx_x[1]>ref_x:
            idx_x[1] =ref_x
        if idx_y[1]>ref_y:
            idx_y[1] =ref_y
        if idx_z[1]>ref_z:
            idx_z[1] =ref_z
        ###############################################################

        vol = self.img[idx_x[0]:idx_x[1],
                       idx_y[0]:idx_y[1],
                       idx_z[0]:idx_z[1]]
        if list(vol.shape) != self.out_shape:
            vol = np.pad(vol,((self.cube_size -vol.shape[0],self.cube_size-vol.shape[0]),
                              (self.cube_size -vol.shape[1],self.cube_size-vol.shape[1]),
                              (self.cube_size -vol.shape[2],self.cube_size-vol.shape[2])),mode="constant", 
                              constant_values=-2048)
        
        #After we have sliced the cube, we can clip and transform it :)#
        if self.model_name=="foundation":
            np.clip(vol,self.min,self.max,vol) #clip the HU values to the minimum and maximum of the image for the foundation model
            vol = (vol - self.min) / (self.max -self.min)
        #For the ResNet50 baseline we need also to transform it into grayscale values and 
        if self.model_name=="resnet-50":
            np.clip(vol,self.min,self.max,vol)
            vol = ((vol - self.min) / (self.max -self.min) * 255).astype(np.uint8) #The imagenet resnets require grayscale values!
            #as the elements are already unsigned int 8 of the array, the ToTensor transform scales the intensities to 0-1 range :)
        if self.transform:
            vol= self.transform(vol)
        else: 
            tr = transforms.ToTensor()
            vol = tr(vol) #at least transform the thing to a tensor :)

        #Remove internal variables that are already used for memory freeing######
        del idx_x
        del idx_y
        del idx_z
        del ref_x
        del ref_y
        del ref_z
        #######################################
        if self.model_name =="foundation":
            return vol.unsqueeze(0) #Add the channel dimension  for the foundation model :)
        if self.model_name=="resnet-50":
            return vol #The channel dimension is added as the 3dgrayscale transform!   
