import torch 
import torchvision.transforms.functional as TF

####################################################################
#Custom transforms
####################################################################   
class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):


        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)
class CenterMassCrop(object):
    def __init__(self, crop_size: int=224, padding_value:int=0):
        """
        Initializes the CenterMassCrop transform.
        
        Args:
        - crop_size (tuple or int): The size of the crop (height, width). If an int is provided, a square crop will be applied.
        """
        self.padding_value=padding_value
        if isinstance(crop_size, (int,float)):
            self.crop_size = (int(crop_size), int(crop_size))
        else:
            self.crop_size = crop_size
    
    def __call__(self, image, CoM):
        """
        Args:
        - image (PIL Image or Tensor): The image to be cropped.
        - mask (PIL Image or Tensor): The segmentation mask used to calculate the center of mass.
        
        Returns:
        - cropped_image (Tensor): The cropped image.
        """
        # Ensure image and mask are tensors
        if not isinstance(image, torch.Tensor):
            image = TF.to_tensor(image)
        
        # Convert mask to numpy for center of mass calculation
          # Assuming mask is single channel
        
        # Calculate the center of mass of the mask
        center_y, center_x = CoM
        
        # Convert to int since we need pixel indices
        center_y, center_x = int(center_y), int(center_x)
        
        # Calculate cropping coordinates
        crop_height, crop_width = self.crop_size
        half_crop_height = crop_height // 2
        half_crop_width = crop_width // 2
        
        top = max(0, center_y - half_crop_height)
        left = max(0, center_x - half_crop_width)
        
        bottom = min(image.shape[1], top + crop_height)
        right = min(image.shape[2], left + crop_width)
        
        # Crop the image and mask
        cropped_image = image[:, top:bottom, left:right]
        #add padding if necessary:
        pad_height = max(0, crop_height - cropped_image.shape[1])
        pad_width = max(0, crop_width - cropped_image.shape[2])
        
        if pad_height > 0 or pad_width > 0:
            # Pad the image and mask to the target crop size (224x224)
            padding = (0, 0, pad_width, pad_height)  # Padding for left, top, right, bottom
            cropped_image = TF.pad(cropped_image, padding, fill=self.padding_value)

        return cropped_image

class TriChanneler(object):
    '''
    Convert a tensor into a three-channel one for ImageNet-based usage 
    '''
    def __init__(self, n_channels:int=3):
        self.channels= n_channels
    def __call__(self, tensor):
        
        assert isinstance(tensor,torch.Tensor), "The input image needs to be type Tensor"
        tensor= tensor.squeeze() #eliminate leading dimensions from previous preprocessing, ToTensor does this 
        assert len(tensor.shape) in [2,3], "The iput should be a 2D or 3D tensor, check the input!"
        
        if len(tensor.shape)==2:
            tensor = tensor.unsqueeze(0).repeat(3, 1, 1)
        else:
            tensor = tensor.unsqueeze(0).repeat(3, 1, 1, 1)
        return tensor