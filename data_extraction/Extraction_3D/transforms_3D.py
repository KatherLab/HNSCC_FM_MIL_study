import torch
import torchvision.transforms.functional as TF
import random

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):


        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)
    
class Grayscale_Norm3d(object):
    def __init__(self, mean, std):
        """
        mean: list or tuple of 3 elements specifying the mean for each channel
        std: list or tuple of 3 elements specifying the standard deviation for each channel
        """
        super().__init__()
        assert len(mean) == 3 and len(std) == 3, "Mean and std must be lists/tuples of length 3."
        self.mean = torch.tensor(mean).view(3, 1, 1, 1)  # Reshape to match [C, 1, 1, 1] for broadcasting
        self.std = torch.tensor(std).view(3, 1, 1, 1)    # Reshape to match [C, 1, 1, 1] for broadcasting

    def __call__(self, image):
        """
        image: 3D tensor of size [H, W, L]
        Returns a 4D tensor [3, H, W, L] where each channel is normalized.
        """
        # Ensure the input image is 3D
        assert image.ndim == 3, "Input image must be a 3D tensor [H, W, L]"

        # Replicate the 3D image into 3 channels
        image_replicated = image.unsqueeze(0).repeat(3, 1, 1, 1)  # [3, H, W, L]

        # Normalize each channel with the specified mean and std
        normalized_image = (image_replicated - self.mean) / self.std
        del image_replicated
        return normalized_image


class RotateAroundZ(object):
    def __init__(self, degrees=(-10, 10)):
        """
        Initialize the rotation with a range of degrees.
        degrees: tuple or int, range of degrees to choose from for random rotation
        """
        if isinstance(degrees, (int, float)):
            self.degrees = (-degrees, degrees)
        else:
            self.degrees = degrees

    def __call__(self, image):
        """

        Apply random 2D rotation on each [y, x] slice along the z-axis.
        We assume LAS-like orientation for the affine so the axial axis is the last one!
        image: 3D tensor [x,y,z]
        """
        # Ensure the image is a 3D tensor
        assert image.ndim == 3, "Input image must be a 3D tensor [z, y, x]"

        # Generate a random angle within the range
        angle = random.uniform(self.degrees[0], self.degrees[1])
        #The angle is the same for every slice of the image the transform is used on, bit changes with each call :)
        # Rotate each slice in the [y, x] plane for each z coordinate
        rotated_image = torch.zeros_like(image)
        for z in range(image.shape[-1]): 
            slice_2d = image[:, :, z]
            rotated_slice = TF.rotate(slice_2d.unsqueeze(0), angle, interpolation=TF.InterpolationMode.BILINEAR)
            rotated_image[:, :, z] = rotated_slice.squeeze(0)
        return rotated_image
    
