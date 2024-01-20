import torch
from torch.utils.data import Dataset
import numpy as np
from skimage.transform import rescale

from skimage.morphology import remove_small_objects

from torch import nn

from utils import *


class PotTorch(Dataset):
    
    def __init__(self, selected_pots, archeo_info, img_size, min_mask_size = 50, max_mask_size = 110, transform=None):
        '''
        A dataset class for simulated fragments and other stuff.
        '''
        self.selected_pots = selected_pots
        self.transform = transform
        self.min_mask_size = min_mask_size
        self.max_mask_size = max_mask_size
        self.img_size = img_size
        if archeo_info is not None:
            self.archeo_info = archeo_info
        else:
            self.archeo_info = []

    def create_square_mask(self, tensor, mask_size):
        '''
        Create fragment.

        '''
        tensor = tensor.clone()
        num_channels, height, width = tensor.shape
        mask = torch.ones(num_channels, height, width)

        indices = tensor.nonzero()
        selected_index = indices[np.random.randint(0, len(indices), 1)]

        y_start = selected_index[:, 1]
        x_start = selected_index[:, 2]

        y_start = max(mask_size // 2, min(y_start, height - mask_size // 2))
        x_start = max(mask_size // 2, min(x_start, width - mask_size // 2))

        mask[:, y_start - mask_size // 2:y_start + mask_size // 2, x_start - mask_size // 2:x_start + mask_size // 2] = 0
        
        mask = mask.bool()

        tensor = tensor.masked_fill(mask, 0)



        tensor_numpy = tensor.numpy()

        tensor_numpy = tensor_numpy.squeeze()

        scale_factor = np.random.randint(40, 100)/100
        
        padded = pad_image_fixed_dim_padded(rescale(minimum_image(tensor_numpy), scale_factor, anti_aliasing=False), self.img_size)


        bounding_box = minimum_image_bb(tensor_numpy)

        bounding_box = torch.tensor(bounding_box)

        ### Remove small objects
        tensor_numpy = remove_small_objects(tensor_numpy, min_size=tensor_numpy.astype(np.uint8).sum()/2)

        tensor = torch.tensor(tensor_numpy).unsqueeze(0)
        ###


        
        padded_tensor = torch.tensor(padded).unsqueeze(0)

        return tensor, padded_tensor, bounding_box, scale_factor, mask #tensor = masked_img

    def gaussian_blur_grayscale(self, tensor, kernel_size, sigma):
        kernel = self.create_gaussian_kernel(kernel_size, sigma).unsqueeze(0).unsqueeze(0)
        
        blurred_tensor = nn.functional.conv2d(tensor, kernel, padding=kernel_size // 2)
        
        return blurred_tensor

    def create_gaussian_kernel(self, kernel_size, sigma):
        kernel = np.fromfunction(lambda x, y: (1 / (2 * np.pi * sigma**2)) * np.exp(-((x - kernel_size // 2)**2 + (y - kernel_size // 2)**2) / (2 * sigma**2)), (kernel_size, kernel_size))
        kernel = kernel / np.sum(kernel)
        kernel = torch.FloatTensor(kernel)
        
        return kernel


    def __len__(self):
        return len(self.selected_pots)

    def __getitem__(self, index):
        pot = self.selected_pots[index]

        if len(self.archeo_info) != 0:
            archeo_info = self.archeo_info[index]
            archeo_info = torch.tensor(archeo_info).float()
        else:
            archeo_info = []
        

        pot = self.transform(pot)

        
        mask_size = np.random.randint(self.min_mask_size, self.max_mask_size + 1)

        masked_img, padded_tensor, bounding_box, scale_factor, mask  = self.create_square_mask(pot, mask_size)

        kernel_size = np.random.randint(10, 25)
        sigma = np.random.randint(5, 10)

        blurred_tensor = self.gaussian_blur_grayscale(pot.float(), kernel_size=kernel_size, sigma=sigma)         

        blurred_tensor = blurred_tensor.unsqueeze(0)

        blurred_tensor =  nn.functional.interpolate(blurred_tensor, size=(128, 128), mode='bilinear', align_corners=False)


        blurred_tensor = blurred_tensor.squeeze(0)

        row_mask = torch.all(torch.eq(masked_img, 0), dim=2, keepdim=True)
        col_mask = torch.all(torch.eq(masked_img, 0), dim=1, keepdim=True)
        x_filled = torch.where(row_mask, blurred_tensor, masked_img)
        x_filled = torch.where(col_mask, blurred_tensor, x_filled)




        return pot, masked_img, padded_tensor, bounding_box, 1/scale_factor, archeo_info, mask, x_filled

    