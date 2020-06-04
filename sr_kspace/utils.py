import os
from math import log10

import torch
import numpy as np
from skimage.measure import compare_ssim
from torch.utils.data import Dataset
from torchvision.utils import save_image


class SRKspaceData(Dataset):
    def __init__(self, path_to_hr, path_to_lr_kspace):
        super().__init__()

        self.path_to_hr = path_to_hr
        self.path_to_lr_kspace = path_to_lr_kspace

        hr_images = set(os.listdir(path_to_hr))
        lr_kspace = set(os.listdir(path_to_lr_kspace))
        intersected_data = hr_images.intersection(lr_kspace)
        data = sorted([s for s in intersected_data if s.endswith('npy')])

        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        hr_image = np.load(os.path.join(self.path_to_hr, self.data[idx]))
        lr_kspace = np.load(os.path.join(self.path_to_lr_kspace, self.data[idx]))

        hr_image = torch.from_numpy(hr_image)
        lr_kspace = torch.from_numpy(lr_kspace)

        return hr_image, lr_kspace


def gray2rgb(gray):
    rgb = torch.cat([gray, gray, gray], dim=-3)
    return rgb


def calculate_ssim(img1, img2):
    img1, img2 = img1[0, 0], img2[0, 0]
    return compare_ssim(img1.detach().cpu().numpy(), img2.detach().cpu().numpy())


def calculate_psnr(hr_img, sr_img, batch_size):
    return 10 * log10((hr_img.max()**2) / (((sr_img - hr_img)** 2).data.mean() / batch_size))


def save_images(*images, path):
    image = torch.cat(images, dim=-1)[0, 0].detach().cpu()
    save_image(image, path)