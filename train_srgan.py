import os
import argparse

import numpy as np
import pandas as pd
from skimage.transform import resize

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, RandomSampler
from torchvision.models.vgg import vgg16

from sr_kspace.model import Generator
from sr_kspace.utils import gray2rgb, calculate_ssim, calculate_psnr, save_images, calculate_mae


os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
PATH_TO_X4_MODEL = 'models/netG_epoch_4_100.pth'


class SRGANMRIDataLoader(Dataset):
    def __init__(self, path_to_data, scale_factor, seed=None):
        super().__init__()

        self.data = [os.path.join(path_to_data, file) for file in os.listdir(path_to_data) 
                       if file.endswith('.npy')]
        self.scale_factor = scale_factor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        
        slice = np.load(self.data[idx])
        h, w = slice.shape[-2:]
        lr_slice = resize(slice[0], (h//self.scale_factor, w//self.scale_factor), preserve_range=True)
        slice = torch.from_numpy(slice).type(torch.float32)
        lr_slice = torch.from_numpy(lr_slice).unsqueeze(0).type(torch.float32)

        return slice, lr_slice
    
    
class VGGLoss(nn.Module):
    def __init__(self, as_gray=False):
        super().__init__()
        vgg = vgg16(pretrained=True)
        loss_network = nn.Sequential(*list(vgg.features)[:31]).eval()
        for param in loss_network.parameters():
            param.requires_grad = False
        self.loss_network = loss_network
        self.as_gray = as_gray
        self.mse_loss = nn.MSELoss()

    def forward(self, img1, img2):
        if self.as_gray:
            img1 = torch.cat([img1, img1, img1], dim=-3)
            img2 = torch.cat([img2, img2, img2], dim=-3)
            
        perception_loss = self.mse_loss(self.loss_network(img1), self.loss_network(img2))
        return perception_loss


def parse_args():
    parser = argparse.ArgumentParser(description='Train Super Resolution Models')

    parser.add_argument('--upscale_factor', default=2, type=int, choices=[2, 4],
                        help='Super resolution upscale factor')
    parser.add_argument('--epochs', default=10, type=int, help='Train epoch number')
    parser.add_argument('--path_to_data', default='data', type=str, help='Path to data')
    parser.add_argument('--lr', default=1e-3, type=float, help='Learning rate')
    parser.add_argument('--batch_size', default=64, type=int, help='Batch size for train loader')
    parser.add_argument('--random_state', default=None, type=int, help='Random state')
    parser.add_argument('--random_subset', default=None, type=int, help='Size of subset for each epoch')
    parser.add_argument('--val_size', default=None, type=int, help='Size of val set')

    return parser.parse_args()


def init_data_loaders(opt):
    path_to_data = opt.path_to_data

    train_set = SRGANMRIDataLoader(os.path.join(path_to_data, 'ax_t2_source_train'), opt.upscale_factor)
    val_set = SRGANMRIDataLoader(os.path.join(path_to_data, 'ax_t2_source_val'), opt.upscale_factor)
    if opt.val_size:
        val_set.data = val_set.data[:opt.val_size]

    if opt.random_subset:
        sampler = RandomSampler(train_set, replacement=True, num_samples=opt.random_subset)
        train_loader = DataLoader(train_set, sampler=sampler, batch_size=opt.batch_size,
                                             shuffle=False, num_workers=4)
    else:
        train_loader = DataLoader(train_set, batch_size=opt.batch_size, shuffle=True, num_workers=4)

    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=4)

    return train_loader, val_loader


def copy_res_layers(netG):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    pretrained_G = Generator(4)
    x4_params = torch.load(PATH_TO_X4_MODEL, map_location=device)
    pretrained_G.load_state_dict(x4_params)

    netG.block2.load_state_dict(pretrained_G.block2.state_dict())
    netG.block3.load_state_dict(pretrained_G.block3.state_dict())
    netG.block4.load_state_dict(pretrained_G.block4.state_dict())
    netG.block5.load_state_dict(pretrained_G.block5.state_dict())
    netG.block6.load_state_dict(pretrained_G.block6.state_dict())

    static_layers = [f'block{i}' for i in range(2, 7)]
    for k, w in netG.named_parameters():
        if any(k.startswith(l) for l in static_layers):
            w.requires_grad = False


def main():
    opt = parse_args()

    if opt.random_state:
        torch.random.manual_seed(opt.random_state)
        np.random.seed(opt.random_state)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    print(opt, end='\n\n')

    netG = Generator(opt.upscale_factor, input_channels=1, output_channels=1)
    copy_res_layers(netG)
    
    print('Number of params:', sum(p.numel() for p in netG.parameters() if p.requires_grad))

    criterion = nn.MSELoss()
    vgg_criterion = VGGLoss(as_gray=True)

    optimizer = optim.Adam(netG.parameters(), lr=opt.lr)

    train_loader, val_loader = init_data_loaders(opt)
    if torch.cuda.is_available():
        netG.cuda()
        vgg_criterion.cuda()

    label = f'srgan_{opt.upscale_factor}'
    print('Model:', label)

    out_path = f'results/{label}/'
    os.makedirs(os.path.join(out_path, 'images'), exist_ok=True)
    os.makedirs(os.path.join(out_path, 'models'), exist_ok=True)

    with open(os.path.join(out_path, 'info'), 'w') as f:
        f.write(f'Device: {device}\n{opt}\n\n')
        f.write(f'{netG}\n\n')

    train(opt, netG, vgg_criterion, criterion, optimizer,
          train_loader, val_loader, out_path, label
         )


def train(opt, netG, vgg_criterion, criterion, optimizer,
          train_loader, val_loader, out_path, label):
    results = {'vgg_loss': [], 'psnr': [], 'ssim': [], 'mae': [], 'mse': []}

    label = f'srgan_{opt.upscale_factor}'

    out_path = f'results/{label}/'
    os.makedirs(os.path.join(out_path, 'images'), exist_ok=True)
    os.makedirs(os.path.join(out_path, 'models'), exist_ok=True)


    for epoch in range(1, opt.epochs + 1):
        train_bar = tqdm(train_loader)
        running_results = {'batch_sizes': 0, 'mse': 0, 'vgg_loss': 0}

        netG.train()
        for hr, lr in train_bar:
            batch_size = hr.size(0)
            running_results['batch_sizes'] += batch_size

            if torch.cuda.is_available():
                hr = hr.cuda()
                lr = lr.cuda()

            sr = netG(lr)
            loss = criterion(hr, sr)
            vgg_loss = vgg_criterion(hr, sr)

            optimizer.zero_grad()
            vgg_loss.backward(retain_graph=True)
            loss.backward()
            optimizer.step()

            running_results['vgg_loss'] += vgg_loss.item() * batch_size
            running_results['mse'] += loss.item() * batch_size

            train_bar.set_description(desc='[%d/%d] MSE: %.4f VGG_loss: %.4f' % (
                epoch, opt.epochs, running_results['mse'] / running_results['batch_sizes'],
                running_results['vgg_loss'] / running_results['batch_sizes']))

            #break


        netG.eval()
        val_bar = tqdm(val_loader)
        valing_results = {'mae': 0, 'ssims': 0, 'psnr': 0, 'ssim': 0, 'batch_sizes': 0}

        for hr, lr in val_bar:
            batch_size = hr.size(0)
            valing_results['batch_sizes'] += batch_size
            if torch.cuda.is_available():
                hr = hr.cuda()
                lr = lr.cuda()

            sr = netG(lr)

            valing_results['mae'] += calculate_mae(hr, sr) * batch_size
            valing_results['ssim'] += calculate_ssim(hr, sr) * batch_size
            valing_results['psnr'] += calculate_psnr(hr, sr) * batch_size
            val_bar.set_description(
                desc='[converting LR images to SR images] PSNR: %.4f dB SSIM: %.4f' % (
                    valing_results['psnr']/valing_results['batch_sizes'], 
                    valing_results['ssim']/valing_results['batch_sizes'],
                    valing_results['mae']/valing_results['batch_sizes'],
                    ))

            #break

        save_images(hr, sr, abs(sr - hr), 
                    path=os.path.join(out_path, 'images', f'val_{label}_{epoch}.png'))
        

        torch.save(netG.state_dict(), 
                os.path.join(out_path, 'models', f'netG_epoch_{label}_{epoch}.pth'))

        results['mse'].append(running_results['mse'] / running_results['batch_sizes'])
        results['vgg_loss'].append(running_results['vgg_loss'] / running_results['batch_sizes'])
        results['psnr'].append(valing_results['psnr'] / valing_results['batch_sizes'])
        results['ssim'].append(valing_results['ssim'] / valing_results['batch_sizes'])
        results['mae'].append(valing_results['mae'] / valing_results['batch_sizes'])

        if epoch % 1 == 0:
            data_frame = pd.DataFrame(
                data={'MSE': results['mse'], 'vgg_loss': results['vgg_loss'],
                    'PSNR': results['psnr'], 'SSIM': results['ssim'], 'MAE': results['mae']},
                index=range(1, epoch + 1))

            data_frame.to_csv(os.path.join(out_path, f'metrics_val_{label}.csv'), 
                            index_label='Epoch')


if __name__ == '__main__':
    main()
