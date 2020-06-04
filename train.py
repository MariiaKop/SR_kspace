import os
import argparse

import numpy as np
import pandas as pd

from tqdm import tqdm

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from sr_kspace import transforms as T
from sr_kspace.model import Generator, Discriminator
from sr_kspace.utils import SRKspaceData, gray2rgb, calculate_ssim, calculate_psnr, save_images
from sr_kspace.loss import GeneratorLoss


os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


def inference(netG, lr_kspace, h_LR, h_HR, skipped_connection):
    input = lr_kspace/h_LR
    input /= abs(input).max()
    out = netG(input) * h_HR
    if skipped_connection:
        out[..., 80:240, 80:240] = out[..., 80:240, 80:240] + lr_kspace

    out = T.k_space_to_image(out, shift=True)

    return out


def parse_args():
    parser = argparse.ArgumentParser(description='Train Super Resolution Models')

    parser.add_argument('--channels', default=64, type=int, help='Number of channels in Residual blocks')
    parser.add_argument('--skip_connection', default=True, type=bool, help='Skip connection')
    parser.add_argument('--bias', default=True, type=bool, help='Bias in Conv layers for Generator')
    parser.add_argument('--upscale_factor', default=2, type=int, choices=[2, 4],
                        help='Super resolution upscale factor')
    parser.add_argument('--epochs', default=10, type=int, help='Train epoch number')
    parser.add_argument('--path_to_data', default='data', type=str, help='Path to data')
    parser.add_argument('--lr', default=1e-3, type=float, help='Learning rate')
    parser.add_argument('--batch_size', default=64, type=int, help='Batch size for train loader')
    parser.add_argument('--random_state', default=None, type=int, help='Random state')


    return parser.parse_args()


def load_h(path_to_data):
    h_HR = np.load(os.path.join(path_to_data, 'hr_mean_kspace.npy'))
    h_LR = np.load(os.path.join(path_to_data, 'lr_mean_kspace.npy'))

    h_HR = torch.from_numpy(h_HR)
    h_LR = torch.from_numpy(h_LR)

    if torch.cuda.is_available():
        h_HR = h_HR.cuda()
        h_LR = h_LR.cuda()

    return h_HR, h_LR


def init_data_loaders(path_to_data, upscale_factor, batch_size):
    train_set = SRKspaceData(os.path.join(path_to_data, 'ax_t2_source_train'), 
                             os.path.join(path_to_data, f'ax_t2_re_im_{320//upscale_factor}_train'))
    val_set = SRKspaceData(os.path.join(path_to_data, 'ax_t2_source_val'), 
                             os.path.join(path_to_data, f'ax_t2_re_im_{320//upscale_factor}_val'))

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=4)

    return train_loader, val_loader


def init_nets(opt):
    netG = Generator(opt.upscale_factor, input_channels=2, output_channels=2, 
                     channels=opt.channels, bias=opt.bias)
    netD = Discriminator(input_channels=1)

    generator_criterion = GeneratorLoss()

    if torch.cuda.is_available():
        netG.cuda()
        netD.cuda()
        generator_criterion.cuda()

    return netG, netD, generator_criterion


def main():
    opt = parse_args()

    if opt.random_state:
        torch.random.manual_seed(opt.random_state)
        np.random.seed(opt.random_state)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    h_HR, h_LR = load_h(opt.path_to_data)
    train_loader, val_loader = init_data_loaders(opt.path_to_data, opt.upscale_factor, opt.batch_size)
    netG, netD, generator_criterion = init_nets(opt)

    optimizerG = optim.Adam(netG.parameters(), lr=opt.lr)
    optimizerD = optim.Adam(netD.parameters(), lr=opt.lr)

    bias_label = '_bias' if opt.bias else ''
    skip_label = '_skip' if opt.skip_connection else ''
    label = f'{opt.channels}_{opt.upscale_factor}{bias_label}{skip_label}'
    print('Model:', label)

    out_path = f'results/{label}/'
    os.makedirs(os.path.join(out_path, 'images'), exist_ok=True)
    os.makedirs(os.path.join(out_path, 'models'), exist_ok=True)

    with open(os.path.join(out_path, 'info'), 'w') as f:
        f.write(f'Device: {device}\n{opt}\n\n')
        f.write(f'{netG}\n\n')
        f.write(f'{netD}\n\n')

    train(opt, netG, netD, generator_criterion, optimizerG, optimizerD, 
          h_HR, h_LR, train_loader, val_loader, out_path, label
         )


def train(opt, netG, netD, generator_criterion, optimizerG, optimizerD, 
          h_HR, h_LR, train_loader, val_loader, out_path, label):
    results = {'d_loss': [], 'g_loss': [], 'd_score': [], 'g_score': [], 'psnr': [], 'ssim': [], 'mae': []}

    for epoch in range(1, opt.epochs + 1):
        train_bar = tqdm(train_loader)
        running_results = {'batch_sizes': 0, 'd_loss': 0, 'g_loss': 0, 'd_score': 0, 'g_score': 0}

        netG.train()
        netD.train()
        for hr_img, lr_kspace in train_bar:
            batch_size = hr_img.size(0)
            running_results['batch_sizes'] += batch_size

            # (1) Update D network: maximize D(x)-1-D(G(z))
            if torch.cuda.is_available():
                hr_img = hr_img.cuda()
                lr_kspace = lr_kspace.cuda()

            fake_img = inference(netG, lr_kspace, h_LR, h_HR, opt.skip_connection)

            netD.zero_grad()
            real_out = netD(hr_img).mean()
            fake_out = netD(fake_img).mean()
            d_loss = 1 - real_out + fake_out
            d_loss.backward(retain_graph=True)
            optimizerD.step()

            # (2) Update G network: minimize 1-D(G(z)) + Perception Loss + Image Loss + TV Loss
            netG.zero_grad()
            g_loss = generator_criterion(fake_out, gray2rgb(fake_img), gray2rgb(hr_img))
            g_loss.backward()

            fake_img = inference(netG, lr_kspace, h_LR, h_HR, opt.skip_connection)
            fake_out = netD(fake_img).mean()

            optimizerG.step()

            # loss for current batch before optimization 
            running_results['g_loss'] += g_loss.item() * batch_size
            running_results['d_loss'] += d_loss.item() * batch_size
            running_results['d_score'] += real_out.item() * batch_size
            running_results['g_score'] += fake_out.item() * batch_size

            train_bar.set_description(desc='[%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f' % (
                epoch, opt.epochs, running_results['d_loss'] / running_results['batch_sizes'],
                running_results['g_loss'] / running_results['batch_sizes'],
                running_results['d_score'] / running_results['batch_sizes'],
                running_results['g_score'] / running_results['batch_sizes']))

            break


        netG.eval()
        val_bar = tqdm(val_loader)
        valing_results = {'mae': 0, 'ssims': 0, 'psnr': 0, 'ssim': 0, 'batch_sizes': 0}

        for i, (hr_img, lr_kspace) in enumerate(val_bar):
            batch_size = hr_img.size(0)
            valing_results['batch_sizes'] += batch_size
            if torch.cuda.is_available():
                hr_img = hr_img.cuda()
                lr_kspace = lr_kspace.cuda()

            sr_img = inference(netG, lr_kspace, h_LR, h_HR, opt.skip_connection)

            batch_mae = abs(sr_img - hr_img).data.mean().item()
            valing_results['mae'] += batch_mae * batch_size
            batch_ssim = calculate_ssim(sr_img, hr_img)
            valing_results['ssims'] += batch_ssim * batch_size
            valing_results['psnr'] = calculate_psnr(hr_img, sr_img, valing_results['batch_sizes'])
            valing_results['ssim'] = valing_results['ssims'] / valing_results['batch_sizes']
            val_bar.set_description(
                desc='[converting LR images to SR images] PSNR: %.4f dB SSIM: %.4f' % (
                    valing_results['psnr'], valing_results['ssim']))

            break

        save_images(hr_img, sr_img, abs(sr_img - hr_img), 
                    path=os.path.join(out_path, 'images', f'val_{label}_{epoch}.png'))
        

        torch.save(netG.state_dict(), 
                os.path.join(out_path, 'models', f'netG_epoch_{label}_{epoch}.pth'))
        torch.save(netD.state_dict(), 
                os.path.join(out_path, 'models', f'netD_epoch_{label}_{epoch}.pth'))


        results['d_loss'].append(running_results['d_loss'] / running_results['batch_sizes'])
        results['g_loss'].append(running_results['g_loss'] / running_results['batch_sizes'])
        results['d_score'].append(running_results['d_score'] / running_results['batch_sizes'])
        results['g_score'].append(running_results['g_score'] / running_results['batch_sizes'])
        results['psnr'].append(valing_results['psnr'])
        results['ssim'].append(valing_results['ssim'])
        results['mae'].append(valing_results['mae'])

        if epoch % 1 == 0:
            data_frame = pd.DataFrame(
                data={'Loss_D': results['d_loss'], 'Loss_G': results['g_loss'], 'Score_D': results['d_score'],
                    'Score_G': results['g_score'], 'PSNR': results['psnr'], 
                    'SSIM': results['ssim'], 'MAE': results['mae']},
                index=range(1, epoch + 1))

            data_frame.to_csv(os.path.join(out_path, f'metrics_val_{label}.csv'), 
                            index_label='Epoch')


if __name__ == '__main__':
    main()
