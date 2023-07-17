
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

import pyexr

import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage

import random
from random import randint

import glob
import os
import time

# some constants

patch_size = 64  # patches are 64x64
n_patches = 400
eps = 0.00316


def show_data(data, figsize=(15, 15), normalize=False):
    if normalize:
        data = np.clip(data, 0, 1)**0.45454545
    plt.figure(figsize=figsize)
    imgplot = plt.imshow(data, aspect='equal')
    imgplot.axes.get_xaxis().set_visible(False)
    imgplot.axes.get_yaxis().set_visible(False)
    plt.show()


def remove_channels(data, channels):
    for c in channels:
        if c in data:
            del data[c]
        else:
            print("Channel {} not found in data!".format(c))


def build_data(img):
    data = img.get()


def preprocess_diffuse(diffuse, albedo):
    return diffuse / (albedo + eps)


def postprocess_diffuse(diffuse, albedo):
    return diffuse * (albedo + eps)


def preprocess_specular(specular):
    assert (np.sum(specular < 0) == 0)
    return np.log(specular + 1)


def postprocess_specular(specular):
    return np.exp(specular - 1e-6) - 1


def preprocess_diff_var(variance, albedo):
    return variance / (albedo + eps)**2


def preprocess_spec_var(variance, specular):
    return variance / (specular+1e-5)**2


def gradients(data):
    h, w, c = data.shape
    dX = data[:, 1:, :] - data[:, :w - 1, :]
    dY = data[1:, :, :] - data[:h - 1, :, :]
    # padding with zeros
    dX = np.concatenate((np.zeros([h, 1, c], dtype=np.float32), dX), axis=1)
    dY = np.concatenate((np.zeros([1, w, c], dtype=np.float32), dY), axis=0)

    return np.concatenate((dX, dY), axis=2)

# crops all channels


def crop(data, pos, patch_size):
    half_patch = patch_size // 2
    sx, sy = half_patch, half_patch
    px, py = pos
    return {key: val[(py-sy):(py+sy+1), (px-sx):(px+sx+1), :]
            for key, val in data.items()}


def crop_like(data, like, debug=False):
    if data.shape[-2:] != like.shape[-2:]:
        # crop
        with torch.no_grad():
            dx, dy = data.shape[-2] - \
                like.shape[-2], data.shape[-1] - like.shape[-1]
            data = data[:, :, dx//2:-dx//2, dy//2:-dy//2]
            if debug:
                print(dx, dy)
                print("After crop:", data.shape)
    return data


def apply_kernel(weights, data):
    recon_kernel_size = 21
    weights = weights.permute((0, 2, 3, 1)).to(device)
    _, _, h, w = data.size()
    weights = F.softmax(weights, dim=3).view(-1, w * h,
                                             recon_kernel_size, recon_kernel_size)

    r = recon_kernel_size // 2
    data = F.pad(data[:, :3, :, :], (r,) * 4, "reflect")
    R = []
    G = []
    B = []
    kernels = []
    for i in range(h):
        for j in range(w):
            pos = i*h+j
            ws = weights[:, pos:pos+1, :, :]
            kernels += [ws, ws, ws]
            sy, ey = i+r-r, i+r+r+1
            sx, ex = j+r-r, j+r+r+1
            R.append(data[:, 0:1, sy:ey, sx:ex])
            G.append(data[:, 1:2, sy:ey, sx:ex])
            B.append(data[:, 2:3, sy:ey, sx:ex])

    reds = (torch.cat(R, dim=1).to(device)*weights).sum(2).sum(2)
    greens = (torch.cat(G, dim=1).to(device)*weights).sum(2).sum(2)
    blues = (torch.cat(B, dim=1).to(device)*weights).sum(2).sum(2)

    res = torch.cat((reds, greens, blues), dim=1).view(-1, 3, h, w).to(device)

    return res


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

permutation = [0, 3, 1, 2]


def to_torch_tensors(data):
    if isinstance(data, dict):
        for k, v in data.items():
            if not isinstance(v, torch.Tensor):
                data[k] = torch.from_numpy(v)
    elif isinstance(data, list):
        for i, v in enumerate(data):
            if not isinstance(v, torch.Tensor):
                data[i] = to_torch_tensors(v)

    return data


def send_to_device(data):
    if isinstance(data, dict):
        for k, v in data.items():
            if isinstance(v, torch.Tensor):
                data[k] = v.to(device)
    elif isinstance(data, list):
        for i, v in enumerate(data):
            if isinstance(v, torch.Tensor):
                data[i] = v.to(device)

    return data


def unsqueeze_all(d):
    for k, v in d.items():
        d[k] = torch.unsqueeze(v, dim=0)
    return d


def denoise(diffuseNet, specularNet, data, debug=False):
    with torch.no_grad():
        criterion = nn.L1Loss()
        dataloader = torch.utils.data.DataLoader(data,  batch_size=4,
                                                 shuffle=True, num_workers=4)
        for _, sample in enumerate(dataloader):

            permutation = [0, 3, 1, 2]
            X_diff = sample['X_diff'].permute(permutation).to(device)
            Y_diff = sample['diffuse_GT'].permute(permutation).to(device)

            outputDiff = diffuseNet(X_diff)
            X_input = crop_like(X_diff, outputDiff)
            outputDiff = apply_kernel(outputDiff, X_input)
            Y_diff = crop_like(Y_diff, outputDiff)

            lossDiff = criterion(outputDiff, Y_diff).item()
            X_spec = sample['X_spec'].permute(permutation).to(device)
            Y_spec = sample['specular_GT'].permute(permutation).to(device)

            outputSpec = specularNet(X_spec)
            X_input = crop_like(X_spec, outputSpec)
            outputSpec = apply_kernel(outputSpec, X_input)
            Y_spec = crop_like(Y_spec, outputSpec)

            lossSpec = criterion(outputSpec, Y_spec).item()

            albedo = sample['albedo'].permute(permutation).to(device)
            outputFinal = outputDiff * \
                (albedo + eps) + torch.exp(outputSpec) - 1.0

            if True:
                print("Sample, denoised, gt")
                sz = 15
                orig = sample['finalInput'].permute(
                    permutation)
                orig = orig.cpu().permute([0, 2, 3, 1]).numpy()[0, :]
                show_data(orig, figsize=(sz, sz), normalize=True)
                img = outputFinal.cpu().permute([0, 2, 3, 1]).numpy()[0, :]
                show_data(img, figsize=(sz, sz), normalize=True)
                gt = sample['finalGt'].permute(
                    permutation)
                gt = gt.cpu().permute([0, 2, 3, 1]).numpy()[0, :]
                show_data(gt, figsize=(sz, sz), normalize=True)

            Y_final = sample['finalGt'].permute(permutation).to(device)

            lossFinal = criterion(outputFinal, Y_final).item()

            if debug:
                print("LossDiff:", lossDiff)
                print("LossSpec:", lossSpec)
                print("LossFinal:", lossFinal)


class KPCNDataset(torch.utils.data.Dataset):
    def __init__(self, samples):
        self.inputs = samples

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx]


def net(conv_L, input_C, mode):

    layers = [
        nn.Conv2d(input_C, 100,  kernel_size=5, padding=2, stride=1),
        nn.ReLU()
    ]
    for _ in range(conv_L-2):
        layers += [
            nn.Conv2d(100, 100, kernel_size=5, padding=2, stride=1),
            nn.ReLU()
        ]
    out_channels = 3 if mode == 'DPCN' else 21**2
    layers += [nn.Conv2d(100, out_channels,  kernel_size=5,
                         padding=2, stride=1)]

    for layer in layers:
        if isinstance(layer, nn.Conv2d):
            nn.init.xavier_uniform_(layer.weight)

    return nn.Sequential(*layers)


if __name__ == '__main__':
    kdiffuseNet = net(9, 29, 'KPCN').to(device)
    kdiffuseNet.load_state_dict(torch.load(
        './model/diffuse/KPCN_diff_6.pth'))
    kdiffuseNet.eval()

    kspecularNet = net(9, 29, 'KPCN').to(device)
    kspecularNet.load_state_dict(torch.load(
        './model/specular/KPCN_spec_6.pth'))
    kspecularNet.eval()

    data_torch = torch.load(
        '../data/test_data/test_torch/97483481-00128spp.pt')
    input_list = to_torch_tensors([data_torch])
    input_list = send_to_device(input_list)

    dataset = KPCNDataset(input_list)

    denoise(kdiffuseNet, kspecularNet, dataset, debug=True)
