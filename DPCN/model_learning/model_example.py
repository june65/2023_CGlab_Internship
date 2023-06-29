# imports

import torch.optim as optim
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

import gc
import sys

# some constants

patch_size = 64  # patches are 64x64
n_patches = 400
eps = 0.00316

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


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


class KPCNDataset(torch.utils.data.Dataset):
    def __init__(self):
        self.samples = cropped

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

# torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)


def make_net(n_layers, mode):
    # create first layer manually
    layers = [
        nn.Conv2d(input_channels, hidden_channels, kernel_size),
        nn.ReLU()
    ]

    for l in range(n_layers-2):
        layers += [
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size),
            nn.ReLU()
        ]

        params = sum(p.numel()
                     for p in layers[-2].parameters() if p.requires_grad)
        print(params)

    out_channels = 3 if mode == 'DPCN' else recon_kernel_size**2
    # , padding=18)]
    layers += [nn.Conv2d(hidden_channels, out_channels, kernel_size)]

    for layer in layers:
        if isinstance(layer, nn.Conv2d):
            nn.init.xavier_uniform_(layer.weight)

    return nn.Sequential(*layers)


def apply_kernel(weights, data):
    # apply softmax to kernel weights
    weights = weights.permute((0, 2, 3, 1)).to(device)
    _, _, h, w = data.size()
    weights = F.softmax(weights, dim=3).view(-1, w * h,
                                             recon_kernel_size, recon_kernel_size)

    # now we have to apply kernels to every pixel
    # first pad the input
    r = recon_kernel_size // 2
    data = F.pad(data[:, :3, :, :], (r,) * 4, "reflect")

    # print(data[0,:,:,:])

    # make slices
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
            # slices.append(data[:,:,sy:ey,sx:ex])

    reds = (torch.cat(R, dim=1).to(device)*weights).sum(2).sum(2)
    greens = (torch.cat(G, dim=1).to(device)*weights).sum(2).sum(2)
    blues = (torch.cat(B, dim=1).to(device)*weights).sum(2).sum(2)

    # pixels = torch.cat(slices, dim=1).to(device)
    # kerns = torch.cat(kernels, dim=1).to(device)

    # print("Kerns:", kerns.size())
    # print(kerns[0,:5,:,:])
    # print("Pixels:", pixels.size())
    # print(pixels[0,:5,:,:])

    # res = (pixels * kerns).sum(2).sum(2).view(-1, 3, h, w).to(device)

    # tmp = (pixels * kerns).sum(2).sum(2)

    # print(tmp.size(), tmp[0,:10])

    # print("Res:", res.size(), res[0,:5,:,:])
    # print("Data:", data[0,:5,:,:])

    res = torch.cat((reds, greens, blues), dim=1).view(-1, 3, h, w).to(device)

    return res


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


def train(mode='KPCN', epochs=20, learning_rate=1e-4, show_images=False):
    dataset = KPCNDataset()
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4,
                                             shuffle=True, num_workers=4)

    # instantiate networks
    diffuseNet = make_net(L, mode).to(device)
    specularNet = make_net(L, mode).to(device)

    print(diffuseNet, "CUDA:", next(diffuseNet.parameters()).is_cuda)
    print(specularNet, "CUDA:", next(specularNet.parameters()).is_cuda)

    criterion = nn.L1Loss()

    optimizerDiff = optim.Adam(diffuseNet.parameters(), lr=learning_rate)
    optimizerSpec = optim.Adam(specularNet.parameters(), lr=learning_rate)

    accuLossDiff = 0
    accuLossSpec = 0
    accuLossFinal = 0

    lDiff = []
    lSpec = []
    lFinal = []

    import time

    start = time.time()

    for epoch in range(epochs):
        for i_batch, sample_batched in enumerate(dataloader):
            # print(i_batch)

            # get the inputs
            X_diff = sample_batched['X_diff'].permute(permutation).to(device)
            Y_diff = sample_batched['Reference'][:, :,
                                                 :, :3].permute(permutation).to(device)

            # zero the parameter gradients
            optimizerDiff.zero_grad()

            # forward + backward + optimize
            outputDiff = diffuseNet(X_diff)

            # print(outputDiff.shape)

            if mode == 'KPCN':
                X_input = crop_like(X_diff, outputDiff)
                outputDiff = apply_kernel(outputDiff, X_input)

            Y_diff = crop_like(Y_diff, outputDiff)

            lossDiff = criterion(outputDiff, Y_diff)
            lossDiff.backward()
            optimizerDiff.step()

            # get the inputs
            X_spec = sample_batched['X_spec'].permute(permutation).to(device)
            Y_spec = sample_batched['Reference'][:, :,
                                                 :, 3:6].permute(permutation).to(device)

            # zero the parameter gradients
            optimizerSpec.zero_grad()

            # forward + backward + optimize
            outputSpec = specularNet(X_spec)

            if mode == 'KPCN':
                X_input = crop_like(X_spec, outputSpec)
                outputSpec = apply_kernel(outputSpec, X_input)

            Y_spec = crop_like(Y_spec, outputSpec)

            lossSpec = criterion(outputSpec, Y_spec)
            lossSpec.backward()
            optimizerSpec.step()

            # calculate final ground truth error
            with torch.no_grad():
                albedo = sample_batched['origAlbedo'].permute(
                    permutation).to(device)
                albedo = crop_like(albedo, outputDiff)
                outputFinal = outputDiff * \
                    (albedo + eps) + torch.exp(outputSpec) - 1.0

                if False:  # i_batch % 500:
                    print("Sample, denoised, gt")
                    sz = 3
                    orig = crop_like(sample_batched['finalInput'].permute(
                        permutation), outputFinal)
                    orig = orig.cpu().permute([0, 2, 3, 1]).numpy()[0, :]
                    show_data(orig, figsize=(sz, sz), normalize=True)
                    img = outputFinal.cpu().permute([0, 2, 3, 1]).numpy()[0, :]
                    show_data(img, figsize=(sz, sz), normalize=True)
                    gt = crop_like(sample_batched['finalGt'].permute(
                        permutation), outputFinal)
                    gt = gt.cpu().permute([0, 2, 3, 1]).numpy()[0, :]
                    show_data(gt, figsize=(sz, sz), normalize=True)

                Y_final = sample_batched['finalGt'].permute(
                    permutation).to(device)

                Y_final = crop_like(Y_final, outputFinal)

                lossFinal = criterion(outputFinal, Y_final)

                accuLossFinal += lossFinal.item()

            accuLossDiff += lossDiff.item()
            accuLossSpec += lossSpec.item()

        print("Epoch {}".format(epoch + 1))
        print("LossDiff: {}".format(accuLossDiff))
        print("LossSpec: {}".format(accuLossSpec))
        print("LossFinal: {}".format(accuLossFinal))

        lDiff.append(accuLossDiff)
        lSpec.append(accuLossSpec)
        lFinal.append(accuLossFinal)

        accuLossDiff = 0
        accuLossSpec = 0
        accuLossFinal = 0

    print('Finished training in mode', mode)
    print('Took', time.time() - start, 'seconds.')

    return diffuseNet, specularNet, lDiff, lSpec, lFinal


if __name__ == '__main__':
    cropped = []

    data_list = sorted(os.listdir(path='../data/sample'))

    for data in data_list:
        v = torch.load('../data/sample/'+data)
        cropped.append(v)

    cropped = to_torch_tensors(cropped)
    cropped = send_to_device(cropped)

    dataset = KPCNDataset()

    mode = 'KPCN'  # 'KPCN' or 'DPCN'

    recon_kernel_size = 21

    # some network parameters
    L = 9  # number of convolutional layers
    n_kernels = 100  # number of kernels in each layer
    kernel_size = 5  # size of kernel (square)

    input_channels = dataset[0]['X_diff'].shape[-1]
    hidden_channels = 100

    print("Input channels:", input_channels)

    # BHWC -> BCHW
    permutation = [0, 3, 1, 2]

    ddiffuseNet, dspecularNet, dlDiff, dlSpec, dlFinal = train(
        mode='DPCN', epochs=40, learning_rate=1e-5)
