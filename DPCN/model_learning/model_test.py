
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
    return np.exp(specular) - 1


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


# returns network input data from noisy .exr file
def preprocess_input(filename, gt, debug=False):

    file = pyexr.open(filename)
    data = file.get_all()

    if debug:
        for k, v in data.items():
            print(k, v.dtype)

    # just in case
    for k, v in data.items():
        data[k] = np.nan_to_num(v)

    file_gt = pyexr.open(gt)
    gt_data = file_gt.get_all()

    # just in case
    for k, v in gt_data.items():
        gt_data[k] = np.nan_to_num(v)

    # clip specular data so we don't have negative values in logarithm
    data['specular'] = np.clip(data['specular'], 0, np.max(data['specular']))
    data['specularVariance'] = np.clip(
        data['specularVariance'], 0, np.max(data['specularVariance']))
    gt_data['specular'] = np.clip(
        data['specular'], 0, np.max(data['specular']))
    gt_data['specularVariance'] = np.clip(
        gt_data['specularVariance'], 0, np.max(gt_data['specularVariance']))

    # save albedo
    data['origAlbedo'] = data['albedo'].copy()

    # save reference data (diffuse and specular)
    diff_ref = preprocess_diffuse(gt_data['diffuse'], gt_data['albedo'])
    spec_ref = preprocess_specular(gt_data['specular'])
    diff_sample = preprocess_diffuse(data['diffuse'], data['albedo'])

    data['Reference'] = np.concatenate(
        (diff_ref[:, :, :3].copy(), spec_ref[:, :, :3].copy()), axis=2)
    data['Sample'] = np.concatenate((diff_sample, data['specular']), axis=2)

    # save final input and reference for error calculation
    # apply albedo and add specular component to get final color
    # postprocess_diffuse(data['Reference'][:,:,:3], data['albedo']) + data['Reference'][:,:,3:]
    data['finalGt'] = gt_data['default']
    # postprocess_diffuse(data['diffuse'][:,:,:3], data['albedo']) + data['specular'][:,:,3:]
    data['finalInput'] = data['default']

    # preprocess diffuse
    data['diffuse'] = preprocess_diffuse(data['diffuse'], data['albedo'])

    # preprocess diffuse variance
    data['diffuseVariance'] = preprocess_diff_var(
        data['diffuseVariance'], data['albedo'])

    # preprocess specular
    data['specular'] = preprocess_specular(data['specular'])

    # preprocess specular variance
    data['specularVariance'] = preprocess_spec_var(
        data['specularVariance'], data['specular'])

    # just in case
    data['depth'] = np.clip(data['depth'], 0, np.max(data['depth']))

    # normalize depth
    max_depth = np.max(data['depth'])
    if (max_depth != 0):
        data['depth'] /= max_depth
        # also have to transform the variance
        data['depthVariance'] /= max_depth * max_depth

    # Calculate gradients of features (not including variances)
    data['gradNormal'] = gradients(data['normal'][:, :, :3].copy())
    data['gradDepth'] = gradients(data['depth'][:, :, :1].copy())
    data['gradAlbedo'] = gradients(data['albedo'][:, :, :3].copy())
    data['gradSpecular'] = gradients(data['specular'][:, :, :3].copy())
    data['gradDiffuse'] = gradients(data['diffuse'][:, :, :3].copy())
    data['gradIrrad'] = gradients(data['default'][:, :, :3].copy())

    # append variances and gradients to data tensors
    data['diffuse'] = np.concatenate(
        (data['diffuse'], data['diffuseVariance'], data['gradDiffuse']), axis=2)
    data['specular'] = np.concatenate(
        (data['specular'], data['specularVariance'], data['gradSpecular']), axis=2)
    data['normal'] = np.concatenate(
        (data['normalVariance'], data['gradNormal']), axis=2)
    data['depth'] = np.concatenate(
        (data['depthVariance'], data['gradDepth']), axis=2)

    if debug:
        for k, v in data.items():
            print(k, v.shape, v.dtype)

    X_diff = np.concatenate((data['diffuse'],
                             data['normal'],
                             data['depth'],
                             data['gradAlbedo']), axis=2)

    X_spec = np.concatenate((data['specular'],
                             data['normal'],
                             data['depth'],
                             data['gradAlbedo']), axis=2)

    assert not np.isnan(X_diff).any()
    assert not np.isnan(X_spec).any()

    print("X_diff shape:", X_diff.shape)
    print(X_diff.dtype, X_spec.dtype)

    data['X_diff'] = X_diff
    data['X_spec'] = X_spec

    remove_channels(data, ('diffuseA', 'specularA', 'normalA', 'albedoA', 'depthA',
                           'visibilityA', 'colorA', 'gradNormal', 'gradDepth', 'gradAlbedo',
                           'gradSpecular', 'gradDiffuse', 'gradIrrad', 'albedo', 'diffuse',
                           'depth', 'specular', 'diffuseVariance', 'specularVariance',
                           'depthVariance', 'visibilityVariance', 'colorVariance',
                           'normalVariance', 'depth', 'visibility'))

    return data


# crops all channels
def crop(data, pos, patch_size):
    half_patch = patch_size // 2
    sx, sy = half_patch, half_patch
    px, py = pos
    return {key: val[(py-sy):(py+sy+1), (px-sx):(px+sx+1), :]
            for key, val in data.items()}


eval_data = preprocess_input(
    "../data/example/10499343-00128spp.exr", "../data/example_GT/10499343-08192spp.exr")

# eval_data = crop(eval_data, (1280//2, 720//2), 300)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# BHWC -> BCHW
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


def unsqueeze_all(d):
    for k, v in d.items():
        d[k] = torch.unsqueeze(v, dim=0)
    return d


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


def denoise(diffuseNet, specularNet, data, debug=False):
    with torch.no_grad():
        criterion = nn.L1Loss()

        data = send_to_device(to_torch_tensors(data))
        if len(data['X_diff'].size()) != 4:
            data = unsqueeze_all(data)

        print(data['X_diff'].size())

        X_diff = data['X_diff'].permute(permutation).to(device)
        Y_diff = data['Reference'][:, :, :, :3].permute(permutation).to(device)

        outputDiff = diffuseNet(X_diff)
        Y_diff = crop_like(Y_diff, outputDiff)

        lossDiff = criterion(outputDiff, Y_diff).item()
        X_spec = data['X_spec'].permute(permutation).to(device)
        Y_spec = data['Reference'][:, :, :, 3:6].permute(
            permutation).to(device)

        outputSpec = specularNet(X_spec)

        Y_spec = crop_like(Y_spec, outputSpec)

        lossSpec = criterion(outputSpec, Y_spec).item()

        albedo = data['origAlbedo'].permute(permutation).to(device)
        albedo = crop_like(albedo, outputDiff)
        outputFinal = outputDiff * (albedo + eps) + torch.exp(outputSpec) - 1.0

        if True:
            print("Sample, denoised, gt")
            sz = 15
            orig = crop_like(data['finalInput'].permute(
                permutation), outputFinal)
            orig = orig.cpu().permute([0, 2, 3, 1]).numpy()[0, :]
            show_data(orig, figsize=(sz, sz), normalize=True)
            img = outputFinal.cpu().permute([0, 2, 3, 1]).numpy()[0, :]
            show_data(img, figsize=(sz, sz), normalize=True)
            gt = crop_like(data['finalGt'].permute(permutation), outputFinal)
            gt = gt.cpu().permute([0, 2, 3, 1]).numpy()[0, :]
            show_data(gt, figsize=(sz, sz), normalize=True)

        Y_final = data['finalGt'].permute(permutation).to(device)

        Y_final = crop_like(Y_final, outputFinal)

        lossFinal = criterion(outputFinal, Y_final).item()

        if debug:
            print("LossDiff:", lossDiff)
            print("LossSpec:", lossSpec)
            print("LossFinal:", lossFinal)


def net(conv_L, input_C):

    layers = [
        nn.Conv2d(input_C, 100,  kernel_size=5, padding=2, stride=1),
        nn.ReLU()
    ]
    for _ in range(conv_L-2):
        layers += [
            nn.Conv2d(100, 100, kernel_size=5, padding=2, stride=1),
            nn.ReLU()

        ]

    layers += [nn.Conv2d(100, 3,  kernel_size=5, padding=2, stride=1)]

    for layer in layers:
        if isinstance(layer, nn.Conv2d):
            nn.init.xavier_uniform_(layer.weight)

    return nn.Sequential(*layers)


kdiffuseNet = net(9, 28).to(device)
kdiffuseNet.load_state_dict(torch.load(
    './model_learning/model/diffuse/DPCN_diff_2.pth'))
kdiffuseNet.eval()

kspecularNet = net(9, 28).to(device)
kspecularNet.load_state_dict(torch.load(
    './model_learning/model/spacular/DPCN_spac_2.pth'))
kspecularNet.eval()


denoise(kdiffuseNet, kspecularNet, eval_data, debug=True)
