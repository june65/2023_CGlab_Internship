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

file = pyexr.open("../data/example/10499343-00128spp.exr")

print("Width:", file.width)
print("Height:", file.height)

print("Available channels:")
file.describe_channels()

print("Default channels:", file.channel_map['default'])


def show_data(data, figsize=(15, 15), normalize=False):
    if normalize:
        data = np.clip(data, 0, 1)**0.45454545
    plt.figure(figsize=figsize)
    imgplot = plt.imshow(data, aspect='equal')
    imgplot.axes.get_xaxis().set_visible(False)
    imgplot.axes.get_yaxis().set_visible(False)
    plt.show()


def show_data_sbs(data1, data2, figsize=(15, 15)):
    f, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=figsize)

    ax1.imshow(data1, aspect='equal')
    ax2.imshow(data2, aspect='equal')

    ax1.axis('off')
    ax2.axis('off')

    plt.show()


def show_channel(img, chan):
    data = img.get(chan)
    print("Channel:", chan)
    print("Shape:", data.shape)
    print(np.max(data), np.min(data))

    if chan in ["default", "diffuse", "albedo", "specular"]:
        data = np.clip(data, 0, 1)**0.45454545

    if chan in ["normal", "normalA"]:
        # normalize
        print("normalizing")
        for i in range(img.height):
            for j in range(img.width):
                data[i][j] = data[i][j] / np.linalg.norm(data[i][j])
        data = np.abs(data)

    if chan in ["depth", "visibility", "normalVariance"] and np.max(data) != 0:
        data /= np.max(data)

    if data.shape[2] == 1:
        print("Reshaping")
        data = data.reshape(img.height, img.width)

    show_data(data)


patch_size = 64  # patches are 64x64
n_patches = 400
eps = 0.00316


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


def remove_channels(data, channels):
    for c in channels:
        if c in data:
            del data[c]
        else:
            print("Channel {} not found in data!".format(c))


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


def getVarianceMap(data, patch_size, relative=False):

    # introduce a dummy third dimension if needed
    if data.ndim < 3:
        data = data[:, :, np.newaxis]

    # compute variance
    mean = ndimage.uniform_filter(data, size=(patch_size, patch_size, 1))
    sqrmean = ndimage.uniform_filter(data**2, size=(patch_size, patch_size, 1))
    variance = np.maximum(sqrmean - mean**2, 0)

    # convert to relative variance if requested
    if relative:
        variance = variance/np.maximum(mean**2, 1e-2)

    # take the max variance along the three channels, gamma correct it to get a
    # less peaky map, and normalize it to the range [0,1]
    variance = variance.max(axis=2)
    variance = np.minimum(variance**(1.0/2.2), 1.0)

    return variance/variance.max()


# Generate importance sampling map based on buffer and desired metric
def getImportanceMap(buffers, metrics, weights, patch_size):
    if len(metrics) != len(buffers):
        metrics = [metrics[0]]*len(buffers)
    if len(weights) != len(buffers):
        weights = [weights[0]]*len(buffers)
    impMap = None
    for buf, metric, weight in zip(buffers, metrics, weights):
        if metric == 'uniform':
            cur = np.ones(buf.shape[:2], dtype=np.float)
        elif metric == 'variance':
            cur = getVarianceMap(buf, patch_size, relative=False)
        elif metric == 'relvar':
            cur = getVarianceMap(buf, patch_size, relative=True)
        else:
            print('Unexpected metric:', metric)
        if impMap is None:
            impMap = cur*weight
        else:
            impMap += cur*weight
    return impMap / impMap.max()


def samplePatchesProg(img_dim, patch_size, n_samples, maxiter=5000):

    # Sample patches using dart throwing (works well for sparse/non-overlapping patches)

    # estimate each sample patch area
    full_area = float(img_dim[0]*img_dim[1])
    sample_area = full_area/n_samples

    # get corresponding dart throwing radius
    radius = np.sqrt(sample_area/np.pi)
    minsqrdist = (2*radius)**2

    # compute the distance to the closest patch
    def get_sqrdist(x, y, patches):
        if len(patches) == 0:
            return np.infty
        dist = patches - [x, y]
        return np.sum(dist**2, axis=1).min()

    # perform dart throwing, progressively reducing the radius
    rate = 0.96
    patches = np.zeros((n_samples, 2), dtype=int)
    xmin, xmax = 0, img_dim[1] - patch_size[1] - 1
    ymin, ymax = 0, img_dim[0] - patch_size[0] - 1
    for patch in range(n_samples):
        done = False
        while not done:
            for i in range(maxiter):
                x = randint(xmin, xmax)
                y = randint(ymin, ymax)
                sqrdist = get_sqrdist(x, y, patches[:patch, :])
                if sqrdist > minsqrdist:
                    patches[patch, :] = [x, y]
                    done = True
                    break
            if not done:
                radius *= rate
                minsqrdist = (2*radius)**2

    return patches


def prunePatches(shape, patches, patchsize, imp):

    pruned = np.empty_like(patches)

    # Generate a set of regions tiling the image using snake ordering.
    def get_regions_list(shape, step):
        regions = []
        for y in range(0, shape[0], step):
            if y//step % 2 == 0:
                xrange = range(0, shape[1], step)
            else:
                xrange = reversed(range(0, shape[1], step))
            for x in xrange:
                regions.append((x, x + step, y, y + step))
        return regions

    # Split 'patches' in current and remaining sets, where 'cur' holds the
    # patches in the requested region, and 'rem' holds the remaining patches.
    def split_patches(patches, region):
        cur = np.empty_like(patches)
        rem = np.empty_like(patches)
        ccount, rcount = 0, 0
        for i in range(patches.shape[0]):
            x, y = patches[i, 0], patches[i, 1]
            if region[0] <= x < region[1] and region[2] <= y < region[3]:
                cur[ccount, :] = [x, y]
                ccount += 1
            else:
                rem[rcount, :] = [x, y]
                rcount += 1
        return cur[:ccount, :], rem[:rcount, :]

    # Process all patches, region by region, pruning them randomly according to
    # their importance value, ie. patches with low importance have a higher
    # chance of getting pruned. To offset the impact of the binary pruning
    # decision, we propagate the discretization error and take it into account
    # when pruning.
    rem = np.copy(patches)
    count, error = 0, 0
    for region in get_regions_list(shape, 4*patchsize):
        cur, rem = split_patches(rem, region)
        for i in range(cur.shape[0]):
            x, y = cur[i, 0], cur[i, 1]
            if imp[y, x] - error > random.random():
                pruned[count, :] = [x, y]
                count += 1
                error += 1 - imp[y, x]
            else:
                error += 0 - imp[y, x]

    return pruned[:count, :]


def importanceSampling(data, debug=False):
    global patch_size, n_patches

    # extract buffers
    buffers = []
    for b in ['default', 'normal']:
        buffers.append(data[b][:, :, :3])

    # build the metric map
    metrics = ['relvar', 'variance']
    weights = [1.0, 1.0]
    imp = getImportanceMap(buffers, metrics, weights, patch_size)

    if debug:
        print("Importance map:")
        plt.figure(figsize=(15, 15))
        imgplot = plt.imshow(imp)
        imgplot.axes.get_xaxis().set_visible(False)
        imgplot.axes.get_yaxis().set_visible(False)
        plt.show()

    # get patches
    patches = samplePatchesProg(
        buffers[0].shape[:2], (patch_size, patch_size), n_patches)

    if debug:
        print("Patches:")
        plt.figure(figsize=(15, 10))
        plt.scatter(list(a[0] for a in patches), list(a[1] for a in patches))
        plt.show()

    selection = buffers[0] * 0.1
    for i in range(patches.shape[0]):
        x, y = patches[i, 0], patches[i, 1]
        selection[y:y+patch_size, x:x+patch_size,
                  :] = buffers[0][y:y+patch_size, x:x+patch_size, :]

    # prune patches
    pad = patch_size // 2
    pruned = np.maximum(0, prunePatches(
        buffers[0].shape[:2], patches + pad, patch_size, imp) - pad)
    selection = buffers[0]*0.1
    for i in range(pruned.shape[0]):
        x, y = pruned[i, 0], pruned[i, 1]
        selection[y:y+patch_size, x:x+patch_size,
                  :] = buffers[0][y:y+patch_size, x:x+patch_size, :]

    if debug:
        print("After pruning:")
        plt.figure(figsize=(15, 10))
        plt.scatter(list(a[0] for a in pruned), list(a[1] for a in pruned))
        plt.show()

    return (pruned + pad)


class KPCNDataset(torch.utils.data.Dataset):
    def __init__(self):
        self.samples = cropped

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


if __name__ == '__main__':

    '''
        file = pyexr.open('../data/example/10499343-00128spp.exr')

        for chan in ["default", "albedo", "diffuse", "specular", "normal", "normalVariance", "depth", "visibility"]:
            show_channel(file, chan)
    '''
    ''' 
    data = preprocess_input("../data/example/10499343-00128spp.exr",
                            "../data/example_GT/10499343-08192spp.exr", debug=True)

    for k, v in data.items():
        print(k, "has nans:", np.isnan(v).any())

    print("Let's check finalGt: ")
    show_data(np.clip(data['finalGt'], 0, 1)**0.45454545)
    '''

    data = preprocess_input("../data/example/10499343-00128spp.exr",
                            "../data/example_GT/10499343-08192spp.exr", debug=True)
    patches = importanceSampling(data, debug=True)

    def crop(data, pos, patch_size):
        half_patch = patch_size // 2
        sx, sy = half_patch, half_patch
        px, py = pos
        return {key: val[(py-sy):(py+sy+1), (px-sx):(px+sx+1), :]
                for key, val in data.items()}

    cropped = list(crop(data, tuple(pos), patch_size) for pos in patches)

    print(len(cropped))

    print("Sampling of chosen patches:")
    for i in range(5):
        data_ = np.clip(cropped[np.random.randint(
            0, len(cropped))]['default'], 0, 1)**0.45454545
        plt.figure(figsize=(5, 5))
        imgplot = plt.imshow(data_)
        imgplot.axes.get_xaxis().set_visible(False)
        imgplot.axes.get_yaxis().set_visible(False)
        plt.show()

    def get_cropped_patches(sample_file, gt_file):
        data = preprocess_input(sample_file, gt_file)
        patches = importanceSampling(data)
        cropped = list(crop(data, tuple(pos), patch_size) for pos in patches)

        return cropped

    for f in glob.glob('sample*.exr'):
        if f == 'sample.exr':
            continue
        num = f[len('sample'):f.index('.')]
        sample_name = 'sample{}.exr'.format(num)
        gt_name = 'gt{}.exr'.format(num)
        print(sample_name, gt_name)
        cropped += get_cropped_patches(sample_name, gt_name)

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

    print(len(cropped))
    print(type(cropped[0]))

    for i, v in enumerate(cropped):
        torch.save(v, '../data/sample/sample'+str(i+1)+'.pt')

    import gc
    import sys

    def getsize(obj):
        marked = {id(obj)}
        obj_q = [obj]
        sz = 0

        while obj_q:
            sz += sum(map(sys.getsizeof, obj_q))

            # Lookup all the object reffered to by the object in obj_q.
            # See: https://docs.python.org/3.7/library/gc.html#gc.get_referents
            all_refr = ((id(o), o) for o in gc.get_referents(*obj_q))

            # Filter object that are already marked.
            # Using dict notation will prevent repeated objects.
            new_refr = {
                o_id: o for o_id, o in all_refr if o_id not in marked and not isinstance(o, type)}

            # The new obj_q will be the ones that were not marked,
            # and we will update marked with their ids so we will
            # not traverse them again.
            obj_q = new_refr.values()
            marked.update(new_refr.keys())

        return sz

    for k, v in cropped[0].items():
        print(k, getsize(v))

    print(getsize(cropped) / 1024 / 1024, "MiB")

    cropped = to_torch_tensors(cropped)
    cropped = send_to_device(cropped)

    dataset = KPCNDataset()

    print("Left: noisy,    Right: reference")

    for i in range(len(dataset)):
        sample = dataset[np.random.randint(0, len(dataset))]

        data_sample = np.clip(sample['default'], 0, 1)**0.45454545
        data_ref = np.clip(sample['finalGt'], 0, 1)**0.45454545
        show_data_sbs(data_sample, data_ref, figsize=(4, 2))

        if i == 6:
            break
