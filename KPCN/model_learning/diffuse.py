
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import os
import numpy as np

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

# model training start


def apply_kernel(weights, data):
    recon_kernel_size = 21

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


def train(mode='KPCN', dataset='', val_dataset='', epochs=40, learning_rate=1e-5):

    input_C = dataset[0]['X_diff'].shape[-1]
    dataloader = torch.utils.data.DataLoader(dataset,  batch_size=4,
                                             shuffle=True, num_workers=4)
    val_dataloader = torch.utils.data.DataLoader(val_dataset,  batch_size=4,
                                                 shuffle=True, num_workers=4)

    permutation = [0, 3, 1, 2]
    diff_Net = net(9, input_C, mode).to(device)
    criterion = nn.L1Loss().cuda()

    diff_Optim = optim.Adam(diff_Net.parameters(), lr=learning_rate)
    train_diff_Loss = 0
    val_diff_Loss = 0
    train_diff_Loss_List = []
    val_diff_Loss_List = []

    for epoch in range(epochs):

        for _, sample_B in enumerate(dataloader):
            X_diff = sample_B['X_diff'].permute(permutation).to(device)
            Y_diff = sample_B['diffuse_GT'].permute(permutation).to(device)

            if mode == 'KPCN':
                outputDiff = diff_Net(X_diff)
                X_input = crop_like(X_diff, outputDiff)
                outputDiff = apply_kernel(outputDiff, X_input)
                Y_diff = crop_like(Y_diff, outputDiff)

            diff_Optim.zero_grad()
            Diff_Loss_ = criterion(outputDiff, Y_diff)
            Diff_Loss_.backward()
            diff_Optim.step()

            train_diff_Loss += Diff_Loss_.item()

        for _, sample_A in enumerate(val_dataloader):
            X_diff = sample_A['X_diff'].permute(permutation).to(device)
            Y_diff = sample_A['diffuse_GT'].permute(permutation).to(device)

            if mode == 'KPCN':
                outputDiff = diff_Net(X_diff)
                X_input = crop_like(X_diff, outputDiff)
                outputDiff = apply_kernel(outputDiff, X_input)
                Y_diff = crop_like(Y_diff, outputDiff)

            diff_Optim.zero_grad()
            Diff_Loss_ = criterion(outputDiff, Y_diff)
            Diff_Loss_.backward()
            diff_Optim.step()

            val_diff_Loss += Diff_Loss_.item()

        train_diff_Loss = train_diff_Loss/len(dataloader)
        val_diff_Loss = val_diff_Loss/len(val_dataloader)

        print("Epoch {}".format(epoch + 1))
        print("train_LossDiff: {}".format(train_diff_Loss))
        print("val_LossDiff: {}".format(val_diff_Loss))

        train_diff_Loss_List.append(train_diff_Loss)
        val_diff_Loss_List.append(val_diff_Loss)

        train_diff_Loss = 0
        val_diff_Loss = 0

    return diff_Net, train_diff_Loss_List, val_diff_Loss_List

# model training end


class KPCNDataset(torch.utils.data.Dataset):
    def __init__(self, samples):
        self.inputs = samples

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx]


class KPCN_val_Dataset(torch.utils.data.Dataset):
    def __init__(self, samples):
        self.inputs = samples

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx]


def diff_model(epochs):
    print('start diff train data load')
    # data reading start
    input_list = []

    data_list = sorted(os.listdir(path='D:/Dataset/sample_KPCN2/KPCN_train'))

    for data in data_list:
        data_torch = torch.load('D:/Dataset/sample_KPCN2/KPCN_train/'+data)
        input_list.append(data_torch)
        print(data)
    
    input_list = to_torch_tensors(input_list)
    input_list = send_to_device(input_list)

    dataset = KPCNDataset(input_list)

    # data reading end
    print('start diff val data load')
    # data reading start
    val_list = []

    val_data_list = sorted(os.listdir(path='D:/Dataset/sample_KPCN2/KPCN_val'))

    for val_data in val_data_list:
        val_data_torch = torch.load(
            'D:/Dataset/sample_KPCN2/KPCN_val/' + val_data)
        val_list.append(val_data_torch)
        print(val_data)

    val_list = to_torch_tensors(val_list)
    val_list = send_to_device(val_list)

    val_dataset = KPCN_val_Dataset(val_list)

    mode = 'KPCN'
    conv_L = 9
    hidden_C = 100
    kernel_S = 5
    kernel_Width = 21

    print('start train')

    diff_N, diff_AC_L, val_diff_AC_L = train(
        mode=mode, dataset=dataset, val_dataset=val_dataset, epochs=epochs, learning_rate=1e-5)

    return diff_N, diff_AC_L, val_diff_AC_L
