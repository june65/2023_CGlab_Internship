
import torch
import torch.optim as optim
import torch.nn as nn
import os

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


def train(mode='DIFFUSE', dataset='', epochs=40, learning_rate=1e-5):
    input_C = dataset[0]['X_diff'].shape[-1]
    permutation = [0, 3, 1, 2]

    dataloader = torch.utils.data.DataLoader(dataset,  batch_size=4,
                                             shuffle=True, num_workers=4)

    diff_Net = net(9, input_C).to(device)
    criterion = nn.L1Loss().cuda()

    diff_Optim = optim.Adam(diff_Net.parameters(), lr=learning_rate)
    diff_Loss = 0
    diff_Loss_List = []

    for epoch in range(epochs):

        for _, sample_B in enumerate(dataloader):

            X_diff = sample_B['X_diff'].permute(permutation).to(device)
            Y_diff = sample_B['Reference'][:, :,
                                           :, :3].permute(permutation).to(device)
            diff_Optim.zero_grad()
            diff_Out = diff_Net(X_diff)
            Diff_Loss_ = criterion(diff_Out, Y_diff)
            Diff_Loss_.backward()
            diff_Optim.step()

            diff_Loss += Diff_Loss_.item()

        print("Epoch {}".format(epoch + 1))
        print("LossDiff: {}".format(diff_Loss))
        diff_Loss = 0

    return diff_Net, diff_Loss_List

# model training end


class KPCNDataset(torch.utils.data.Dataset):
    def __init__(self):
        self.samples = cropped

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


if __name__ == '__main__':

    # data reading start

    cropped = []

    data_list = sorted(os.listdir(path='../data/sample'))

    for data in data_list:
        v = torch.load('../data/sample/'+data)
        cropped.append(v)

    cropped = to_torch_tensors(cropped)
    cropped = send_to_device(cropped)

    dataset = KPCNDataset()

    # data reading end

    mode = 'DIFFUSE'
    conv_L = 9
    hidden_C = 100
    kernel_S = 5
    kernel_Width = 21

    diff_N, diff_AC_L = train(
        mode=mode, dataset=dataset, epochs=40, learning_rate=1e-5)
