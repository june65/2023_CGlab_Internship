
import torch
from model_train import *
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
