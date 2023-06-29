
import torch
import torch.optim as optim
import torch.nn as nn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
