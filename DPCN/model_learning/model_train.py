
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

    Net = net(9, input_C).to(device)
    criterion = nn.L1Loss().cuda()

    Optim = optim.Adam(Net.parameters(), lr=learning_rate)
    Loss = 0
    Loss_List = []

    for epoch in range(epochs):

        for _, sample_B in enumerate(dataloader):

            if mode == 'DIFFUSE':
                X_diff = sample_B['X_diff'].permute(permutation).to(device)
                Y_diff = sample_B['Reference'][:, :,
                                               :, :3].permute(permutation).to(device)
            elif mode == 'SPACULAR':
                X_diff = sample_B['X_spec'].permute(permutation).to(device)
                Y_diff = sample_B['Reference'][:, :,
                                               :, 3:6].permute(permutation).to(device)

            Optim.zero_grad()
            Out = Net(X_diff)
            Loss_ = criterion(Out, Y_diff)
            Loss_.backward()
            Optim.step()

            Loss += Loss_.item()

        print("Epoch {}".format(epoch + 1))
        print("LossDiff: {}".format(Loss))
        Loss = 0

    return Net, Loss_List
