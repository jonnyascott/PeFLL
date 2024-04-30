from collections import OrderedDict
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils import spectral_norm


class CNNHyper(nn.Module):
    def __init__(
            self, n_nodes, embedding_dim, in_channels=3, out_dim=10, n_kernels=16, hidden_dim=100,
            spec_norm=False, n_hidden=1):
        super().__init__()

        self.in_channels = in_channels
        self.out_dim = out_dim
        self.n_kernels = n_kernels

        layers = [
            spectral_norm(nn.Linear(embedding_dim, hidden_dim)) if spec_norm else nn.Linear(embedding_dim, hidden_dim),
        ]
        for _ in range(n_hidden):
            layers.append(nn.ReLU(inplace=True))
            layers.append(
                spectral_norm(nn.Linear(hidden_dim, hidden_dim)) if spec_norm else nn.Linear(hidden_dim, hidden_dim),
            )

        self.mlp = nn.Sequential(*layers)

        self.c1_weights = nn.Linear(hidden_dim, self.n_kernels * self.in_channels * 5 * 5)
        self.c1_bias = nn.Linear(hidden_dim, self.n_kernels)
        self.c2_weights = nn.Linear(hidden_dim, 2 * self.n_kernels * self.n_kernels * 5 * 5)
        self.c2_bias = nn.Linear(hidden_dim, 2 * self.n_kernels)
        self.l1_weights = nn.Linear(hidden_dim, 120 * 2 * self.n_kernels * 5 * 5)
        self.l1_bias = nn.Linear(hidden_dim, 120)
        self.l2_weights = nn.Linear(hidden_dim, 84 * 120)
        self.l2_bias = nn.Linear(hidden_dim, 84)
        self.l3_weights = nn.Linear(hidden_dim, self.out_dim * 84)
        self.l3_bias = nn.Linear(hidden_dim, self.out_dim)

        if spec_norm:
            self.c1_weights = spectral_norm(self.c1_weights)
            self.c1_bias = spectral_norm(self.c1_bias)
            self.c2_weights = spectral_norm(self.c2_weights)
            self.c2_bias = spectral_norm(self.c2_bias)
            self.l1_weights = spectral_norm(self.l1_weights)
            self.l1_bias = spectral_norm(self.l1_bias)
            self.l2_weights = spectral_norm(self.l2_weights)
            self.l2_bias = spectral_norm(self.l2_bias)
            self.l3_weights = spectral_norm(self.l3_weights)
            self.l3_bias = spectral_norm(self.l3_bias)

    def forward(self, v):
        # emd = self.embeddings(idx)
        features = self.mlp(v)

        weights = OrderedDict({
            "conv1.weight": self.c1_weights(features).view(self.n_kernels, self.in_channels, 5, 5),
            "conv1.bias": self.c1_bias(features).view(-1),
            "conv2.weight": self.c2_weights(features).view(2 * self.n_kernels, self.n_kernels, 5, 5),
            "conv2.bias": self.c2_bias(features).view(-1),
            "fc1.weight": self.l1_weights(features).view(120, 2 * self.n_kernels * 5 * 5),
            "fc1.bias": self.l1_bias(features).view(-1),
            "fc2.weight": self.l2_weights(features).view(84, 120),
            "fc2.bias": self.l2_bias(features).view(-1),
            "fc3.weight": self.l3_weights(features).view(self.out_dim, 84),
            "fc3.bias": self.l3_bias(features).view(-1),
        })
        return weights


class CNNTarget(nn.Module):
    def __init__(self, in_channels=3, n_kernels=16, out_dim=10):
        super(CNNTarget, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, n_kernels, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(n_kernels, 2 * n_kernels, 5)
        self.fc1 = nn.Linear(2 * n_kernels * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, out_dim)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class MLP(nn.Module):
    def __init__(self, embed_x, embed_y, dim_x, dim_y, hidden_dim, out_dim, n_layers):
        super(MLP, self).__init__()

        in_dim = embed_x * dim_x + embed_y * dim_y
        assert in_dim != 0

        if n_layers == 1:
            layers = [nn.Linear(in_dim, out_dim)]
        else:
            layers = [nn.Linear(in_dim, hidden_dim), nn.ReLU()]
            for _ in range(n_layers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                layers.append(nn.ReLU())

            layers.append(nn.Linear(hidden_dim, out_dim))

        self.mlp = nn.Sequential(*layers)
        # print(self.mlp.state_dict())
        self.embed_x = embed_x
        self.embed_y = embed_y

    def forward(self, B):
        x, y = B
        if self.embed_x:
            inp = torch.flatten(x, start_dim=1, end_dim=-1)
            if self.embed_y:
                inp = torch.cat((inp, y), dim=1)
        else:
            inp = y

        out = self.mlp(inp)
        # print(torch.mean(out, 0), torch.var(out, 0))
        return out


class CNNEmbed(nn.Module):
    def __init__(self, embed_y, dim_y, embed_dim, device, in_channels=3, n_kernels=16):
        super(CNNEmbed, self).__init__()

        in_channels += embed_y * dim_y

        self.conv1 = nn.Conv2d(in_channels, n_kernels, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(n_kernels, 2 * n_kernels, 5)
        self.fc1 = nn.Linear(2 * n_kernels * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, embed_dim)

        self.embed_y = embed_y
        self.device = device

    def forward(self, B):
        x, y = B
        if self.embed_y:
            y = y.view(y.size(0), y.size(1), 1, 1)
            c = torch.zeros((x.size(0), y.size(1), x.size(2), x.size(3))).to(self.device)
            c += y
            inp = torch.cat((x, c), dim=1)
        else:
            inp = x

        x = self.pool(F.relu(self.conv1(inp)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class MLPEmbed(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(MLPEmbed, self).__init__()

        self.fc = nn.Linear(in_dim, out_dim, bias=True)

    def forward(self, B):
        _, y = B
        v = self.fc(y)
        return v


class EmbedHyper(nn.Module):
    def __init__(self, embednet, hypernet):
        super(EmbedHyper, self).__init__()

        self.embednet = embednet
        self.hypernet = hypernet

    def forward(self, x):
        v = self.embednet(x)
        out = self.hypernet(v)
        return out

    def embed(self, x):
        return self.embednet(x)

    def hyper(self, v):
        return self.hypernet(v)
