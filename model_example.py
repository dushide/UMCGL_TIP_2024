import torch.nn.functional as F
import torch
import torch.nn as nn
import numpy as np
import numpy.linalg as LA
from sklearn.preprocessing import normalize
from utils.clusteringPerformance import similarity_function, StatisticClustering

class Discriminator(nn.Module):
    """
    View-specific discriminator
    """
    def __init__(self, D0_channel_1, D0_channel_2, D0_kernel_1, D0_stride_1, D0_padding_1, D0_kernel_2, D0_stride_2, D0_padding_2, use_leaky_relu, device):
        super(Discriminator, self).__init__()

        self.device = device
        self.use_leaky_relu = use_leaky_relu

        if use_leaky_relu:
            self.D = nn.Sequential(
                nn.Conv2d(D0_channel_1, D0_channel_2, D0_kernel_1, D0_stride_1, D0_padding_1).float(),
                nn.LeakyReLU(negative_slope=0.1, inplace=True),
                nn.Conv2d(D0_channel_2, D0_channel_1, D0_kernel_2, D0_stride_2, D0_padding_2).float(),
                nn.LeakyReLU(negative_slope=0.1),
            ).to(self.device)
        else:
            self.D = nn.Sequential(
                nn.Conv2d(D0_channel_1, D0_channel_2, D0_kernel_1, D0_stride_1, D0_padding_1).float(),
                nn.ReLU(inplace=True),
                nn.Conv2d(D0_channel_2, D0_channel_1, D0_kernel_2, D0_stride_2, D0_padding_2).float(),
                nn.ReLU(),
            ).to(self.device)

    def forward(self, W):
        return self.D(W)

class Generator(nn.Module):
    """
    Shared generator
    """
    def __init__(self, epochs, lr, gnd, si, nc, n, n_view,
                 conv_channel_1, conv_channel_2, conv_kernel_1, conv_stride_1, conv_padding_1, conv_kernel_2, conv_stride_2, conv_padding_2,
                 convv_channel_2, convv_kernel_1, convv_stride_1, convv_padding_1,
                 conv1_channel_1, conv1_channel_2, conv1_kernel_1, conv1_stride_1, conv1_padding_1,
                 conv3_channel_2, conv3_channel_1, conv3_kernel_2, conv3_stride_2, conv3_padding_2,
                 up1_channel_1, up1_channel_2, up1_kernel_1, up1_stride_1, up1_padding_1,
                 up3_channel_2, up3_channel_1, up3_kernel_2, up3_stride_2, up3_padding_2,
                 G_channel_1, G_channel_2, G_kernel_1, G_stride_1, G_padding_1, G_kernel_2, G_stride_2, G_padding_2,
                 use_relu, use_relu_1, device):
        super(Generator, self).__init__()

        self.gnd = gnd
        self.epochs = epochs
        self.lr = lr
        self.si = si
        self.nc = nc
        self.n_view = n_view
        self.device = device
        self.use_relu = use_relu
        self.use_relu_1 = use_relu_1
        self.n = n

        if use_relu_1:
            self.conv = nn.Sequential(
                nn.Conv2d(conv_channel_1, conv_channel_2, conv_kernel_1, conv_stride_1, conv_padding_1).float(),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(conv_channel_2, conv_channel_1, conv_kernel_2, conv_stride_2, conv_padding_2).float(),
                nn.LeakyReLU(),
            ).to(self.device)

            self.convv = nn.Sequential(nn.Conv2d(self.n_view, convv_channel_2, convv_kernel_1, convv_stride_1, convv_padding_1).float(),
                                       nn.LeakyReLU(inplace=True), ).to(self.device)

            self.conv1 = nn.Conv2d(conv1_channel_1, conv1_channel_2, conv1_kernel_1, conv1_stride_1, conv1_padding_1).float().to(self.device)
            self.conv3 = nn.Conv2d(conv3_channel_2, conv3_channel_1, conv3_kernel_2, conv3_stride_2, conv3_padding_2).float().to(self.device)

            self.up1 = nn.ConvTranspose2d(up1_channel_1, up1_channel_2, up1_kernel_1, up1_stride_1, up1_padding_1).float().to(self.device)
            self.up3 = nn.ConvTranspose2d(up3_channel_2, up3_channel_1, up3_kernel_2, up3_stride_2, up3_padding_2).float().to(self.device)

            self.G = nn.Sequential(
                nn.Conv2d(G_channel_1, G_channel_2, G_kernel_1, G_stride_1, G_padding_1).float(),
                nn.ReLU(inplace=True),
                nn.Conv2d(G_channel_2, G_channel_1, G_kernel_2, G_stride_2, G_padding_2).float(),
                nn.ReLU(),
            ).to(self.device)
        elif use_relu:
            self.conv = nn.Sequential(
                nn.Conv2d(conv_channel_1, conv_channel_2, conv_kernel_1, conv_stride_1, conv_padding_1).float(),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(conv_channel_2, conv_channel_1, conv_kernel_2, conv_stride_2, conv_padding_2).float(),
                nn.LeakyReLU(),
            ).to(self.device)

            self.convv = nn.Sequential(nn.Conv2d(self.n_view, convv_channel_2, convv_kernel_1, convv_stride_1, convv_padding_1).float(),
                                       nn.LeakyReLU(inplace=True), ).to(self.device)

            self.G = nn.Sequential(
                nn.Conv2d(G_channel_1, G_channel_2, G_kernel_1, G_stride_1, G_padding_1).float(),
                nn.ReLU(inplace=True),
                nn.Conv2d(G_channel_2, G_channel_1, G_kernel_2, G_stride_2, G_padding_2).float(),
                nn.ReLU(),
            ).to(self.device)

            self.conv1 = nn.Conv2d(conv1_channel_1, conv1_channel_2, conv1_kernel_1, conv1_stride_1, conv1_padding_1).float().to(self.device)
            self.conv3 = nn.Conv2d(conv3_channel_2, conv3_channel_1, conv3_kernel_2, conv3_stride_2, conv3_padding_2).float().to(self.device)

            self.up1 = nn.ConvTranspose2d(up1_channel_1, up1_channel_2, up1_kernel_1, up1_stride_1, up1_padding_1).float().to(self.device)
            self.up3 = nn.ConvTranspose2d(up3_channel_2, up3_channel_1, up3_kernel_2, up3_stride_2, up3_padding_2).float().to(self.device)
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(conv_channel_1, conv_channel_2, conv_kernel_1, conv_stride_1, conv_padding_1).float(),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(conv_channel_2, conv_channel_1, conv_kernel_2, conv_stride_2, conv_padding_2).float(),
                nn.LeakyReLU(),
            ).to(self.device)

            self.convv = nn.Sequential(
                nn.Conv2d(self.n_view, convv_channel_2, convv_kernel_1, convv_stride_1, convv_padding_1).float(),
                nn.LeakyReLU(inplace=True), ).to(self.device)

            self.conv1 = nn.Conv2d(conv1_channel_1, conv1_channel_2, conv1_kernel_1, conv1_stride_1,
                                   conv1_padding_1).float().to(self.device)
            self.conv3 = nn.Conv2d(conv3_channel_2, conv3_channel_1, conv3_kernel_2, conv3_stride_2,
                                   conv3_padding_2).float().to(self.device)

            self.up1 = nn.ConvTranspose2d(up1_channel_1, up1_channel_2, up1_kernel_1, up1_stride_1,
                                          up1_padding_1).float().to(self.device)
            self.up3 = nn.ConvTranspose2d(up3_channel_2, up3_channel_1, up3_kernel_2, up3_stride_2,
                                          up3_padding_2).float().to(self.device)

            self.G = nn.Sequential(
                nn.Conv2d(G_channel_1, G_channel_2, G_kernel_1, G_stride_1, G_padding_1).float(),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(G_channel_2, G_channel_1, G_kernel_2, G_stride_2, G_padding_2).float(),
                nn.LeakyReLU(),
            ).to(self.device)

    def encoder(self, x):
        """
        Convolutional autoencoder encoder
        """
        conv1 = self.conv1(x)
        relu1 = F.relu(conv1)

        conv3 = self.conv3(relu1)
        relu3 = F.relu(conv3).view(1, 1, self.si, self.si)
        return relu3

    def decoder(self, encoding):
        """
        Convolutional autoencoder decoder
        """
        up1 = self.up1(encoding)
        up1_relu = F.relu(up1)
        up3 = self.up3(up1_relu)
        return (F.relu(up3)).view(1, 1, self.si, self.si)

    def forward_one(self, x):
        """
        Multi-channel fusion
        """
        x = self.conv(x)
        x = x.view(1, 1, self.n, self.n)
        return x

    def forward(self, W, W_ran):
        """
        Network Modules of UMCGL
        """

        G = {}
        out = {}
        # Generative View-specific Diversity Module
        for i in range(self.n_view):
            G[i] = self.G(W_ran[i])

        # Contrastive Cross-view Diversity Module
        for i in range(self.n_view):
            out[i] = self.forward_one(G[i])

        dis = torch.abs(out[0] - out[1])
        for i in range(2, self.n_view):
            dis = torch.abs(dis - out[i])

        # View-specific feature extractor of Multi-channel Graph Consistency Module
        encodings = {}
        decodings = {}
        for i in range(self.n_view):
            encodings[i] = self.encoder(W[i])
            decodings[i] = self.decoder(encodings[i])

        dis_out = {}
        # Multi-channel fusion of Multi-channel Graph Consistency Module
        for i in range(self.n_view):
            dis_out[i] = encodings[i]
            dis_out[i] = dis_out[i].squeeze()

        fussion = torch.zeros([1, self.n_view, self.si, self.si]).to(self.device)
        for i in range(self.n_view):
            fussion[0][i] = dis_out[i]

        out = self.convv(fussion)
        fus_out = out

        # Consensus Graph Learning Module
        out = dis + out
        encoding = self.encoder(out)  # Final Consensus Graph
        decoding = self.decoder(encoding)

        return G, dis, fus_out, out, encoding, decoding, encodings, decodings