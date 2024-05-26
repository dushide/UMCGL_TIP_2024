import torch.nn.functional as F
import torch
import random
import os
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.preprocessing import normalize
from utils.DataLoader import loadMGW, loadMGW1, loadMGW2, loadMGW3, loadMGW4, loadData
from utils.Utils import spectral_clustering, GaussianNoise, GaussianNoise1, clustering, weights_init
from model_example import Discriminator, Generator
from args_UMCGL import parameter_parser

def train(D_net, G_net, X, X_ran, lr, epochss, d_steps, g_steps, n_view, n, gnd, alpha, beta, datasetname, device):
    """
    Train UMCGL
    """
    d_steps = d_steps
    g_steps = g_steps
    epochss = epochss
    g_1 = []
    loss_total = []
    per_ = []
    per_ACC = []
    per_NMI = []
    per_Purity = []
    per_ARI = []
    per_Fscore = []
    per_Precision = []
    per_Recall = []
    for i in range (n_view):
        X[i] = X[i].float().to(device)
        X_ran[i] = X_ran[i].float().to(device)

    criterion = nn.BCEWithLogitsLoss()
    criterion1 = nn.MSELoss()
    optimizer_D={}
    scheduler_D={}
    if datasetname.startswith('MITIndoor'):
        for i in range(n_view):
            optimizer_D[i] = optim.Adam(D_net[i].parameters(), lr=lr, betas=(0.90, 0.92), eps=0.01, weight_decay=0.15)
            scheduler_D[i] = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_D[i], factor=0.3, patience=15, verbose=True, min_lr=1e-8)
        optimizer_G = optim.Adam(G_net.parameters(), lr=lr, betas=(0.90, 0.92), eps=0.01, weight_decay=0.15)
        scheduler_G = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_G, factor=0.3, patience=15, verbose=True, min_lr=1e-8)
    else:
        for i in range(n_view):
            optimizer_D[i] = optim.SGD(D_net[i].parameters(), lr=lr)
            scheduler_D[i] = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_D[i], factor=0.3, patience=15, verbose=True, min_lr=1e-8)
        optimizer_G = optim.SGD(G_net.parameters(), lr=lr)
        scheduler_G = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_G, factor=0.3, patience=15, verbose=True, min_lr=1e-8)

    for epoch in range(1, epochss + 1):
        print("epoch", epoch)
        loss_d = list()
        for i in range(n_view):
            for d_index in range(d_steps):
                D_net[i].zero_grad()
                # Step 1: Train D_real

                out_real_score = D_net[i](X[i])
                loss_d_real = criterion(out_real_score, torch.autograd.Variable(torch.ones_like(out_real_score)))

                # Step 2: Train D_fake
                out_fake = G_net(X, X_ran)[0][i].detach()
                out_fake_score = D_net[i](out_fake)

                optimizer_D[i].zero_grad()
                loss_d_fake = loss_d_real+ criterion(out_fake_score,
                                                        torch.autograd.Variable(torch.zeros_like(out_real_score)))
                loss_d_fake.backward()
                optimizer_D[i].step()
                scheduler_D[i].step(loss_d_fake)
                print("view:", i + 1, "d_index", d_index, "loss:{:.16f}".format(loss_d_fake))
                loss_d.append(loss_d_fake)

        for g_index in range(g_steps):
            # Step 3: Train G_fake
            G_net.zero_grad()

            out_fake, dis5, fus_out, out0, W, out1, encoding, decoding= G_net(X, X_ran)
            out_fake_score={}
            for i in range(n_view):
               out_fake_score[i] = D_net[i](out_fake[i])
            optimizer_G.zero_grad()

            loss_f_fake = sum(criterion(out_fake_score[i], torch.autograd.Variable(torch.ones_like(out_fake_score[i]))) for i in range(n_view)) +\
                          alpha * criterion1(dis5, out1) + beta * criterion1(fus_out, out1) + beta * criterion1(out0, out1) + \
                          sum(beta * criterion1(X[i], out1) + beta * criterion1(out_fake[i], decoding[i]) for i in range(n_view))

            loss_f_fake.backward()
            optimizer_G.step()
            scheduler_G.step(loss_f_fake)
            print("common g_index", g_index, "loss:{:.16f}".format(loss_f_fake))
            loss_g1 = loss_f_fake
            g_1.append(loss_g1.cpu().detach().numpy())

            if (epoch % 1 == 0):
                print("epoch", epoch)
                W = W.cpu().detach().numpy().reshape(n, -1)
                nc = np.unique(gnd).shape[0]
                [ACC, NMI, Purity, ARI, Fscore, Precision, Recall] = spectral_clustering(W, nc, gnd, repnum=10)
                per_.append([ACC, NMI, Purity, ARI, Fscore, Precision, Recall])
                per_ACC.append(ACC[0] * 100)
                per_NMI.append(NMI[0] * 100)
                per_Purity.append(Purity[0] * 100)
                per_ARI.append(ARI[0] * 100)
                per_Fscore.append(Fscore[0] * 100)
                per_Precision.append(Precision[0] * 100)
                per_Recall.append(Recall[0] * 100)
                # loss_total.append((loss_d1 + loss_d2 + loss_d3 + loss_d4 + loss_g1).cpu().detach().numpy())
    return W

def main(data, dataW, args):

    if args.fix_seed:
        seed = args.seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    device = torch.device('cpu' if args.device == 'cpu' else 'cuda:' + args.device)
    data_dir = args.data_path
    datasetW_dir = args.datasetW_dir

    use_leaky_relu = args.use_leaky_relu
    use_relu = args.use_relu
    use_relu_1 = args.use_relu_1
    use_leaky_relu_1 = True

    epochs_all = args.epoch
    lr_all = args.lr
    d_steps = args.d_steps
    g_steps = args.g_steps
    alpha = args.alpha
    beta = args.beta

    D0_channel_1 = args.D0_channel_1
    D0_channel_2 = args.D0_channel_2
    D0_kernel_1 = args.D0_kernel_1
    D0_stride_1 = args.D0_stride_1
    D0_padding_1 = args.D0_padding_1
    D0_kernel_2 = args.D0_kernel_2
    D0_stride_2 = args.D0_stride_2
    D0_padding_2 = args.D0_padding_2

    D1_channel_1 = args.D1_channel_1
    D1_channel_2 = args.D1_channel_2
    D1_kernel_1 = args.D1_kernel_1
    D1_stride_1 = args.D1_stride_1
    D1_padding_1 = args.D1_padding_1
    D1_kernel_2 = args.D1_kernel_2
    D1_stride_2 = args.D1_stride_2
    D1_padding_2 = args.D1_padding_2

    D2_channel_1 = args.D2_channel_1
    D2_channel_2 = args.D2_channel_2
    D2_kernel_1 = args.D2_kernel_1
    D2_stride_1 = args.D2_stride_1
    D2_padding_1 = args.D2_padding_1
    D2_kernel_2 = args.D2_kernel_2
    D2_stride_2 = args.D2_stride_2
    D2_padding_2 = args.D2_padding_2

    conv_channel_1 = args.conv_channel_1
    conv_channel_2 = args.conv_channel_2
    conv_kernel_1 = args.conv_kernel_1
    conv_stride_1 = args.conv_stride_1
    conv_padding_1 = args.conv_padding_1
    conv_kernel_2 = args.conv_kernel_2
    conv_stride_2 = args.conv_stride_2
    conv_padding_2 = args.conv_padding_2

    convv_channel_2 = args.convv_channel_2
    convv_kernel_1 = args.convv_kernel_1
    convv_stride_1 = args.convv_stride_1
    convv_padding_1 = args.convv_padding_1

    conv1_channel_1 = args.conv1_channel_1
    conv1_channel_2 = args.conv1_channel_2
    conv1_kernel_1 = args.conv1_kernel_1
    conv1_stride_1 = args.conv1_stride_1
    conv1_padding_1 = args.conv1_padding_1
    conv3_channel_2 = args.conv3_channel_2
    conv3_channel_1 = args.conv3_channel_1
    conv3_kernel_2 = args.conv3_kernel_2
    conv3_stride_2 = args.conv3_stride_2
    conv3_padding_2 = args.conv3_padding_2

    up1_channel_1 = args.up1_channel_1
    up1_channel_2 = args.up1_channel_2
    up1_kernel_1 = args.up1_kernel_1
    up1_stride_1 = args.up1_stride_1
    up1_padding_1 = args.up1_padding_1
    up3_channel_2 = args.up3_channel_2
    up3_channel_1 = args.up3_channel_1
    up3_kernel_2 = args.up3_kernel_2
    up3_stride_2 = args.up3_stride_2
    up3_padding_2 = args.up3_padding_2

    G_channel_1 = args.G_channel_1
    G_channel_2 = args.G_channel_2
    G_kernel_1 = args.G_kernel_1
    G_stride_1 = args.G_stride_1
    G_padding_1 = args.G_padding_1
    G_kernel_2 = args.G_kernel_2
    G_stride_2 = args.G_stride_2
    G_padding_2 = args.G_padding_2

    features, gnd = loadData(os.path.join(data_dir, data + ".mat"))
    gnd = gnd - 1
    nc = np.unique(gnd).shape[0]
    n = gnd.shape[0]

    if data.startswith('MNIST10k'):
        MNIST = ['MNIST10k', 'MNIST10k_Per0.1_nan', 'MNIST10k_Per0.3_nan', 'MNIST10k_Per0.5_nan', 'MNIST10k_Per0.7_nan']
        datesetW1 = {1: 'MNIST10k1GW', 2: 'MNIST10kPer01nan1GW', 3: 'MNIST10kPer03nan1GW', 4: 'MNIST10kPer05nan1GW',
                    5: 'MNIST10kPer07nan1GW'}
        datesetW2 = {1: 'MNIST10k2GW', 2: 'MNIST10kPer01nan2GW', 3: 'MNIST10kPer03nan2GW', 4: 'MNIST10kPer05nan2GW',
                     5: 'MNIST10kPer07nan2GW'}
        datesetW3 = {1: 'MNIST10k3GW', 2: 'MNIST10kPer01nan3GW', 3: 'MNIST10kPer03nan3GW', 4: 'MNIST10kPer05nan3GW',
                     5: 'MNIST10kPer07nan3GW'}
        jj = MNIST.index(data) + 1
        MG0 = loadMGW1(os.path.join(datasetW_dir, datesetW1[jj] + ".mat"))
        MG1 = loadMGW2(os.path.join(datasetW_dir, datesetW2[jj] + ".mat"))
        MG2 = loadMGW3(os.path.join(datasetW_dir, datesetW3[jj] + ".mat"))
        MG = [MG0, MG1, MG2]
    else:
        MG = loadMGW(os.path.join(datasetW_dir, dataW + ".mat"), dataW)
    si = n
    n_view = len(MG)
    W_ran0 = np.random.rand(si, si)
    MG_ran = list()
    inputs_W = list()
    inputs_W_ran = list()
    D = list()
    for i in range(n_view):
        MG_ran.append(GaussianNoise1(MG[i]))
        inputs_W.append(torch.from_numpy(MG[i]).view(1, 1, MG[i].shape[0], MG[i].shape[1]))
        inputs_W_ran.append(torch.from_numpy(MG_ran[i]).view(1, 1, MG[i].shape[0], MG[i].shape[1]))

    if data.startswith('ALOI') or data.startswith('MITIndoor'):

        D.append(Discriminator(D2_channel_1, D2_channel_2, D2_kernel_1, D2_stride_1, D2_padding_1, D2_kernel_2, D2_stride_2, D2_padding_2, use_leaky_relu_1, device))
        D.append(Discriminator(D0_channel_1, D0_channel_2, D0_kernel_1, D0_stride_1, D0_padding_1, D0_kernel_2, D0_stride_2, D0_padding_2, use_leaky_relu, device))
        D.append(Discriminator(D0_channel_1, D0_channel_2, D0_kernel_1, D0_stride_1, D0_padding_1, D0_kernel_2, D0_stride_2, D0_padding_2, use_leaky_relu, device))
        D.append(Discriminator(D1_channel_1, D1_channel_2, D1_kernel_1, D1_stride_1, D1_padding_1, D1_kernel_2, D1_stride_2, D1_padding_2, use_leaky_relu, device))
    elif data.startswith('scene15'):
        D.append(Discriminator(D2_channel_1, D2_channel_2, D2_kernel_1, D2_stride_1, D2_padding_1, D2_kernel_2, D2_stride_2, D2_padding_2, use_leaky_relu_1, device))
        D.append(Discriminator(D0_channel_1, D0_channel_2, D0_kernel_1, D0_stride_1, D0_padding_1, D0_kernel_2, D0_stride_2, D0_padding_2, use_leaky_relu, device))
        D.append(Discriminator(D0_channel_1, D0_channel_2, D0_kernel_1, D0_stride_1, D0_padding_1, D0_kernel_2, D0_stride_2, D0_padding_2, use_leaky_relu, device))
    else:
        for i in range(n_view):
            D.append(Discriminator(D0_channel_1, D0_channel_2, D0_kernel_1, D0_stride_1, D0_padding_1, D0_kernel_2, D0_stride_2, D0_padding_2, use_leaky_relu, device))

    G = Generator(epochs_all, lr_all, gnd, si, nc, n, n_view,
                  conv_channel_1, conv_channel_2, conv_kernel_1, conv_stride_1, conv_padding_1, conv_kernel_2,
                  conv_stride_2, conv_padding_2,
                  convv_channel_2, convv_kernel_1, convv_stride_1, convv_padding_1,
                  conv1_channel_1, conv1_channel_2, conv1_kernel_1, conv1_stride_1, conv1_padding_1,
                  conv3_channel_2, conv3_channel_1, conv3_kernel_2, conv3_stride_2, conv3_padding_2,
                  up1_channel_1, up1_channel_2, up1_kernel_1, up1_stride_1, up1_padding_1,
                  up3_channel_2, up3_channel_1, up3_kernel_2, up3_stride_2, up3_padding_2,
                  G_channel_1, G_channel_2, G_kernel_1, G_stride_1, G_padding_1, G_kernel_2, G_stride_2, G_padding_2,
                  use_relu, use_relu_1, device)

    for i in range(n_view):
        D[i].to(device)
        D[i].apply(weights_init)

    G.apply(weights_init).to(device)
    ################################################################################################
    print("Begin to train UMCGL network of " + data+ " dataset")
    W = train(D, G, inputs_W,
        inputs_W_ran, lr_all, epochs_all, d_steps, g_steps, n_view, n, gnd, alpha, beta, data, device)
    print("UMCGL performance on " + data + " dataset is:")
    [ACC, NMI, Purity, ARI, Fscore, Precision, Recall] = spectral_clustering(W, nc, gnd, repnum=10)
    print("-----------Ending-----------")
    ################################################################################################
