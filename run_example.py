import sys
from main_example import main
import argparse
from utils.config import load_config
if __name__ == '__main__':

    ## Parameter setting
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_path", type=str, default="./_multiview datasets", help="Path of datasets.")
    parser.add_argument("--datasetW_dir", type=str, default="./datasetW", help="Path of W.")

    parser.add_argument("--fix_seed", action='store_true', default=True, help="")
    parser.add_argument("--seed", type=int, default=2020, help="Random seed, default is 2020.")

    parser.add_argument("--d_steps", type=int, default=2, help="")
    parser.add_argument("--g_steps", type=int, default=1, help="")
    parser.add_argument("--alpha", type=int, default=1, help="")
    parser.add_argument("--beta", type=int, default=1, help="")

    parser.add_argument("--D0_channel_1", type=int, default=1, help="")
    parser.add_argument("--D0_channel_2", type=int, default=2, help="")
    parser.add_argument("--D0_kernel_1", type=int, default=7, help="")
    parser.add_argument("--D0_stride_1", type=int, default=1, help="")
    parser.add_argument("--D0_padding_1", type=int, default=3, help="")
    parser.add_argument("--D0_kernel_2", type=int, default=7, help="")
    parser.add_argument("--D0_stride_2", type=int, default=1, help="")
    parser.add_argument("--D0_padding_2", type=int, default=3, help="")

    parser.add_argument("--D1_channel_1", type=int, default=1, help="")
    parser.add_argument("--D1_channel_2", type=int, default=4, help="")
    parser.add_argument("--D1_kernel_1", type=int, default=3, help="")
    parser.add_argument("--D1_stride_1", type=int, default=1, help="")
    parser.add_argument("--D1_padding_1", type=int, default=1, help="")
    parser.add_argument("--D1_kernel_2", type=int, default=3, help="")
    parser.add_argument("--D1_stride_2", type=int, default=1, help="")
    parser.add_argument("--D1_padding_2", type=int, default=1, help="")

    parser.add_argument("--D2_channel_1", type=int, default=1, help="")
    parser.add_argument("--D2_channel_2", type=int, default=8, help="")
    parser.add_argument("--D2_kernel_1", type=int, default=3, help="")
    parser.add_argument("--D2_stride_1", type=int, default=1, help="")
    parser.add_argument("--D2_padding_1", type=int, default=1, help="")
    parser.add_argument("--D2_kernel_2", type=int, default=1, help="")
    parser.add_argument("--D2_stride_2", type=int, default=1, help="")
    parser.add_argument("--D2_padding_2", type=int, default=0, help="")

    parser.add_argument("--conv_channel_1", type=int, default=1, help="")
    parser.add_argument("--conv_stride_1", type=int, default=1, help="")
    parser.add_argument("--conv_stride_2", type=int, default=1, help="")

    parser.add_argument("--convv_channel_2", type=int, default=1, help="")
    parser.add_argument("--convv_stride_1", type=int, default=1, help="")

    parser.add_argument("--conv1_channel_1", type=int, default=1, help="")
    parser.add_argument("--conv1_stride_1", type=int, default=1, help="")
    parser.add_argument("--conv3_channel_1", type=int, default=1, help="")
    parser.add_argument("--conv3_stride_2", type=int, default=1, help="")


    parser.add_argument("--up1_channel_1", type=int, default=1, help="")
    parser.add_argument("--up1_kernel_1", type=int, default=3, help="")
    parser.add_argument("--up1_stride_1", type=int, default=1, help="")
    parser.add_argument("--up1_padding_1", type=int, default=1, help="")
    parser.add_argument("--up3_channel_1", type=int, default=1, help="")
    parser.add_argument("--up3_kernel_2", type=int, default=3, help="")
    parser.add_argument("--up3_stride_2", type=int, default=1, help="")
    parser.add_argument("--up3_padding_2", type=int, default=1, help="")

    parser.add_argument("--G_channel_1", type=int, default=1, help="")
    parser.add_argument("--G_stride_1", type=int, default=1, help="")
    parser.add_argument("--G_stride_2", type=int, default=1, help="")

    args = parser.parse_args()

    dataset = { 1: 'ALOI', 2: 'ALOI_Per0.1_nan', 3: 'ALOI_Per0.3_nan', 4: 'ALOI_Per0.5_nan', 5: 'ALOI_Per0.7_nan',
                6: 'Caltech101-7', 7: 'Caltech101-7_Per0.1_nan', 8: 'Caltech101-7_Per0.3_nan', 9: 'Caltech101-7_Per0.5_nan', 10: 'Caltech101-7_Per0.7_nan',
                11: 'Caltech101-20', 12: 'Caltech101-20_Per0.1_nan', 13: 'Caltech101-20_Per0.3_nan', 14: 'Caltech101-20_Per0.5_nan', 15: 'Caltech101-20_Per0.7_nan',
                16: 'HW', 17: 'HW_Per0.1_nan', 18: 'HW_Per0.3_nan', 19: 'HW_Per0.5_nan', 20: 'HW_Per0.7_nan',
                21: 'MITIndoor', 22: 'MITIndoor_Per0.1_nan', 23: 'MITIndoor_Per0.3_nan', 24: 'MITIndoor_Per0.5_nan', 25: 'MITIndoor_Per0.7_nan',
                26: 'MNIST10k', 27: 'MNIST10k_Per0.1_nan', 28: 'MNIST10k_Per0.3_nan', 29: 'MNIST10k_Per0.5_nan', 30: 'MNIST10k_Per0.7_nan',
                31: 'NUS-WIDE', 32: 'NUS-WIDE_Per0.1_nan', 33: 'NUS-WIDE_Per0.3_nan', 34: 'NUS-WIDE_Per0.5_nan', 35: 'NUS-WIDE_Per0.7_nan',
                36: 'scene15', 37: 'scene15_Per0.1_nan', 38: 'scene15_Per0.3_nan', 39: 'scene15_Per0.5_nan', 40: 'scene15_Per0.7_nan'}

    datesetW = { 1: 'ALOIMGW', 2: 'ALOIPer01nanMGW', 3: 'ALOIPer03nanMGW', 4: 'ALOIPer05nanMGW', 5: 'ALOIPer07nanMGW',
                 6: 'Caltech1017MGW', 7: 'Caltech1017Per01nanMGW', 8: 'Caltech1017Per03nanMGW', 9: 'Caltech1017Per05nanMGW', 10: 'Caltech1017Per07nanMGW',
                 11: 'Caltech10120MGW', 12: 'Caltech10120Per01nanMGW', 13: 'Caltech10120Per03nanMGW', 14: 'Caltech10120Per05nanMGW', 15: 'Caltech10120Per07nanMGW',
                 16: 'HWMGW', 17: 'HWPer01nanMGW', 18: 'HWPer03nanMGW', 19: 'HWPer05nanMGW', 20: 'HWPer07nanMGW',
                 21: 'MITIndoorMGW', 22: 'MITIndoorPer01nanMGW', 23: 'MITIndoorPer03nanMGW', 24: 'MITIndoorPer05nanMGW', 25: 'MITIndoorPer07nanMGW',
                 26: 'MNIST10k1GW', 27: 'MNIST10kPer01nan1GW', 28: 'MNIST10kPer03nan1GW', 29: 'MNIST10kPer05nan1GW', 30: 'MNIST10kPer07nan1GW',
                 31: 'NUSWIDEMGW', 32: 'NUSWIDEPer01nanMGW', 33: 'NUSWIDEPer03nanMGW', 34: 'NUSWIDEPer05nanMGW', 35: 'NUSWIDEPer07nanMGW',
                 36: 'scene15MGW', 37: 'scene15Per01nanMGW', 38: 'scene15Per03nanMGW', 39: 'scene15Per05nanMGW', 40: 'scene15Per07nanMGW'}

    ALOI_select_dataset = [1, 2, 3, 4, 5]
    Caltech1017_select_dataset = [6, 7, 8, 9, 10]
    Caltech10120_select_dataset = [11, 12, 13, 14, 15]
    HW_select_dataset = [16, 17, 18, 19, 20]
    MITIndoor_select_dataset = [21, 22, 23, 24, 25]
    MNIST10k_select_dataset = [26, 27, 28, 29, 30]
    NUSWIDE_select_dataset = [31, 32, 33, 34, 35]
    scene15_select_dataset = [36, 37, 38, 39, 40]
    random_select_dataset = [16]

    # all_select_dataset = [ALOI_select_dataset, Caltech1017_select_dataset, Caltech10120_select_dataset, HW_select_dataset,
    #                       MITIndoor_select_dataset, MNIST10k_select_dataset, NUSWIDE_select_dataset, scene15_select_dataset]
    # all_select_dataset = [Caltech1017_select_dataset, Caltech10120_select_dataset, HW_select_dataset, MNIST10k_select_dataset, NUSWIDE_select_dataset]
    # all_select_dataset = [Caltech1017_select_dataset, Caltech10120_select_dataset, HW_select_dataset, NUSWIDE_select_dataset]
    all_select_dataset = [HW_select_dataset]

    for ii in range(len(all_select_dataset)):
        first_dataset_index = all_select_dataset[ii][0]
        for i in all_select_dataset[ii]:

            config = load_config('./config/' + dataset[first_dataset_index])

            args.use_leaky_relu = config['D_use_leaky_relu']
            args.use_relu = config['G_use_relu']
            args.use_relu_1 = config['MNIST_use_relu_1']
            args.device = str(config['device'])

            args.epoch = config['epoch']
            args.lr = config['lr']

            args.conv_channel_2 = config['conv_channel_2']
            args.conv_kernel_1 = config['conv_kernel_1']
            args.conv_padding_1 = config['conv_padding_1']
            args.conv_kernel_2 = config['conv_kernel_2']
            args.conv_padding_2 = config['conv_padding_2']

            args.convv_kernel_1 = config['convv_kernel_1']
            args.convv_padding_1 = config['convv_padding_1']

            args.conv1_channel_2 = config['conv1_channel_2']
            args.conv1_kernel_1 = config['conv1_kernel_1']
            args.conv1_padding_1 = config['conv1_padding_1']
            args.conv3_channel_2 = config['conv3_channel_2']
            args.conv3_kernel_2 = config['conv3_kernel_2']
            args.conv3_padding_2 = config['conv3_padding_2']

            args.up1_channel_2 = config['up1_channel_2']
            args.up3_channel_2 = config['up3_channel_2']

            args.G_channel_2 = config['G_channel_2']
            args.G_kernel_1 = config['G_kernel_1']
            args.G_padding_1 = config['G_padding_1']
            args.G_kernel_2 = config['G_kernel_2']
            args.G_padding_2 = config['G_padding_2']

            main(dataset[i], datesetW[i], args)
