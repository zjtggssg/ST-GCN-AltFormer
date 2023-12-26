import numpy as np
import torch

import torch.optim as optim
import time
import argparse
import os
# from model.Net import *

from numpy.distutils.fcompiler import str2bool

from model.ST_GCN_AltFormer import ST_GCN_AltFormer



from data_process.Hand_Dataset import Hand_Dataset
from mutil.dataset_motion import split_train_sport_test

parser = argparse.ArgumentParser()

parser.add_argument("-b", "--batch_size", type=int, default=32)  # 16
parser.add_argument("-lr", "--learning_rate", type=float, default=1e-3)
parser.add_argument('--cuda', default=True, help='enables cuda')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')
parser.add_argument('--epochs', default=10000, type=int, metavar='N',
                    help='number of total epochs to run')  # 1000

parser.add_argument('--patiences', default=1000, type=int,
                    help='number of epochs to tolerate the no improvement of val_loss')  # 1000

parser.add_argument('--data_cfg', type=int, default=0,
                    help='0 for 14 class, 1 for 28')

parser.add_argument(
    '--nesterov', type=str2bool, default=True, help='use nesterov or not')

parser.add_argument('--dp_rate', type=float, default=0.2,
                    help='dropout rate')  # 1000


def init_data_loader(data_cfg):
    train_data, test_data = split_train_sport_test(data_cfg)
    all_dataset = test_data + train_data


    for i in range(1,15):
        all_data = [sample for sample in all_dataset if sample["label"] == i]

        sum_list = []
        num_sum = 0
        for dataset in all_data:
            total_sum = 0
            data = np.array(dataset["skeleton2"])
            for row in data:
                for element in row:
                    total_sum += abs(element)
            sum_list.append(total_sum)



        print(type(sum_list))
        print("sum_{}".format(i), sum(sum_list))
        print("num_sum_{}".format(i), sum(sum(sum_list)))

    print("batch size:", args.batch_size)
    print("workers:", args.workers)


    return train_data, test_data


if __name__ == "__main__":
    args = parser.parse_args()
    print(args)

    train_loader, val_loader = init_data_loader(args.data_cfg)

