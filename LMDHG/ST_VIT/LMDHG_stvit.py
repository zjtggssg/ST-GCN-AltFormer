import numpy as np
import torch

import torch.optim as optim
import time
import argparse
import os
# from model.Net import *

from numpy.distutils.fcompiler import str2bool

from data_process.LMDHG_Hand import LMDHG_Hand_Dataset
from LMDHG.LMDHG_dataset import get_LMDHG_dataset
from model.AltFormer.ST_GCN_AltFormer import ST_GCN_AltFormer
from model.ST_Vit.ST_GCN_Trans import ST_GCN_Trans

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

# parser.add_argument('--data_cfg', type=int, default=0,
#                     help='0 for 14 class, 1 for 28')

parser.add_argument(
    '--nesterov', type=str2bool, default=True, help='use nesterov or not')

parser.add_argument('--dp_rate', type=float, default=0.2,
                    help='dropout rate')  # 1000


def init_data_loader():
    train_data, test_data = get_LMDHG_dataset()

    train_dataset = LMDHG_Hand_Dataset(train_data, use_data_aug=True, time_len=180)

    test_dataset = LMDHG_Hand_Dataset(test_data, use_data_aug=False, time_len=180)

    print("train data num: ", len(train_dataset))
    print("test data num: ", len(test_dataset))

    print("batch size:", args.batch_size)
    print("workers:", args.workers)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=False)

    val_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=False)

    return train_loader, val_loader


def init_model():

    class_num = 14
    output_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    l = [('labeling_mode', 'spatial')]
    graph_args = dict(l)
    address = 'graph.LMDHG'
    model = ST_GCN_Trans(channel=3, num_class=class_num, window_size=300, num_point=46, attention=True,
                      only_attention=True,
                      tcn_attention=False, all_layers=False, only_temporal_attention=True, attention_3=False,
                      relative=False,
                      double_channel=True,
                      drop_connect=True, concat_original=True, dv=0.25, dk=0.25, Nh=8, dim_block1=10, dim_block2=30,
                      dim_block3=75,
                      data_normalization=True, visualization=False, skip_conn=True, adjacency=False,
                      kernel_temporal=9, bn_flag=True, weight_matrix=2, more_channels=False, n=4, device=output_device,
                      graph=address,
                      graph_args=graph_args
                      )

    model = torch.nn.DataParallel(model).cuda()

    return model


def model_foreward(sample_batched, model, criterion):
    data = sample_batched["skeleton"].float()
    label = sample_batched["label"]
    label = label.type(torch.LongTensor)
    label = label.cuda()
    label = torch.autograd.Variable(label, requires_grad=False)

    score = model(data)

    loss = criterion(score, label)

    acc = get_acc(score, label)

    return score, loss, acc


def get_acc(score, labels):
    score = score.cpu().data.numpy()
    labels = labels.cpu().data.numpy()
    outputs = np.argmax(score, axis=1)
    return np.sum(outputs == labels) / float(labels.size)


# def val_get_pic(score, labels,val_cc):
#
#     labels_name = ['Catch', 'Catch With Two Hands','Draw C','Scroll','Draw Line','Rotate', 'Point To',
#                    'Point To With Two Hands','Rest', 'Shake','Shake Down',
#                    'Shake With Two Hands','Slice','Zoom'
#                    ]
#     drawconfusionmatrix = DrawConfusionMatrix(labels_name=labels_name,val_cc = val_cc)
#     score = score.cpu().data.numpy()
#     outputs = np.argmax(score, axis=1)
#     # print("shape:",outputs.shape)
#     # print("outputs:",outputs)
#     # print(type(labels))
#     labels = labels.type(torch.IntTensor)
#     drawconfusionmatrix.update(outputs, labels)
#     drawconfusionmatrix.drawMatrix()  # 根据所有predict和label，画出混淆矩阵
#     # confusion_mat = drawconfusionmatrix.getMatrix()
#     print("succeed!")


torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

if __name__ == "__main__":

    print("\nhyperparamter......")
    args = parser.parse_args()
    print(args)

    # fold for saving trained model...
    # change this path to the fold where you want to save your pre-trained model
    model_fold = "/data/zjt/LMDHG/ST_VIT/LMDHG_dp-{}_lr-{}/".format(args.dp_rate,
                                                            args.learning_rate)
    try:
        os.mkdir(model_fold)
    except:
        pass

    train_loader, val_loader = init_data_loader()

    # .........inital model
    print("\ninit model.............")
    model = init_model()

    model_solver = optim.SGD(
        model.parameters(),
        lr=0.01,
        momentum=0.9,
        nesterov=args.nesterov,
        weight_decay=0.0005)

    # ........set loss
    criterion = torch.nn.CrossEntropyLoss()

    #
    train_data_num = 1960
    test_data_num = 840
    iter_per_epoch = int(train_data_num / args.batch_size)

    # parameters recording training log
    max_acc = 0
    no_improve_epoch = 0
    n_iter = 0

    # ***********training#***********
    for epoch in range(args.epochs):
        print("\ntraining.............")
        model.train()
        start_time = time.time()
        train_acc = 0
        train_loss = 0
        for i, sample_batched in enumerate(train_loader):
            n_iter += 1
            # print("training i:",i)
            if i + 1 > iter_per_epoch:
                continue
            score, loss, acc = model_foreward(sample_batched, model, criterion)

            model.zero_grad()

            loss.backward()
            # clip_grad_norm_(model.parameters(), 0.1)
            model_solver.step()

            train_acc += acc
            train_loss += loss

            # print(i)

        train_acc /= float(i + 1)
        train_loss /= float(i + 1)

        print("*** SHREC  Epoch: [%2d] time: %4.4f, "
              "cls_loss: %.4f  train_ACC: %.6f ***"
              % (epoch + 1, time.time() - start_time,
                 train_loss.data, train_acc))
        start_time = time.time()

        # adjust_learning_rate(model_solver, epoch + 1, args)
        # print(print(model.module.encoder.gcn_network[0].edg_weight))

        # ***********evaluation***********
        with torch.no_grad():
            val_loss = 0
            acc_sum = 0
            model.eval()
            for i, sample_batched in enumerate(val_loader):
                # print("testing i:", i)
                label = sample_batched["label"]
                score, loss, acc = model_foreward(sample_batched, model, criterion)
                val_loss += loss

                if i == 0:
                    score_list = score
                    label_list = label
                else:
                    score_list = torch.cat((score_list, score), 0)
                    label_list = torch.cat((label_list, label), 0)

            val_loss = val_loss / float(i + 1)
            val_cc = get_acc(score_list, label_list)

            print("*** SHREC  Epoch: [%2d], "
                  "val_loss: %.6f,"
                  "val_ACC: %.6f ***"
                  % (epoch + 1, val_loss, val_cc))

            # save best model
            if val_cc > max_acc:
                max_acc = val_cc
                no_improve_epoch = 0

                val_cc = round(val_cc, 10)

                torch.save(model.state_dict(),
                           '{}/epoch_{}_acc_{}.pth'.format(model_fold, epoch + 1, val_cc))
                print("performance improve, saved the new model......best acc: {}".format(max_acc))
            else:
                no_improve_epoch += 1
                print("no_improve_epoch: {} best acc {}".format(no_improve_epoch, max_acc))

            if no_improve_epoch > args.patiences:
                print("stop training....")
                break