import itertools
import time

import torch
import argparse
import os
import torch.nn.functional as F
from matplotlib import pyplot as plt
from matplotlib.colors import BoundaryNorm
from sklearn.metrics import confusion_matrix
from torch import softmax
from visualizer import get_local
import pickle
import numpy as np
from tqdm import tqdm

get_local.activate() # 激活装饰器
from SHREC.Confusion import DrawConfusionMatrix
from model.AltFormer.ST_GCN_AltFormer import ST_GCN_AltFormer
from SHREC.dataset_node import *
from data_process.Hand_Dataset import Hand_Dataset


parser = argparse.ArgumentParser()

parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--cuda', default=True, help='enables cuda')
parser.add_argument('--data_cfg', type=int, default=0)

def init_data_loader(data_cfg, expand):
    train_data, test_data = split_train_test(data_cfg)
    all_data = test_data
    np.save('/data/zjt/HandGestureDataset_SHREC2017/gesture/test14.npy',all_data)
    # all_data = np.load('/data/zjt/HandGestureDataset_SHREC2017/gesture/test14.npy',allow_pickle=True)
    # all_data = [sample for sample in all_data if sample["label"] == 2]

    all_dataset = Hand_Dataset(all_data, use_data_aug=False, time_len=180,expand = expand)

    print("test data num: ", len(all_dataset))

    print("batch size:", args.batch_size)

    all_loader = torch.utils.data.DataLoader(
        all_dataset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=8, pin_memory=False)


    return all_loader

def init_model_st_ts(data_cfg):
    if data_cfg == 0:
        class_num = 14
    elif data_cfg == 1:
        class_num = 28

    output_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    l = [('labeling_mode', 'spatial')]
    graph_args = dict(l)
    address = 'graph.SHRE'
    model = ST_GCN_AltFormer(channel=3, num_class=class_num, window_size=300, num_point=22, attention=True,
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
    model_path = '/home/zjt/Desktop/ST-GCN-AltFormer/weight/0.975.pth'
    state_dict = torch.load(model_path)

    model.load_state_dict(state_dict)

    model.eval()
    return model

def init_model_st(data_cfg):
    if data_cfg == 0:
        class_num = 14
    elif data_cfg == 1:
        class_num = 28

    output_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    l = [('labeling_mode', 'spatial')]
    graph_args = dict(l)
    address = 'graph.SHRE'
    model = ST_GCN_AltFormer(channel=3, backbone_in_c=128, num_frame=180, num_joints=22, num_class=class_num,
                             style='ST',
                             graph=address,
                             graph_args=graph_args
                             )

    model = torch.nn.DataParallel(model).cuda()
    model_path = '/data/zjt/AltFormer_code/SHREC/weight/st_14.pth'
    # '/data/zjt/AltFormer_code/weight/st_14.pth'
    state_dict = torch.load(model_path)

    model.load_state_dict(state_dict)

    model.eval()
    return model


def init_model_ts(data_cfg):
    if data_cfg == 0:
        class_num = 14
    elif data_cfg == 1:
        class_num = 28

    output_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    l = [('labeling_mode', 'spatial')]
    graph_args = dict(l)
    address = 'graph.SHRE'
    model = ST_GCN_AltFormer(channel=3, backbone_in_c=128, num_frame=180, num_joints=22, num_class=class_num,
                             style='TS',
                             graph=address,
                             graph_args=graph_args
                             )

    model = torch.nn.DataParallel(model).cuda()
    model_path = '/data/zjt/AltFormer_code/SHREC/weight/ts_14.pth'
    state_dict = torch.load(model_path)

    model.load_state_dict(state_dict)

    model.eval()
    return model


def model_forward(sample_batched, model):
    data = sample_batched["skeleton"].float()
    # data = data.unsqueeze(0)
    # print(data.shape)
    score = model(data)

    return score


def get_acc(score, labels):

    outputs = np.argmax(score, axis=1)
    labels = labels.flatten()

    return np.sum(outputs == labels) / float(labels.size)

def get_wrong(score, labels, pred):
    outputs = np.argmax(score, axis=1)
    labels = labels.flatten()
    start = np.where(labels == pred)[0][0]
    print(start)
    mismatched_indices = np.where(outputs != labels)[0]
    print(mismatched_indices,len(mismatched_indices))
    wrong_list = []
    for mis in mismatched_indices:
        if labels[mis] == pred:
            x = mis - start
            y = outputs[mis]
            wrong_list.append((x,y))

    print("worng",wrong_list)


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion Matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.figure(figsize=(19.2, 10.8))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def plot_confusion(predicted_labels,true_labels,style):
    predicted_labels = np.argmax(predicted_labels, axis=1)
    cm = confusion_matrix(true_labels, predicted_labels)
    # # class_names = ['Grab(1)', 'Grab(2)','Tap(1)','Tap(2)','Expand(1)','Expand(2)', 'Pinch(1)', 'Pinch(2)','Rotation CW(1)', 'Rotation CW(2)','Rotation CCW(1)', 'Rotation CCW(2)','Swipe Right(1)','Swipe Right(2)','Swipe Left(1)','Swipe Left(2)',
    # #                'Swipe Up(1)', 'Swipe Up(2)','Swipe Down(1)', 'Swipe Down(2)','Swipe X(1)','Swipe X(2)', 'Swipe +(1)','Swipe +(2)','Swipe V(1)', 'Swipe V(2)', 'Shake(1)','Shake(2)'] # 替换为你的类别标签
    class_names = ['Grab', 'Tap', 'Expand', 'Pinch', 'Rotation CW', 'Rotation CCW', 'Swipe Right', 'Swipe Left',
                   'Swipe Up',
                   'Swipe Down', 'Swipe X', 'Swipe +', 'Swipe V', 'Shake']
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    np.savetxt("/data/zjt/handgesture/data.txt", cm_normalized, fmt="%.2f")

    plot_confusion_matrix(cm, classes=class_names, normalize=True, title='Normalized Confusion Matrix')
    plt.xticks(rotation=45, ha="right")
    plt.savefig("/data/zjt/handgesture/pic/{}_LMDHG_confusion_matrix.png".format(style))

def emsemble_acc(st_ts,st,ts,label,st_motion = None,st_bone = None,ts_motion = None , ts_bone = None,st_ts_motion = None , st_ts_bone = None):

    a = 0.8
    combiend =    a * st + (1-a) * ts
    # combiend = motion +  ts + bone
    acc = get_acc(combiend,label)
    # plot_confusion(combiend,true_labels,style = 'st_new')
    # val_get_pic(combiend,label,acc,style = 'st_ts')
    # plot_confusion(combiend,true_labels,style = 'st_ts_sanliu')
    get_wrong(combiend,label,pred = 7)
    return acc



if __name__ == '__main__':
    args = parser.parse_args()
    print(args)


    # 初始化一个用于投票的列表
    st_list = []
    ts_list = []
    motion_list = []
    bone_list = []

    # 迭代测试数据集并获取每个模型的预测结果
    pd = 5

    if pd == 0:
        model_st = init_model_st_ts(args.data_cfg)
        test_loader = init_data_loader(args.data_cfg, expand=0)
        start_time = time.time()
        for data in test_loader:
            with torch.no_grad():
                pred= model_forward(data, model_st)
                pred = pred.cpu().numpy()
                st_list.append(pred)
        end_time = time.time()  # End measuring inference time
        inference_time = end_time - start_time
        print(f"Inference time: {inference_time:.2f} seconds")

        st_list = np.concatenate(st_list, axis=0)
        print(st_list.shape)
        st_list = st_list.reshape(840, 14)
        del model_st, pred
        print("st succeed!")
        np.save('/data/zjt/HandGestureDataset_SHREC2017/gesture/combined/st_ts.npy', st_list)

    elif pd == 1:
        model_st = init_model_st(args.data_cfg)
        test_loader = init_data_loader(args.data_cfg,expand = 0)
        start_time = time.time()
        for data in test_loader:
            with torch.no_grad():
                pred= model_forward(data, model_st)
                pred = pred.cpu().numpy()

                st_list.append(pred)
        end_time = time.time()  # End measuring inference time
        inference_time = end_time - start_time
        print(f"Inference time: {inference_time:.2f} seconds")

        st_list = np.concatenate(st_list, axis=0)
        print(st_list.shape)
        st_list = st_list.reshape(840, 14)
        del model_st, pred
        print("st succeed!")
        np.save('/data/zjt/HandGestureDataset_SHREC2017/gesture/combined/st_14.npy',st_list)


    elif pd == 2:
        model_ts = init_model_ts(args.data_cfg)
        test_loader = init_data_loader(args.data_cfg,expand = 0)
        for data in test_loader:
            with torch.no_grad():
                pred2= model_forward(data, model_ts)
                pred2 = pred2.cpu().numpy()

                ts_list.append(pred2)
        ts_list = np.concatenate(ts_list, axis=0)
        ts_list = ts_list.reshape(840, 14)
        del model_ts, pred2
        print("ts succeed!")
        np.save('/data/zjt/HandGestureDataset_SHREC2017/gesture/combined/ts_14.npy',ts_list)

    #motion
    # elif pd == 3:
    #     model_motion = init_model_motion(args.data_cfg)
    #
    #     test_loader = init_data_loader(args.data_cfg,expand = 1)
    #     for data in test_loader:
    #         with torch.no_grad():
    #             pred2, st, ts1 = model_forward(data, model_motion)
    #             pred2 = pred2.cpu().numpy()
    #             motion_list.append(pred2)
    #     motion_list = np.concatenate(motion_list, axis=0)
    #     motion_list = motion_list.reshape(840, 14)
    #     del model_motion, pred2, st, ts1
    #     print("motion succeed!")
    #     np.save('/data/zjt/HandGestureDataset_SHREC2017/gesture/combined/st_ts_motion.npy',motion_list)
    #
    # elif pd == 4:
    #     model_bone = init_model_bone(args.data_cfg)
    #     test_loader = init_data_loader(args.data_cfg,expand = 2)
    #     for data in test_loader:
    #         with torch.no_grad():
    #             pred2, st, ts1 = model_forward(data, model_bone)
    #             pred2 = pred2.cpu().numpy()
    #             bone_list.append(pred2)
    #     bone_list = np.concatenate(bone_list, axis=0)
    #     bone_list= bone_list.reshape(840, 14)
    #     del model_bone, pred2, st, ts1
    #     print("bone succeed!")
    #     np.save('/data/zjt/HandGestureDataset_SHREC2017/gesture/combined/st_ts_bone.npy',bone_list)


    else:
        st_list = np.load('/data/zjt/HandGestureDataset_SHREC2017/gesture/combined/st_14.npy')
        print(st_list.shape)
        ts_list = np.load('/data/zjt/HandGestureDataset_SHREC2017/gesture/combined/ts_14.npy')
        print(ts_list.shape)
        st_motion_list = np.load('/data/zjt/HandGestureDataset_SHREC2017/gesture/combined/st_motion.npy')

        st_bone_list = np.load('/data/zjt/HandGestureDataset_SHREC2017/gesture/combined/st_bone.npy')

        ts_motion_list = np.load('/data/zjt/HandGestureDataset_SHREC2017/gesture/combined/ts_motion.npy')

        ts_bone_list = np.load('/data/zjt/HandGestureDataset_SHREC2017/gesture/combined/ts_bone.npy')

        st_ts_list = np.load('/data/zjt/HandGestureDataset_SHREC2017/gesture/combined/st_ts.npy')

        st_ts_motion_list = np.load('/data/zjt/HandGestureDataset_SHREC2017/gesture/combined/st_ts_motion.npy')

        st_ts_bone_list = np.load('/data/zjt/HandGestureDataset_SHREC2017/gesture/combined/st_ts_bone.npy')

        # 获取真实标签
        true_labels = []
        test_loader = init_data_loader(args.data_cfg,expand = 0)
        for data in test_loader:
                label = data['label']
                true_labels.append(label)
        true_labels = np.concatenate(true_labels, axis=0)
        true_labels = true_labels.reshape(840, 1)

        acc_st_ts = get_acc(st_ts_list, true_labels)
        acc_st_ts_motion = get_acc(st_ts_motion_list, true_labels)
        acc_st_ts_bone = get_acc(st_ts_bone_list, true_labels)
        acc_st = get_acc(st_list, true_labels)
        acc_ts = get_acc(ts_list, true_labels)
        acc_st_motion = get_acc(st_motion_list, true_labels)
        acc_st_bone = get_acc(st_bone_list, true_labels)
        acc_ts_motion = get_acc(ts_motion_list, true_labels)
        acc_ts_bone = get_acc(ts_bone_list, true_labels)
        print("st_ts acc:", acc_st_ts)
        print("st_ts_motion acc:", acc_st_ts_motion)
        print("st_ts_bone acc:", acc_st_ts_bone)
        print("st acc:", acc_st)
        print("ts acc:", acc_ts)
        print("st_motion acc:", acc_st_motion)
        print("st_bone acc:", acc_st_bone)
        print("ts_motion acc:", acc_ts_motion)
        print("ts_bone acc:", acc_ts_bone)

        # val_get_pic(st_list, true_labels, acc_st,style = 'st')
        # val_get_pic(ts_list, true_labels, acc_ts,style = 'ts')
        # plot_confusion(st_list, true_labels, style='st')
        # plot_confusion(ts_list, true_labels, style='ts')
        # 计算集成结果的准确率
        accuracy = emsemble_acc(st_ts_list,st_list, ts_list, true_labels,st_motion_list,st_bone_list,ts_motion_list,ts_bone_list,st_ts_motion_list,st_ts_bone_list)

        print("Ensemble acc:", accuracy)




