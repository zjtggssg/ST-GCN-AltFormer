import itertools
import time

import torch
import argparse
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np

from LMDHG.LMDHG_Hand import LMDHG_Hand_Dataset
from LMDHG.LMDHG_dataset import get_LMDHG_dataset

from model.ST_GCN_AltFormer import ST_GCN_AltFormer

parser = argparse.ArgumentParser()

parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--cuda', default=True, help='enables cuda')
parser.add_argument('--data_cfg', type=int, default=0)
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')

def init_data_loader():
    # train_data, test_data = get_LMDHG_dataset()
    test_data = np.load('/data/zjt/LMDHG/npy/test.npy',allow_pickle=True)
    test_dataset = LMDHG_Hand_Dataset(test_data, use_data_aug=False, time_len=180)

    print("test data num: ", len(test_dataset))


    val_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=False)

    return val_loader



def init_model_st():
    class_num = 14
    output_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    l = [('labeling_mode', 'spatial')]
    graph_args = dict(l)
    address = 'graph.LMDHG'
    model = ST_GCN_AltFormer(channel=3, num_class=class_num, window_size=300, num_point=46, attention=True,
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
    model_path = '/home/zjt/Desktop/ST-GCN-AltFormer/LMDHG/weight/ST_TS/0.9774.pth'
    state_dict = torch.load(model_path)

    model.load_state_dict(state_dict)

    model.eval()
    return model


def init_model_ts():
    class_num = 14

    output_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    l = [('labeling_mode', 'spatial')]
    graph_args = dict(l)
    address = 'graph.LMDHG'
    model = ST_GCN_AltFormer(channel=3, num_class=class_num, window_size=300, num_point=46, attention=True,
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
    model_path = '/home/zjt/Desktop/ST-GCN-AltFormer/LMDHG/weight/ST_TS/ts.pth'
    state_dict = torch.load(model_path)

    model.load_state_dict(state_dict)

    model.eval()
    return model

# def init_model_motion(data_cfg):
#     if data_cfg == 0:
#         class_num = 14
#     elif data_cfg == 1:
#         class_num = 28
#
#     output_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     l = [('labeling_mode', 'spatial')]
#     graph_args = dict(l)
#     address = 'graph.SHRE'
#     model = ST_GCN_AltFormer(channel=3, num_class=class_num, window_size=300, num_point=22, attention=True,
#                              only_attention=True,
#                              tcn_attention=False, all_layers=False, only_temporal_attention=True, attention_3=False,
#                              relative=False,
#                              double_channel=True,
#                              drop_connect=True, concat_original=True, dv=0.25, dk=0.25, Nh=8, dim_block1=10,
#                              dim_block2=30,
#                              dim_block3=75,
#                              data_normalization=True, visualization=False, skip_conn=True, adjacency=False,
#                              kernel_temporal=9, bn_flag=True, weight_matrix=2, more_channels=False, n=4,
#                              device=output_device,
#                              graph=address,
#                              graph_args=graph_args
#                              )
#
#     model = torch.nn.DataParallel(model).cuda()
#     model_path = '/home/zjt/Desktop/ST-GCN-AltFormer/weight/0.9393.pth'
#     state_dict = torch.load(model_path)
#
#     model.load_state_dict(state_dict)
#
#     model.eval()
#     return model
#
# def init_model_bone(data_cfg):
#     if data_cfg == 0:
#         class_num = 14
#     elif data_cfg == 1:
#         class_num = 28
#
#     output_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     l = [('labeling_mode', 'spatial')]
#     graph_args = dict(l)
#     address = 'graph.SHRE'
#     model = ST_GCN_AltFormer(channel=3, num_class=class_num, window_size=300, num_point=22, attention=True,
#                              only_attention=True,
#                              tcn_attention=False, all_layers=False, only_temporal_attention=True, attention_3=False,
#                              relative=False,
#                              double_channel=True,
#                              drop_connect=True, concat_original=True, dv=0.25, dk=0.25, Nh=8, dim_block1=10,
#                              dim_block2=30,
#                              dim_block3=75,
#                              data_normalization=True, visualization=False, skip_conn=True, adjacency=False,
#                              kernel_temporal=9, bn_flag=True, weight_matrix=2, more_channels=False, n=4,
#                              device=output_device,
#                              graph=address,
#                              graph_args=graph_args
#                              )
#
#     model = torch.nn.DataParallel(model).cuda()
#     model_path = '/home/zjt/Desktop/ST-GCN-AltFormer/weight/0.8608.pth'
#     state_dict = torch.load(model_path)
#
#     model.load_state_dict(state_dict)
#
#     model.eval()
#     return model

def model_forward(sample_batched, model):
    data = sample_batched["skeleton"].float()
    # data = data.unsqueeze(0)
    # print(data.shape)
    score,st,ts = model(data)

    return score,st,ts


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


def findwrong(index):
    test_data = np.load('/data/zjt/LMDHG/npy/test.npy', allow_pickle=True)
    wrong = test_data[index]['skeleton']
    print(wrong)


# def val_get_pic(score, labels,val_cc,style):
#     labels_name = ['Grab', 'Tap', 'Expand', 'Pinch', 'Rotation CW', 'Rotation CCW', 'Swipe Right','Swipe Left','Swipe Up','Swipe Down','Swipe X','Swipe +','Swipe V','Shake']
#     # labels_name = ['Grab(1)', 'Grab(2)','Tap(1)','Tap(2)','Expand(1)','Expand(2)', 'Pinch(1)', 'Pinch(2)','Rotation CW(1)', 'Rotation CW(2)','Rotation CCW(1)', 'Rotation CCW(2)','Swipe Right(1)','Swipe Right(2)','Swipe Left(1)','Swipe Left(2)',
#     #                'Swipe Up(1)', 'Swipe Up(2)','Swipe Down(1)', 'Swipe Down(2)','Swipe X(1)','Swipe X(2)', 'Swipe +(1)','Swipe +(2)','Swipe V(1)', 'Swipe V(2)', 'Shake(1)','Shake(2)']
#     drawconfusionmatrix = DrawConfusionMatrix(labels_name=labels_name,style=style,val_cc = val_cc)
#     outputs = np.argmax(score, axis=1)
#     drawconfusionmatrix.update(outputs, labels)
#     drawconfusionmatrix.drawMatrix()  # 根据所有predict和label，画出混淆矩阵
#     # confusion_mat = drawconfusionmatrix.getMatrix()
#
#     print("succeed!")

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
    class_names = ['Catch','Catch with two hands','Draw C', 'Scroll','Draw Line', 'Rotate','Point to',  'Point to with two hands','Rest','Shake','Shake down','Shake with two hands',
     'Slice' ,'Zoom']
    plot_confusion_matrix(cm, classes=class_names, normalize=True, title='Normalized Confusion Matrix')
    plt.xticks(rotation=45, ha="right")
    plt.savefig("/data/zjt/handgesture/pic/{}_LMDHG_confusion_matrix.png".format(style))


def emsemble_acc(st,ts,label,motion = None,bone = None):

    a = 0.0
    combiend =  st + ts
    # combiend = ts
    acc = get_acc(combiend ,label)
    plot_confusion(combiend,true_labels,style = 'ST_TS')
    # val_get_pic(combiend,label,acc,style = 'st_ts')
    get_wrong(combiend,label,pred = 13)
    # findwrong(142)
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
    pd = 3

    if pd == 1:
        model_st = init_model_st()
        test_loader = init_data_loader()
        start_time = time.time()
        for data in test_loader:
            with torch.no_grad():
                pred, st1, ts = model_forward(data, model_st)
                pred = pred.cpu().numpy()
                st1 = st1.cpu().numpy()
                st_list.append(st1)
        end_time = time.time()  # End measuring inference time
        inference_time = end_time - start_time
        print(f"Inference time: {inference_time:.2f} seconds")

        st_list = np.concatenate(st_list, axis=0)
        print(st_list.shape)
        st_list = st_list.reshape(355, 14)
        del model_st, pred, st1, ts
        print("st succeed!")
        np.save('/data/zjt/LMDHG/combined/st.npy',st_list)


    elif pd == 2:
        model_ts = init_model_ts()
        test_loader = init_data_loader()
        for data in test_loader:
            with torch.no_grad():
                pred2, st, ts1 = model_forward(data, model_ts)
                pred2 = pred2.cpu().numpy()
                ts_list.append(pred2)
        ts_list = np.concatenate(ts_list, axis=0)
        ts_list = ts_list.reshape(355, 14)
        del model_ts, pred2, st, ts1
        print("ts succeed!")
        np.save('/data/zjt/LMDHG/combined/ts.npy',ts_list)

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
    #     np.save('/data/zjt/HandGestureDataset_SHREC2017/gesture/combined/ts_motion.npy',motion_list)
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
    #     np.save('/data/zjt/HandGestureDataset_SHREC2017/gesture/combined/ts_bone.npy',bone_list)


    else:
        st_list = np.load('/data/zjt/LMDHG/combined/st.npy')
        print(st_list.shape)
        ts_list = np.load('/data/zjt/LMDHG/combined/ts.npy')
        print(ts_list.shape)
        # motion_list = np.load('/data/zjt/HandGestureDataset_SHREC2017/gesture/combined/motion.npy')
        # print(motion_list.shape)
        # bone_list = np.load('/data/zjt/HandGestureDataset_SHREC2017/gesture/combined/bone.npy')
        # print(bone_list.shape)
        # 获取真实标签
        true_labels = []
        test_loader = init_data_loader()
        for data in test_loader:
                label = data['label']
                true_labels.append(label)
        true_labels = np.concatenate(true_labels, axis=0)
        true_labels = true_labels.reshape(355, 1)


        acc_st = get_acc(st_list, true_labels)
        acc_ts = get_acc(ts_list, true_labels)
        # acc_motion = get_acc(motion_list, true_labels)
        # acc_bone = get_acc(bone_list, true_labels)
        print("st acc:", acc_st)
        print("ts acc:", acc_ts)
        # print("st acc:", acc_motion)
        # print("ts acc:", acc_bone)

        # val_get_pic(st_list, true_labels, acc_st,style = 'st')
        # val_get_pic(ts_list, true_labels, acc_ts,style = 'ts')
        # plot_confusion(st_list, true_labels, style='st')
        # plot_confusion(ts_list, true_labels, style='ts')
        # 计算集成结果的准确率
        accuracy = emsemble_acc(st_list, ts_list, true_labels,motion_list,bone_list)

        print("Ensemble acc:", accuracy)