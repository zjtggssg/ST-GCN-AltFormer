import torch
import argparse
import os
import torch.nn.functional as F
from matplotlib import pyplot as plt
from matplotlib.colors import BoundaryNorm
from sklearn.metrics import confusion_matrix
from torch import softmax
from visualizer import get_local

from SHREC.visualize_funtions import visualize_head, visualize_heads, visualiize_head_mean
from model.ST_TR.ST_TR_new import ST_TR_new

get_local.activate() # 激活装饰器
from SHREC.Confusion import DrawConfusionMatrix
from model.ST_GCN_AltFormer import ST_GCN_AltFormer
from dataset_node import *
from data_process.Hand_Dataset import Hand_Dataset


parser = argparse.ArgumentParser()

parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--cuda', default=True, help='enables cuda')
parser.add_argument('--data_cfg', type=int, default=0)

def init_data_loader(data_cfg,hand_class):
    # train_data, test_data = split_train_test(data_cfg)
    # all_data = test_data
    all_data = np.load('/data/zjt/HandGestureDataset_SHREC2017/gesture/test.npy', allow_pickle=True)
    all_data = [sample for sample in all_data if sample["label"] == hand_class]

    all_dataset = Hand_Dataset(all_data, use_data_aug=False, time_len=180,expand = 0)

    print("test data num: ", len(all_dataset))

    print("batch size:", args.batch_size)

    all_loader = torch.utils.data.DataLoader(
        all_dataset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=8, pin_memory=False)


    return all_loader

def init_model_str(data_cfg):
    if data_cfg == 0:
        class_num = 14
    elif data_cfg == 1:
        class_num = 28

    output_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    l = [('labeling_mode', 'spatial')]
    graph_args = dict(l)
    address = 'graph.SHRE'
    model = ST_TR_new(channel=3, num_class=class_num, window_size=300, num_point=22, attention=True,
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
    model_path = '/home/zjt/Desktop/ST-GCN-AltFormer/weight/str.pth'
    state_dict = torch.load(model_path)

    model.load_state_dict(state_dict)

    model.eval()
    return model


def init_model_ttr(data_cfg):
    if data_cfg == 0:
        class_num = 14
    elif data_cfg == 1:
        class_num = 28

    output_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    l = [('labeling_mode', 'spatial')]
    graph_args = dict(l)
    address = 'graph.SHRE'
    model = ST_TR_new(channel=3, num_class=class_num, window_size=300, num_point=22, attention=False,
                      only_attention=True,
                      tcn_attention=True, all_layers=False, only_temporal_attention=True, attention_3=False,
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
    model_path = '/home/zjt/Desktop/ST-GCN-AltFormer/weight/ttr.pth'
    state_dict = torch.load(model_path)

    model.load_state_dict(state_dict)

    model.eval()
    return model

def model_forward(sample_batched, model):
    data = sample_batched["skeleton"].float()
    data = data.unsqueeze(0)
    score= model(data)

    return score

def get_acc(score, labels):
    score = score.cpu().data.numpy()
    labels = labels.cpu().data.numpy()
    outputs = np.argmax(score, axis=1)
    return np.sum(outputs == labels) / float(labels.size)

fmap_block = dict()  # 装feature map
grad_block = dict()


def normalize(normalized):
    normalized =  np.abs(normalized)
    min_val = normalized.min()
    max_val = normalized.max()
    normalized_matrix = (normalized - min_val) / (max_val - min_val)
    return normalized_matrix


def plot_cam_pic(normalized_matrix,style,pd = None):
    median_value = np.median(normalized_matrix)
    print(median_value)
    percentile_80 = np.percentile(normalized_matrix, 80)
    normalized_matrix[normalized_matrix <= percentile_80] = 0
    cmap = plt.get_cmap('viridis')

    plt.imshow(normalized_matrix.T, cmap=cmap,interpolation='none', aspect='auto')

    plt.colorbar()

    plt.title("{}_{}".format(style,selected_index))

    plt.xlabel("frames")

    plt.ylabel("joints")
    plt.yticks(np.arange(0, normalized_matrix.shape[1], 1))
    plt.savefig('/data/zjt/HandGestureDataset_SHREC2017/gesture/{}/{}_{}_{}.png'.format(pd,style,selected_index,pd))


if __name__ == '__main__':

    #
    hand = 10
    selected_index = 58  # 替换为你想要的索引
    style = 'Swipe Down'
    pd = 'str'
    target = 9
    npy = 0

    if npy == 1:
        matrix = np.load(
            '/data/zjt/HandGestureDataset_SHREC2017/gesture/{}/{}_{}_{}.npy'.format(pd, style, selected_index,
                                                                                    pd))
        plot_cam_pic(matrix, style, pd)



    else:
        args = parser.parse_args()
        print(args)

        test_loader = init_data_loader(args.data_cfg, hand_class=hand)



        print(len(test_loader))

        # 获取数据和标签
        sample_batched = test_loader.dataset[selected_index]  # 通过索引获取数据集中的样本

        title = sample_batched["title"]
        print(title)

        label = sample_batched["label"]
        label = torch.LongTensor([label]).cuda()

        l = []
        l_st = []
        l_ts = []

        if pd == 'str':
            model = init_model_str(args.data_cfg)
            score_final = model_forward(sample_batched, model)
            acc1 = get_acc(score_final , label)
            print("x_st Accuracy: {:.2f}%".format(acc1 * 100))
            pred_st = F.softmax(score_final , dim=1)
            class_st = pred_st[:, target]
            class_st = class_st.item()
            print("gailv", pred_st)
            m = np.array(sample_batched['skeleton'])

            zeros_start = 0

            for i in range(m.shape[0]):
                if np.all(m[i:] == m[-1]):
                    zeros_start = i
                    break

            print(zeros_start)
            for i in range(zeros_start):
                print(i)
                for j in range(22):
                    n = np.zeros(3)
                    m_copy = m.copy()
                    m_copy[i][j] = n
                    data = torch.tensor(m_copy).float().unsqueeze(0)
                    score = model(data)
                    pred_st = F.softmax(score, dim=1)
                    class_prob_st = pred_st[:, target].item()
                    class_st_new = class_st - class_prob_st

                    l_st.append(class_st_new)

                    del n, m_copy, data, score

            print(len(l_st))
            normalized = np.array(l_st).reshape(zeros_start, 22)
            normalized_matrix = normalize(normalized)
            np.save(
                '/data/zjt/HandGestureDataset_SHREC2017/gesture/{}/{}_{}_{}.npy'.format(pd, style, selected_index, pd),
                normalized_matrix)
            plot_cam_pic(normalized_matrix, style, pd)

        elif pd == 'ttr':
            model = init_model_ttr(args.data_cfg)
            score_final = model_forward(sample_batched, model)
            acc2 = get_acc(score_final, label)
            print("x_ts Accuracy: {:.2f}%".format(acc2 * 100))

            pred_ts = F.softmax(score_final, dim=1)
            class_ts = pred_ts[:, target]
            class_ts = class_ts.item()
            print("gailv", pred_ts)
            m = np.array(sample_batched['skeleton'])

            zeros_start = 0

            for i in range(m.shape[0]):
                if np.all(m[i:] == m[-1]):
                    zeros_start = i
                    break

            print(zeros_start)
            for i in range(zeros_start):
                print(i)
                for j in range(22):
                    n = np.zeros(3)
                    m_copy = m.copy()
                    m_copy[i][j] = n
                    data = torch.tensor(m_copy).float().unsqueeze(0)
                    score = model(data)
                    pred_ts = F.softmax(score, dim=1)
                    class_prob_ts = pred_ts[:, target].item()
                    class_ts_new = class_ts - class_prob_ts
                    l_ts.append(class_ts_new)
                    del n, m_copy, data, score

            print(len(l_ts))

            normalized = np.array(l_ts).reshape(zeros_start, 22)

            normalized_matrix = normalize(normalized)
            np.save(
                '/data/zjt/HandGestureDataset_SHREC2017/gesture/{}/{}_{}_{}.npy'.format(pd, style, selected_index, pd),
                normalized_matrix)
            plot_cam_pic(normalized_matrix, style, pd)

        else:
            pd1 = 'str'
            pd2 = 'ttr'
            str_matrix = np.load(
                '/data/zjt/HandGestureDataset_SHREC2017/gesture/{}/{}_{}_{}.npy'.format(pd1, style, selected_index,
                                                                                        pd1))
            ttr_matrix = np.load(
                '/data/zjt/HandGestureDataset_SHREC2017/gesture/{}/{}_{}_{}.npy'.format(pd2, style, selected_index,
                                                                                        pd2))

            st_tt_matrix =  str_matrix +  ttr_matrix
            normalized_matrix = normalize(st_tt_matrix)
            np.save(
                '/data/zjt/HandGestureDataset_SHREC2017/gesture/{}/{}_{}_{}.npy'.format(pd, style, selected_index, pd),
                normalized_matrix)
            plot_cam_pic(normalized_matrix, style, pd)