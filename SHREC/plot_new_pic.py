import torch
import argparse
import os
import torch.nn.functional as F
from matplotlib import pyplot as plt, colors
from matplotlib.colors import BoundaryNorm
from sklearn.metrics import confusion_matrix
from torch import softmax
from visualizer import get_local

from SHREC.visualize_funtions import visualize_head, visualize_heads, visualiize_head_mean

get_local.activate() # 激活装饰器
from SHREC.Confusion import DrawConfusionMatrix
from model.ST_GCN_AltFormer import ST_GCN_AltFormer
from dataset_node import *
from data_process.Hand_Dataset import Hand_Dataset


parser = argparse.ArgumentParser()

parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--cuda', default=True, help='enables cuda')
parser.add_argument('--data_cfg', type=int, default=0)

def init_data_loader(data_cfg,hand_class,expand):
    train_data, test_data = split_train_test(data_cfg)
    all_data = test_data
    # all_data = np.load('/data/zjt/HandGestureDataset_SHREC2017/gesture/test.npy', allow_pickle=True)
    all_data = [sample for sample in all_data if sample["label"] == hand_class]

    all_dataset = Hand_Dataset(all_data, use_data_aug=False, time_len=180,expand = expand)

    print("test data num: ", len(all_dataset))

    print("batch size:", args.batch_size)

    all_loader = torch.utils.data.DataLoader(
        all_dataset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=8, pin_memory=False)


    return all_loader

def init_model(data_cfg,model_style):
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
    if model_style ==  'st':
        model_path = '/home/zjt/Desktop/ST-GCN-AltFormer/weight/0.9703.pth'
    elif model_style ==  'ts':
        model_path = '/home/zjt/Desktop/ST-GCN-AltFormer/weight/0.9631.pth'
    else:
        model_path = '/home/zjt/Desktop/ST-GCN-AltFormer/weight/0.975.pth'

    state_dict = torch.load(model_path)

    model.load_state_dict(state_dict)

    model.eval()
    return model

def init_model_motion(data_cfg):
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
                             drop_connect=True, concat_original=True, dv=0.25, dk=0.25, Nh=8, dim_block1=10,
                             dim_block2=30,
                             dim_block3=75,
                             data_normalization=True, visualization=False, skip_conn=True, adjacency=False,
                             kernel_temporal=9, bn_flag=True, weight_matrix=2, more_channels=False, n=4,
                             device=output_device,
                             graph=address,
                             graph_args=graph_args
                             )

    model = torch.nn.DataParallel(model).cuda()
    model_path = '/home/zjt/Desktop/ST-GCN-AltFormer/weight/0.9476_motion.pth'
    state_dict = torch.load(model_path)

    model.load_state_dict(state_dict)

    model.eval()
    return model

def init_model_bone(data_cfg):
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
                             drop_connect=True, concat_original=True, dv=0.25, dk=0.25, Nh=8, dim_block1=10,
                             dim_block2=30,
                             dim_block3=75,
                             data_normalization=True, visualization=False, skip_conn=True, adjacency=False,
                             kernel_temporal=9, bn_flag=True, weight_matrix=2, more_channels=False, n=4,
                             device=output_device,
                             graph=address,
                             graph_args=graph_args
                             )

    model = torch.nn.DataParallel(model).cuda()
    model_path = '/home/zjt/Desktop/ST-GCN-AltFormer/weight/0.8690_bone.pth'
    state_dict = torch.load(model_path)

    model.load_state_dict(state_dict)

    model.eval()
    return model

def model_forward(sample_batched, model):
    data = sample_batched["skeleton"].float()
    data = data.unsqueeze(0)
    score,x_st,x_ts = model(data)

    return score,x_st,x_ts

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


def plot_cam_pic(normalized_matrix, style, pd=None):
    normalized_matrix = np.abs(normalized_matrix)
    median_value = np.median(normalized_matrix)
    max_val = normalized_matrix.max()
    print(max_val)
    percentile_80 = np.percentile(normalized_matrix, 80)
    normalized_matrix[normalized_matrix <= percentile_80] = 0
    cmap = plt.get_cmap('viridis')

    plt.imshow(normalized_matrix.T, cmap=cmap, interpolation='none', aspect='auto')


    cbar = plt.colorbar()
    plt.clim(0, 0.15)  # Set colorbar range from 0 to 0.6
    cbar.set_ticks(np.arange(0, 0.2, 0.05))
    plt.title("{}_{}".format(style, selected_index))

    plt.xlabel("frames")

    plt.ylabel("joints")
    plt.yticks(np.arange(0, normalized_matrix.shape[1], 1))

    plt.savefig('/data/zjt/HandGestureDataset_SHREC2017/gesture/{}/{}_{}_{}.png'.format(pd, style, selected_index, pd))







def plot_cam_pic_mutil(normalized_matrix,style,pd = None,emsemble = None):
    median_value = np.median(normalized_matrix)
    print(median_value)
    percentile_80 = np.percentile(normalized_matrix, 80)
    normalized_matrix[normalized_matrix <= percentile_80] = 0
    cmap = plt.get_cmap('viridis')

    plt.imshow(normalized_matrix.T, cmap=cmap,interpolation='none', aspect='auto')

    cbar = plt.colorbar()
    plt.clim(0, 0.1)  # Set colorbar range from 0 to 0.6
    cbar.set_ticks(np.arange(0, 0.15, 0.05))
    plt.title("{}_{}".format(style, selected_index))

    plt.xlabel("frames")

    plt.ylabel("joints")
    plt.yticks(np.arange(0, normalized_matrix.shape[1], 1))
    plt.savefig('/data/zjt/HandGestureDataset_SHREC2017/gesture/mutil/{}/{}_{}_{}.png'.format(emsemble,style,selected_index,pd))


if __name__ == '__main__':

    #
    hand = 6
    selected_index = 49  # 替换为你想要的索引
    style = 'Rotation CCW'
    pd = 'ts'
    target = 5
    npy = 0
    emsemble = 'joint'


    if npy == 1:
        if pd == 'st_ts':
            matrix = np.load(
                '/data/zjt/HandGestureDataset_SHREC2017/gesture/mutil/{}/{}_{}_{}.npy'.format(emsemble, style, selected_index,
                                                                                              pd))
            plot_cam_pic_mutil(matrix, style, pd=pd, emsemble=emsemble)
        else:
            matrix = np.load(
                '/data/zjt/HandGestureDataset_SHREC2017/gesture/{}/{}_{}_{}_new.npy'.format(pd, style, selected_index,
                                                                                              pd))
            plot_cam_pic(matrix, style, pd)


    else:
        args = parser.parse_args()
        print(args)
        test_loader = init_data_loader(args.data_cfg, hand_class=hand, expand=0)
        # if emsemble == 'joint':
        #     print(1)
        #     test_loader = init_data_loader(args.data_cfg, hand_class=hand, expand=0)
        #
        # if emsemble ==  'bone' :
        #     print(2)
        #     test_loader = init_data_loader(args.data_cfg, hand_class=hand, expand=2)
        #
        # else:
        #     test_loader = init_data_loader(args.data_cfg, hand_class=hand, expand=1)




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
        l_motion = []
        l_bone = []

        if pd == 'st':
            model = init_model(args.data_cfg, model_style=pd)
            score_final, x_st, x_ts = model_forward(sample_batched, model)
            acc1 = get_acc(x_st, label)
            print("x_st Accuracy: {:.2f}%".format(acc1 * 100))
            pred_st = F.softmax(x_st, dim=1)
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
                    score, x_st, x_ts = model(data)
                    pred_st = F.softmax(x_st, dim=1)
                    class_prob_st = pred_st[:, target].item()
                    class_st_new = class_st - class_prob_st

                    l_st.append(class_st_new)

                    del n, m_copy, data, score

            print(len(l_st))
            normalized = np.array(l_st).reshape(zeros_start, 22)
            # normalized_matrix = normalize(normalized)
            np.save(
                '/data/zjt/HandGestureDataset_SHREC2017/gesture/{}/{}_{}_{}_new.npy'.format(pd, style, selected_index, pd),
                normalized)
            plot_cam_pic(normalized, style, pd)

        elif pd == 'ts':
            model = init_model(args.data_cfg, model_style=pd)
            score_final, x_st, x_ts = model_forward(sample_batched, model)
            acc2 = get_acc(x_ts, label)
            print("x_ts Accuracy: {:.2f}%".format(acc2 * 100))

            pred_ts = F.softmax(x_ts, dim=1)
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
                    score, x_st, x_ts = model(data)
                    pred_ts = F.softmax(x_ts, dim=1)
                    class_prob_ts = pred_ts[:, target].item()
                    class_ts_new = class_ts - class_prob_ts
                    l_ts.append(class_ts_new)
                    del n, m_copy, data, score

            print(len(l_ts))

            normalized = np.array(l_ts).reshape(zeros_start, 22)

            # normalized_matrix = normalize(normalized)
            np.save(
                '/data/zjt/HandGestureDataset_SHREC2017/gesture/{}/{}_{}_{}_new.npy'.format(pd, style, selected_index, pd),
                normalized)
            plot_cam_pic(normalized, style, pd)

        elif pd == 'bone':

            model = init_model_bone(args.data_cfg)
            score_final, x_st, x_ts = model_forward(sample_batched, model)
            acc2 = get_acc(score_final, label)
            print("bone Accuracy: {:.2f}%".format(acc2 * 100))

            pred_bone = F.softmax(score_final, dim=1)
            class_bone = pred_bone[:, target]
            class_bone = class_bone.item()
            print("gailv", pred_bone)
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
                    score, x_st, x_ts = model(data)
                    pred_bone = F.softmax(score, dim=1)
                    class_prob_bone = pred_bone[:, target].item()
                    class_bone_new = class_bone - class_prob_bone
                    l_bone.append(class_bone_new)
                    del n, m_copy, data, score

            print(len(l_bone))

            normalized = np.array(l_bone).reshape(zeros_start, 22)

            normalized_matrix = normalize(normalized)
            np.save(
                '/data/zjt/HandGestureDataset_SHREC2017/gesture/{}/{}_{}_{}.npy'.format(pd, style, selected_index, pd),
                normalized_matrix)
            plot_cam_pic(normalized_matrix, style, pd)

        elif pd =='motion':
            model = init_model_motion(args.data_cfg)
            score_final, x_st, x_ts = model_forward(sample_batched, model)
            acc2 = get_acc(score_final, label)
            print("bone Accuracy: {:.2f}%".format(acc2 * 100))

            pred_motion = F.softmax(score_final, dim=1)
            class_motion = pred_motion[:, target]
            class_motion = class_motion.item()
            print("gailv", pred_motion)
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
                    score, x_st, x_ts = model(data)
                    pred_motion= F.softmax(score, dim=1)
                    class_prob_motion = pred_motion[:, target].item()
                    class_motion_new = class_motion - class_prob_motion
                    l_motion.append(class_motion_new)
                    del n, m_copy, data, score

            print(len(l_motion))

            normalized = np.array(l_motion).reshape(zeros_start, 22)

            normalized_matrix = normalize(normalized)
            np.save(
                '/data/zjt/HandGestureDataset_SHREC2017/gesture/{}/{}_{}_{}.npy'.format(pd, style, selected_index, pd),
                normalized_matrix)
            plot_cam_pic(normalized_matrix, style, pd)


        else:
            pd1 = 'st'
            pd2 = 'ts'
            pd3 = 'bone'
            pd4 = 'motion'
            st_matrix = np.load(
                '/data/zjt/HandGestureDataset_SHREC2017/gesture/{}/{}_{}_{}_new.npy'.format(pd1, style, selected_index,
                                                                                        pd1))
            ts_matrix = np.load(
                '/data/zjt/HandGestureDataset_SHREC2017/gesture/{}/{}_{}_{}_new.npy'.format(pd2, style, selected_index,
                                                                                        pd2))

            # motion_matrix = np.load(
            #     '/data/zjt/HandGestureDataset_SHREC2017/gesture/{}/{}_{}_{}.npy'.format(pd4, style, selected_index,
            #                                                                             pd4))

            # emsemble_matrix = st_matrix +  ts_matrix
            emsemble_matrix = 0.8 * st_matrix + 0.2 *  ts_matrix
            # emsemble_matrix = st_matrix + ts_matrix + bone_matrix + motion_matrix
            # emsemble_matrix = st_matrix + bone_matrix
            # st_ts_matrix =  st_matrix +  ts_matrix



            if emsemble == 'joint':
                # normalized_matrix = normalize(emsemble_matrix)
                np.save(
                    '/data/zjt/HandGestureDataset_SHREC2017/gesture/mutil/{}/{}_{}_{}.npy'.format(emsemble, style,
                                                                                                  selected_index, pd),
                    emsemble_matrix)
                plot_cam_pic_mutil(emsemble_matrix, style, pd=pd, emsemble=emsemble)

            else:
                bone_matrix = np.load(
                    '/data/zjt/HandGestureDataset_SHREC2017/gesture/{}/{}_{}_{}.npy'.format(pd3, style, selected_index,
                                                                                            pd3))
                normalized_matrix = normalize(emsemble_matrix)
                bone_matrix = normalize(bone_matrix)

                mutil_matrix = np.concatenate((normalized_matrix , bone_matrix ), axis=1)
                np.save(
                    '/data/zjt/HandGestureDataset_SHREC2017/gesture/mutil/{}/{}_{}_{}.npy'.format(emsemble, style,
                                                                                                  selected_index, pd),
                    normalized_matrix)
                plot_cam_pic_mutil(mutil_matrix , style, pd=pd, emsemble=emsemble)