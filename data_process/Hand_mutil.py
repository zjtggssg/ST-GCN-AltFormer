import random
from torch.utils.data import Dataset
import torch
from random import randint,shuffle
import numpy as  np
##选出八帧进行标签平滑

l = []

class Hand_mutil(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, data, time_len, use_data_aug,inputs,connect_joint):
        """
        Args:
            data: 视频列表及其标签
            time_len: 输入视频长度
            use_data_aug: 数据扩充
        """
        self.use_data_aug = use_data_aug
        self.conn = connect_joint
        self.data = data
        self.inputs = inputs
        self.time_len = time_len
        self.compoent_num = 22



    def __len__(self):
        return len(self.data)

    def __getitem__(self, ind):
        # print("ind:",ind)
        data_ele = self.data[ind]

        #hand skeleton
        data = data_ele["skeleton"]

        data = np.array(data)

        if self.use_data_aug:
            data = self.data_aug(data)

        joint, velocity, bone = self.multi_input(data[:, :self.time_len, :])

        data_new = []

        if 'J' in self.inputs:
            data_new.append(joint)
        if 'V' in self.inputs:
            data_new.append(velocity)
        if 'B' in self.inputs:
            data_new.append(bone)

        skeleton = np.stack(data_new, axis=0)
        # print(skeleton.shape[0])


        # sample time_len frames from whole video
        data_num = skeleton.shape[0]
        idx_list = self.sample_frame(data_num)

        skeleton = [skeleton[idx] for idx in idx_list]
        skeleton = np.array(skeleton)

        skeleton = torch.from_numpy(skeleton).float()
        # label
        label = data_ele["label"] - 1 #

        sample = {'skeleton': skeleton, "label" : label}

        return sample

    def data_aug(self, skeleton):

        def scale(skeleton):
            ratio = 0.2
            low = 1 - ratio
            high = 1 + ratio
            factor = np.random.uniform(low, high)
            video_len = skeleton.shape[0]
            for t in range(video_len):
                for j_id in range(self.compoent_num):
                    skeleton[t][j_id] *= factor
            skeleton = np.array(skeleton)
            return skeleton

        def shift(skeleton):
            low = -0.1
            high = -low
            offset = np.random.uniform(low, high, 3)
            video_len = skeleton.shape[0]
            for t in range(video_len):
                for j_id in range(self.compoent_num):
                    skeleton[t][j_id] += offset
            skeleton = np.array(skeleton)
            return skeleton

        def noise(skeleton):
            low = -0.1
            high = -low
            #select 4 joints
            all_joint = list(range(self.compoent_num))
            shuffle(all_joint)
            selected_joint = all_joint[0:4]

            for j_id in selected_joint:
                noise_offset = np.random.uniform(low, high, 3)
                for t in range(self.time_len):
                    skeleton[t][j_id] += noise_offset
            skeleton = np.array(skeleton)
            return skeleton

        def time_interpolate(skeleton):
            skeleton = np.array(skeleton)
            video_len = skeleton.shape[0]

            r = np.random.uniform(0, 1)

            result = []

            for i in range(1, video_len):
                displace = skeleton[i] - skeleton[i - 1]#d_t = s_t+1 - s_t
                displace *= r
                result.append(skeleton[i -1] + displace)# r*disp

            while len(result) < self.time_len:
                result.append(result[-1]) #padding
            result = np.array(result)
            return result




        # og_id = np.random.randint(3)
        aug_num = 4
        ag_id = randint(0, aug_num - 1)
        if ag_id == 0:
            skeleton = scale(skeleton)
        elif ag_id == 1:
            skeleton = shift(skeleton)
        elif ag_id == 2:
            skeleton = noise(skeleton)
        elif ag_id == 3:
            skeleton = time_interpolate(skeleton)

        return skeleton




    def sample_frame(self, data_num):
        #sample #time_len frames from whole video
        sample_size = self.time_len
        each_num = (data_num - 1) / (sample_size - 1)
        idx_list = [0, data_num - 1]
        for i in range(sample_size):
            index = round(each_num * i)
            if index not in idx_list and index < data_num:
                idx_list.append(index)
        idx_list.sort()

        while len(idx_list) < sample_size:
            idx = int(random.uniform(0, data_num - 1))
            # idx = random.randint(0, data_num - 1)
            if idx not in idx_list:
                idx_list.append(idx)
        idx_list.sort()

        return idx_list

    def multi_input(self, data):
        # print("data_start",data.shape) #3 288 25 2

        T, V, C = data.shape
        data = data.reshape(C,T,V)
        joint = np.zeros((C * 2, T, V))
        velocity = np.zeros((C * 2, T, V))
        bone = np.zeros((C * 2, T, V))
        joint[:C, :, :] = data
        # print("data",data.shape) #3 288 25 2
        for i in range(V):
            joint[C:, :, i] = data[:, :, i] - data[:, :, 1]
        for i in range(T - 2):
            velocity[:C, i, :] = data[:, i + 1, :] - data[:, i, :]
            velocity[C:, i, :] = data[:, i + 2, :] - data[:, i, :]
        for i in range(len(self.conn)):
            # print("len(self.conn)",len(self.conn))
            bone[:C, :, i] = data[:, :, i] - data[:, :, self.conn[i]]
        bone_length = 0
        for i in range(C):
            bone_length += bone[i, :, :] ** 2
        bone_length = np.sqrt(bone_length) + 0.0001
        # print("bone_length ",bone_length.shape)
        for i in range(C):
            bone[C + i, :, :] = np.arccos(bone[i, :, :] / bone_length)
        return joint, velocity, bone
