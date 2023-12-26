import random
from torch.utils.data import Dataset
import torch
from random import randint,shuffle
import numpy as  np
##选出八帧进行标签平滑

l = []

class Hand_Dataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, data, time_len, use_data_aug,expand):
        """
        Args:
            data: 视频列表及其标签
            time_len: 输入视频长度
            use_data_aug: 数据扩充
        """
        self.use_data_aug = use_data_aug
        self.expand = expand
        self.data = data

        self.time_len = time_len
        self.compoent_num = 22



    def __len__(self):
        return len(self.data)

    def __getitem__(self, ind):
        # print("ind:",ind)
        data_ele = self.data[ind]
        title = data_ele["title"]

        #hand skeleton
        skeleton = data_ele["skeleton"]
        skeleton = np.array(skeleton)


        # print(skeleton.shape[0])

        if self.use_data_aug:
            skeleton = self.data_aug(skeleton)


            # print(skeleton)
            # print("skeleton",skeleton.shape)


        # sample time_len frames from whole video
        data_num = skeleton.shape[0]
        idx_list = self.sample_frame(data_num)

        skeleton = [skeleton[idx] for idx in idx_list]
        skeleton = np.array(skeleton)
        # print("1:",skeleton.shape)

        #normalize by palm center
        skeleton -= skeleton[0][1]

        if self.expand == 1:
            skeleton = self.motion(skeleton)

        elif self.expand == 2:
            skeleton = self.bone(skeleton)

        skeleton = torch.from_numpy(skeleton).float()
        # print("2:", skeleton.shape)


        # skeleton = torch.from_numpy(skeleton).float()

        # print("s:", type(skeleton))
        #print(skeleton.shape)
        # label
        label = data_ele["label"] - 1 #

        sample = {'skeleton': skeleton, "label" : label,"title": title}

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


    def motion(self, skeleton):

        data = skeleton
        dataset = np.zeros((180,22,3))

        data_num = data.shape[0] - 1

        for i in range(0, data_num):
            for j in range(0, 22):
                dataset[i][j] = data[i + 1][j] - data[i][j]
        # data = np.delete(data, data_num, axis=0)
        dataset[data_num] = data[data_num] - data[data_num]

        new_skeleton  = np.array(dataset)

        return new_skeleton

    def bone (self, skeleton):
        start = [0, 0, 0, 2, 3, 4, 1, 6, 7, 8, 1, 10, 11, 12, 1, 14, 15, 16, 1, 18, 19, 20]
        end = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]

        data = skeleton
        dataset = np.zeros((180, 22, 3))

        data_num = data.shape[0]

        for i in range(0, data_num):
            for j in range(0, 22):
                s = int(start[j])
                e = int(end[j])
                dataset[i][j] = data[i][e] - data[i][s]
        # data = np.delete(data, data_num, axis=0)
        new_skeleton = np.array(dataset)

        return new_skeleton

    # def Bone(self,skeleton):





# if __name__ == '__main__':
#
#     train_data, test_data = get_train_test_data(3,0)
#     # print(np.array(train_data[2000]))
#     train_dataset = Hand_Dataset(train_data, use_data_aug=True, time_len=8)
#     test_dataset = Hand_Dataset(test_data, use_data_aug=False, time_len=8)
    # print(train_dataset)
    # print("train data num: ", len(train_dataset))
    # print(np.array(list(train_data.__getitem__(13)['skeleton'])).shape)
    # print(list(train_data.__getitem__(0)['skeleton']))
    # print("test data num: ", len(test_dataset))
