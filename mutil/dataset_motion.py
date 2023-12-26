import numpy as np
min_seq = 180
#Change the path to your downloaded SHREC2017 dataset
dataset_fold = "/data/zjt/HandGestureDataset_SHREC2017"
c = np.zeros((22,3))
def sport_dataset(node_dataset):

    data = node_dataset ##list
    # print("len:",len(data))
    data = np.array(data)
    data_num = data.shape[0] - 1

    for i in range(0, data_num):
        for j in range(0, 22):
            data[i][j] = data[i + 1][j] - data[i][j]
    # data = np.delete(data, data_num, axis=0)
    data[data_num] = data[data_num] - data[data_num]

    data = list(data)

    return data




def split_train_sport_test(data_cfg):
    def parse_file(data_file,data_cfg):
        #parse train / test file
        label_list = []
        all_data = []
        for line in data_file:
            data_ele = {}
            data = line.split() #【id_gesture， id_finger， id_subject， id_essai， 14_labels， 28_labels size_sequence】
            #video label
            if data_cfg == 0:
                label = int(data[4])
            elif data_cfg == 1:
                label = int(data[5])
            label_list.append(label) #add label to label list
            data_ele["label"] = label
            #video
            video = []
            joint_path = dataset_fold + "/gesture_{}/finger_{}/subject_{}/essai_{}/skeletons_world.txt".format(data[0],data[1],data[2],data[3])
            joint_file = open(joint_path)
            for joint_line in joint_file:
                joint_data = joint_line.split()
                joint_data = [float(ele) for ele in joint_data]#convert to float
                joint_data = np.array(joint_data).reshape(22,3)#[[x1,y1,z1], [x2,y2,z2],.....]
                video.append(joint_data)
            while len(video) < min_seq:
                video.append(c)


            video = sport_dataset(video)
            # print(np.array(video).shape)
            data_ele["skeleton2"] = video
            data_ele["name"] = line
            all_data.append(data_ele)
            joint_file.close()

        #
        # print(all_data[0]["skeleton2"][0][0])

        return all_data, label_list



    print("loading training data edge........")
    train_path = dataset_fold + "/train_gestures.txt"
    train_file = open(train_path)
    train_data, train_label = parse_file(train_file,data_cfg)
    assert len(train_data) == len(train_label)

    print("training data num {}".format(len(train_data)))

    # print(train_data[0])

    print("loading testing data edge........")
    test_path = dataset_fold + "/test_gestures.txt"
    test_file = open(test_path)
    test_data, test_label = parse_file(test_file, data_cfg)
    assert len(test_data) == len(test_label)

    print("testing data num {}".format(len(test_data)))

    return train_data, test_data