import numpy as np
data_fold = "/data/zjt/archive/DHG2016"
min_seq = 150
c = np.zeros((22,3))

def read_data_from_disk():
    def parse_data(src_file):
        video = []
        for line in src_file:
            line = line.split("\n")[0]
            data = line.split(" ")
            frame = []
            point = []
            for data_ele in data:
                point.append(float(data_ele))
                if len(point) == 3:
                    frame.append(point)
                    point = []
            video.append(frame)

        return video

    result = {}
    for g_id in range(1, 15):
        # print("gesture {} / {}".format(g_id, 14))
        for f_id in range(1, 3):
            for sub_id in range(1, 21):
                for e_id in range(1, 6):
                    src_path = data_fold + "/gesture_{}/finger_{}/subject_{}/essai_{}/skeleton_world.txt".format(g_id,
                                                                                                                 f_id,
                                                                                                                 sub_id,
                                                                                                                 e_id)
                    src_file = open(src_path)
                    video = parse_data(src_file)  # the 22 points for each frame of the video
                    key = "{}_{}_{}_{}".format(g_id, f_id, sub_id, e_id)
                    result[key] = video

                    src_file.close()
    return result

def get_valid_frame(video_data):
    # filter frames using annotation
    info_path = data_fold + "/informations_troncage_sequences.txt"
    info_file = open(info_path)
    used_key = []

    for line in info_file:
        line = line.split("\n")[0]
        data = line.split(" ")
        g_id = data[0]
        f_id = data[1]
        sub_id = data[2]
        e_id = data[3]
        key = "{}_{}_{}_{}".format(g_id, f_id, sub_id, e_id)
        used_key.append(key)
        start_frame = int(data[4])
        end_frame = int(data[5])
        data = video_data[key]
        video_data[key] = data[(start_frame): end_frame + 1]

    return video_data

##将坐标拆成训练集与测试集
def split_train_test(test_subject_id,filtered_video_data,cfg):
    #split data into train and test
    #cfg = 0 >>>>>>> 14 categories      cfg = 1 >>>>>>>>>>>> 28 cate
    train_data = []
    test_data = []
    for g_id in range(1, 15):
        for f_id in range(1, 3):
            for sub_id in range(1, 21):
                for e_id in range(1, 6):
                    key = "{}_{}_{}_{}".format(g_id, f_id, sub_id, e_id)

                    #set table to 14 or
                    if cfg == 0:
                        label = g_id
                    elif cfg == 1:
                        if f_id == 1:
                            label = g_id
                        else:
                            label = g_id + 14

                    #split to train and test list
                    data = filtered_video_data[key]
                    while len(data) < min_seq:
                        data.append(c)

                    sample = {"skeleton":data, "label":label,"title": key}
                    if sub_id == test_subject_id:
                        test_data.append(sample)
                    else:
                        train_data.append(sample)
    if len(test_data) == 0:
        raise "no such test subject"

    return train_data, test_data

def get_train_test_data(test_subject_id, cfg):
    print("reading data from desk.......")
    ##未处理帧
    video_data = read_data_from_disk()
    # print("4:", np.array(video_data['1_1_1_1']).shape)
    print("filtering frames .......")
    ##处理帧
    filtered_video_data = get_valid_frame(video_data)
    print(type(filtered_video_data))
    # print("3:",np.array(filtered_video_data['1_1_1_1']).shape)
    train_data, test_data = split_train_test(test_subject_id,filtered_video_data,cfg) # list
    # print(type(train_data))
    # print(train_data[0])
    return train_data,test_data


