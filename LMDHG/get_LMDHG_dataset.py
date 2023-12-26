import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection

# 连接映射
from scipy.io import loadmat

label_dict = {'ATTRAPER': 0, #'Catching'
			'ATTRAPER_MAIN_LEVEE': 1,#'Catch with two hands'
			'C': 2, #'Draw C'
			'DEFILER_DOIGT': 3,#'Scroll'
			'LIGNE': 4, #'Line'
			'PIVOTER': 5,#'Rotate'
			'POINTER': 6,#'Point to'
			'POINTER_MAIN_LEVEE': 7,#'Point to with two hands'
			'REPOS': 8,#'Rest'
			'SECOUER': 9,#'Shake'
			'SECOUER_BAS': 10,#'Shake down'
			'SECOUER_POING_LEVE': 11,#'Shake with two hands'
			'TRANCHER': 12,#'Slice'
			'ZOOM': 13,} #'Zoom'


def generate_dataset(skeleton, labels, Anotations):
    datasets =[]

    for i in range(0,len(Anotations)):
        start =Anotations[i][0] -1
        end = Anotations[i][1] - 1
        subset = skeleton[start:end + 1]
        # print(len(subset))
        label = labels[i][0]
        # print(label)
        dataset = {'skeleton': subset , 'label':label_dict.get(label) }
        # print(dataset)
        datasets.append(dataset)

    return datasets

if __name__ == '__main__':
    # 加载数据
    # 假设有一个名为 'DataFile1.npy' 的 NumPy 文件，其中包含骨骼数据
    train_dataset = []
    test_dataset = []

    for i in range(1,36):
        data = loadmat('/data/zjt/LMDHG_MAT/LMDHG/DataFile{}.mat'.format(i))
        # print(data)
        skeleton = data['skeleton']

        labels = np.array([item[0] for item in data['labels']], dtype=str)
        # print(labels)

        # 删除dtype字段
        Anotations = data['Anotations']
        Anotations = np.array([(item[0], item[1]) for item in Anotations], dtype=int)

        # print(Anotations)
        datasets = generate_dataset(skeleton, labels,Anotations)
        train_dataset = train_dataset + datasets

    selected_dicts = [item for item in train_dataset if item['label'] == 8]
    print(len(train_dataset))
    print(len(selected_dicts))

    for j in range(36, 51):
        data = loadmat('/data/zjt/LMDHG_MAT/LMDHG/DataFile{}.mat'.format(j))
        # print(data)
        skeleton = data['skeleton']

        labels = np.array([item[0] for item in data['labels']], dtype=str)
        # print(labels)

        # 删除dtype字段
        Anotations = data['Anotations']
        Anotations = np.array([(item[0], item[1]) for item in Anotations], dtype=int)
        # print(Anotations)
        datasets = generate_dataset(skeleton, labels, Anotations)
        test_dataset = test_dataset + datasets

    selected_dicts = [item for item in test_dataset if item['label'] == 8]
    print(len(test_dataset))
    print(len(selected_dicts))

    np.save('/data/zjt/LMDHG/npy/train_new.npy', train_dataset)
    np.save('/data/zjt/LMDHG/npy/test_new.npy', test_dataset)


