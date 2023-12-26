import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection

# 连接映射
from scipy.io import loadmat

CONNECTION_MAP = np.array([[1, 2], [2, 3], [2, 4], [2, 20], [3, 4], [3, 20], [4, 5], [5, 6], [6, 7], [4, 8], [8, 9],
                            [9, 10], [10, 11], [8, 12], [12, 13], [13, 14], [14, 15], [12, 16], [16, 17], [17, 18],
                            [18, 19], [16, 20], [20, 21], [21, 22], [22, 23], [24, 25], [25, 26], [25, 27], [25, 43],
                            [26, 27], [26, 43], [27, 28], [28, 29], [29, 30], [27, 31], [31, 32], [32, 33], [33, 34],
                            [31, 35], [35, 36], [36, 37], [37, 38], [35, 39], [39, 40], [40, 41], [41, 42], [39, 43],
                            [43, 44], [44, 45], [45, 46]])


if __name__ == '__main__':
    # 加载数据
    # 假设有一个名为 'DataFile1.npy' 的 NumPy 文件，其中包含骨骼数据
    a = 0
    for i in range(1,2):
        data = loadmat('/data/zjt/LMDHG_MAT/LMDHG/DataFile{}.mat'.format(i))
        # print(data)
        skeleton = data['skeleton']
        print(skeleton[734])  # 13756
        labels = data['labels']  # 26
        a = a + len(labels)
        print(labels)
        Anotations = data['Anotations']  # 26
        print(Anotations)

    print(a)

    # 初始化图形窗口和坐标轴
    fig = plt.figure('Viewer')
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim([-400, 400])
    ax.set_ylim([-400, 400])
    ax.set_zlim([-400, 400])
    ax.grid(True)

    # 绘制骨骼
    zoom_factor = 1
    current_action = 1
    class_label = labels[current_action][0]  # Change square brackets to access the element

    try:
        i = 0
        while i < skeleton.shape[0]:
            x = np.column_stack(
                [skeleton[i][CONNECTION_MAP[:, 0], 0], skeleton[i][CONNECTION_MAP[:, 1], 0]]) * zoom_factor
            y = np.column_stack(
                [skeleton[i][CONNECTION_MAP[:, 0], 1], skeleton[i][CONNECTION_MAP[:, 1], 1]]) * zoom_factor
            z = np.column_stack(
                [skeleton[i][CONNECTION_MAP[:, 0], 2], skeleton[i][CONNECTION_MAP[:, 1], 2]]) * zoom_factor

            # 更新骨骼图
            ax.cla()
            ax.set_title(f'[Frame = {i}, Class = {class_label}]')
            ax.set_xlim([-400, 400])
            ax.set_ylim([-400, 400])
            ax.set_zlim([-400, 400])
            ax.grid(True)
            ax.add_collection3d(Line3DCollection([list(zip(x[k], y[k], z[k])) for k in range(len(CONNECTION_MAP))],
                                                 colors='b', linewidths=3))

            plt.draw()
            plt.pause(0.0003)

            # 类别标签切换
            if i > Anotations[current_action, 1]:
                current_action += 1
                class_label = labels[current_action][0]  # Change square brackets to access the element

            i += 1


    except Exception as e:
        plt.close('all')
        print(f'force closed! Error: {e}')

    plt.show()
