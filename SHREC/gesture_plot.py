import cv2
import matplotlib.pyplot as plt
import numpy as np

connection_data = [(0, 2), (2, 3), (3, 4), (4, 5), (0, 1), (1, 6), (6, 7), (7, 8),
          (8, 9), (1, 10), (10, 11), (11, 12), (12, 13), (1, 14), (14, 15), (15, 16),
          (16, 17),(1,18),(18,19),(19,20),(20,21)]

# 从txt文档中读取数据
def read_data_from_txt(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # 获取关节点数据和连接情况数据
    node_data = []

    for line in lines:
        line = line.strip().split(' ')
        if len(line) == 44:
            node_data.append([float(x) for x in line])

    return np.array(node_data)


# 绘制手势图
# def plot_gesture(node_data, connection_data, frame_index):
#     # 获取指定帧的关节点数据
#     frame_nodes = node_data[frame_index]
#
#     # 创建一个新的图形
#     plt.figure(figsize=(8, 8))
#
#     # 绘制关节点
#     for i in range(22):
#         x, y = frame_nodes[i * 2], frame_nodes[i * 2 + 1]
#         plt.scatter(x, y, c='b', marker='o')
#         plt.text(x, y, str(i), fontsize=12, ha='center', va='bottom', color='r')
#
#     # 绘制连接情况
#     for connection in connection_data:
#         start_node, end_node = connection
#         print(start_node)
#         x1, y1 = frame_nodes[start_node * 2], frame_nodes[start_node * 2 + 1]
#         x2, y2 = frame_nodes[end_node * 2], frame_nodes[end_node * 2 + 1]
#         plt.plot([x1, x2], [y1, y2], 'k-')
#
#     plt.axis('equal')
#     plt.title(f'Gesture Frame {frame_index}')
#     plt.xlabel('X-axis')
#     plt.ylabel('Y-axis')
#     plt.grid(True)
#     # plt.show()
#     file_name = "/data/zjt/HandGestureDataset_SHREC2017/gesture/Tap_{}.png".format(frame_index)
#     plt.savefig(file_name)
def plot_gesture_on_depth_image(depth_image_path, node_data, connection_data, frame_index):
    # 获取指定帧的关节点数据
    frame_nodes = node_data[frame_index]
    depth_image = cv2.imread(depth_image_path,cv2.IMREAD_GRAYSCALE)
    # 创建一个与深度图像相同大小的图像
    gesture_image = cv2.cvtColor(depth_image, cv2.COLOR_GRAY2BGR)
    node_colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 0, 255), (255, 255, 0)]
    line_colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 0, 255), (255, 255, 0)]

    # 绘制关节点
    for i in range(22):
        x, y = int(frame_nodes[i * 2]), int(frame_nodes[i * 2 + 1])
        cv2.circle(gesture_image, (x, y), 2, node_colors[i % len(node_colors)], -1,cv2.LINE_AA)  # 增加标记的大小
        cv2.putText(gesture_image, str(i), (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)  # 标记关节点数字

    # 绘制连接情况
    for i, connection in enumerate(connection_data):
        start_node, end_node = connection
        x1, y1 = int(frame_nodes[start_node * 2]), int(frame_nodes[start_node * 2 + 1])
        x2, y2 = int(frame_nodes[end_node * 2]), int(frame_nodes[end_node * 2 + 1])
        cv2.line(gesture_image, (x1, y1), (x2, y2), line_colors[i % len(line_colors)], 1,cv2.LINE_AA)  # 减小连接线的粗细

    return gesture_image

if __name__ == "__main__":
    # 指定txt文档路径
    file_path = '/data/zjt/HandGestureDataset_SHREC2017/gesture_8/finger_2/subject_28/essai_1/skeletons_image.txt'

    # 读取数据
    node_data = read_data_from_txt(file_path)
    print(node_data.shape)
    # 指定要绘制的帧索引
    for i in range(len(node_data)):
        frame_index = i  # 修改为你想要绘制的帧的索引
        depth_image_path = '/data/zjt/HandGestureDataset_SHREC2017/gesture_2/finger_1/subject_24/essai_1/{}_depth.png'.format(frame_index)
        # 绘制手势图
        # plot_gesture(node_data, connection_data, frame_index_to_plot)
        result_image = plot_gesture_on_depth_image(depth_image_path, node_data, connection_data, frame_index)
        result_image = cv2.resize(result_image, (640, 480))
        # 显示或保存结果图像
        # cv2.imshow('Gesture on Depth Image', result_image)

        # 如果要保存结果图像
        cv2.imwrite('/data/zjt/HandGestureDataset_SHREC2017/gesture/Tap_{}.png'.format(frame_index), result_image)
