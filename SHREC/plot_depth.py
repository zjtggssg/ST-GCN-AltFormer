import os

import cv2
from PIL import Image, ImageEnhance

# 打开PNG图片
from matplotlib import pyplot as plt

if __name__ == '__main__':

    style = 'Rotation CCW_49'
    list =['6', '2', '23', '4']


    output_directory = '/data/zjt/HandGestureDataset_SHREC2017/gesture/depth/{}'.format(style)
    os.makedirs(output_directory, exist_ok=True)



    for i in range(100):
        # i = 18
        path = '/data/zjt/HandGestureDataset_SHREC2017/gesture_{}/finger_{}/subject_{}/essai_{}/{}_depth.png'.format(list[0],list[1],list[2],list[3],i)
        depth_image = Image.open(path)
        depth_image = depth_image.convert("RGB")
        enhancer = ImageEnhance.Brightness(depth_image)
        depth_image = enhancer.enhance(2.0)  # 调整亮度，2.0是放大倍数
        enhancer = ImageEnhance.Contrast(depth_image)
        depth_image = enhancer.enhance(2.0)

        new_path = os.path.join(output_directory, '{}_depth.png'.format(i))
        depth_image.save(new_path )





    # with open('/data/zjt/HandGestureDataset_SHREC2017/gesture_9/finger_1/subject_1/essai_1/general_informations.txt', 'r') as file:
    #     # 初始化一个空列表来存储结果
    #     data_list = []
    #
    #     # 逐行读取文件内容
    #     for line in file:
    #         # 使用split函数将每行的文本分割成多个部分
    #         parts = line.strip().split()
    #         # 将分割后的部分转换为适当的数据类型，并创建元组
    #         data_tuple = (float(parts[0]), int(parts[1]), int(parts[2]), int(parts[3]), int(parts[4]))
    #         data_list.append(data_tuple)
    #
    #
    # print(data_list[60])
    # # 打印结果
    # x, y, width, height = data_list[59][1],data_list[59][2],data_list[59][3],data_list[59][4]
    # region = depth_image[y:y+height, x:x+width]
    # plt.figure(figsize=(6, 6))
    # plt.imshow(region, cmap='jet')  # 使用'jet' colormap来显示深度信息
    # plt.colorbar()
    # plt.show()


