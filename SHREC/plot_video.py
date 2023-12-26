import cv2
import os
from PIL import Image
import numpy as np


if __name__ == '__main__':


    input_folder = '/data/zjt/HandGestureDataset_SHREC2017/gesture_2/finger_1/subject_24/essai_1/'
    image_extension = 'png'

    # 获取文件夹中的图像文件列表
    image_files = [os.path.join(input_folder, file) for file in os.listdir(input_folder) if
                   file.endswith('.' + image_extension)]
    image_files.sort()  # 确保文件按顺序排列


    if not image_files:
        print("没有找到图像文件。")
        exit()

    # 读取第一幅图像以获取图像尺寸
    first_image = cv2.imread(image_files[0])

    height, width, layers = first_image.shape

    # 设置输出视频文件的名称和编解码器（根据需要进行更改）
    output_video_file = '/data/zjt/HandGestureDataset_SHREC2017/gesture/output_video/output_video.avi'
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # 使用MP4编解码器

    # 创建视频写入对象
    out = cv2.VideoWriter(output_video_file, fourcc, 30.0, (width, height))

    # 逐一读取图像并将其写入视频
    for i in range(49):
        image_file = input_folder + '{}_depth.png'.format(i)

        img = cv2.imread(image_file)
        out.write(img)

    # 释放视频写入对象和销毁所有窗口
    out.release()
    cv2.destroyAllWindows()

    print(f"已生成视频文件：{output_video_file}")