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





