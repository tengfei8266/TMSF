#定义几个图像处理的类，用于数据增强，提升深度学习的泛化能力
import random
import numpy as np
import torch
from skimage import transform
# from PIL import Image, ImageFilter, ImageEnhance
random.seed(1997)


class Compose():        #用于组合做个操作()ops,按顺序输入高分辨率图像、低分辨率图像和特征
    def __init__(self, *ops):
        self.ops = ops

    def __call__(self, hr, lr, feats):
        for op in self.ops:
            hr, lr, feats = op(hr, lr, feats)
        return hr, lr, feats


class ToTensor(object):  #将图像数据转换为PyTorch张量格式，这是深度学习模型所需的输入格式
    def __init__(self, mode: str = "train") -> None:
        self.mode = mode
        pass

    def __call__(self, hr, lr, feats):
        return self.__to_tensor__(hr), self.__to_tensor__(lr), [self.__to_tensor__(feat) for feat in feats]

    def __to_tensor__(self, input):
        input = input[:, :, np.newaxis].astype(np.float32)
        input = torch.from_numpy(input)
        input = input.permute(2, 0, 1)

        return input


class RandomScaleCrop(object):   #随机裁剪操作
    '''
    scale = [1,1,1,1,1.5,1.5,2,2.5]
    '''

    def __init__(self, crop_size: int = 96):
        self.crop_size = crop_size

    def __call__(self, hr, lr):
        crop_size = self.crop_size
        # Get the dimensions of the HR array
        hr_height, hr_width = hr.shape[:2]

        # Calculate the maximum starting indices for random cropping
        max_hr_start_y = hr_height - crop_size
        max_hr_start_x = hr_width - crop_size

        # Generate random starting indices for cropping HR array
        hr_start_y = random.randint(0, max_hr_start_y)
        hr_start_x = random.randint(0, max_hr_start_x)

        # Calculate the corresponding starting indices for cropping LR array
        lr_start_y = int(hr_start_y // (hr_height / lr.shape[0]))
        lr_start_x = int(hr_start_x // (hr_width / lr.shape[1]))

        # Crop the HR and LR arrays
        cropped_hr = hr[hr_start_y:hr_start_y +
                        crop_size, hr_start_x:hr_start_x+crop_size]
        cropped_lr = lr[lr_start_y:lr_start_y+int(crop_size//(
            hr_height / lr.shape[0])), lr_start_x:lr_start_x+int(crop_size//(hr_width / lr.shape[1]))]

        return cropped_hr, cropped_lr


class RandomHorizontalFlip(object):  #随机水平翻转操作
    def __call__(self, hr, lr, feats):
        if random.random() < 0.5:
            hr = hr[:, ::-1].copy()
            lr = lr[:, ::-1].copy()
            feats = [feat[:, ::-1].copy() for feat in feats]

        return hr, lr, feats


class RandomVerticalFlip(object):  #随机垂直旋转操作
    def __call__(self, hr, lr, feats):
        if random.random() < 0.5:
            hr = hr[::-1, :].copy()
            lr = lr[::-1, :].copy()
            feats = [feat[::-1, :].copy() for feat in feats]
        return hr, lr, feats


class RandomRotation(object):   #随机旋转操作
    def __init__(self, rotation_lists=[0, 90, 180, 270]):
        self.rotation_lists = rotation_lists

    def __call__(self, hr, lr):
        rotation = random.choice(self.rotation_lists)
        hr = np.rot90(hr, rotation // 90)
        lr = np.rot90(lr, rotation // 90)

        return hr, lr


if __name__ == "__main__":
    rotation = random.choice([0, 90, 180, 270])
    hr = np.random.rand(192, 192)
    lr = np.random.rand(48, 48)
    hr = transform.rotate(hr, rotation)  # 会进行插值，不建议使用
    lr = transform.rotate(lr, rotation)
