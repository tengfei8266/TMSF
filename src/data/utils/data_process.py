'''
@File    :   __init__.py
@Time    :   2024/06/10 11:22:40
@Author  :   ztf 
@Contact :   tengfeizhang@whu.edu.cn
@License :   (C)Copyright 2024, Wuhan University
@Desc    :   None
'''

# here put the import lib
import os
os.environ['PROJ_LIB'] = "/home/ztf/anaconda3/envs/ztf/share/proj"
import random
from osgeo import gdal
gdal.DontUseExceptions()

class DataProcess():
    """a class for pre-processing data containing some methods
    """

    def __init__(self) -> None:
        pass

    def crop(self, input_path: str, output_path: str):
        """crop complete images to some patches

        Args:
            input_path (str): complete image path.
            output_path (str): output path of patches.
        """
        # read domain tiff data
        dataset = gdal.Open(input_path)  # X：30720 Y：15360
        img_array = dataset.ReadAsArray()
        proj = dataset.GetProjection()
        transf = (0.0, 2.0, 0.0, 0.0, 0.0, -2.0)
        driver = gdal.GetDriverByName("GTiff")

        # target area
        hr_array = img_array[7680:, 15360:30720]
        hr = driver.Create(output_path, 15360, 7680, 1, gdal.GDT_Float32)
        hr.SetGeoTransform(transf)
        hr.SetProjection(proj)
        hr.GetRasterBand(1).WriteArray(hr_array)
        del hr


def tiles_generate(input_path: str, mode: str, scale: int, out_dir: str, start_index: int = 1) -> int:
    """crop image to tiles

    Args:
        input_path (str): img path.
        mode (str): crop size (HR:192, LR:192//scale).
        scale (int): scale for super-resolution.
        out_dir (str): output directory of tiles.
        start_index (int, optional): start index of tiles. Defaults to 1.
    """
    dataset = gdal.Open(input_path)
    img_array = dataset.ReadAsArray()
    proj = dataset.GetProjection()
    transf = dataset.GetGeoTransform()
    driver = gdal.GetDriverByName("GTiff")
    print("current img to crop:", input_path, "mode:", mode)
    # save path
    if not os.path.exists(os.path.join(out_dir)):
        os.makedirs(os.path.join(out_dir))
    height, width = img_array.shape
    crop_size = 192//scale if mode == "LR" else 192
    index = start_index
    prefix = "Dem_" + str(crop_size) + "_" + str(crop_size) + "_"
    for h in range(0, height, crop_size):
        for w in range(0, width, crop_size):
            curr_index = str(index).zfill(5)
            index += 1
            curr_img = img_array[h:h+crop_size, w:w+crop_size]
            img = driver.Create(os.path.join(
                out_dir, prefix+curr_index+".tif"), crop_size, crop_size, 1, gdal.GDT_Float32)
            curr_transf = (transf[0]+w*transf[1], transf[1], 0.0,
                           transf[3]-h*transf[1], 0.0, -transf[1])
            img.SetGeoTransform(curr_transf)
            img.SetProjection(proj)
            img.GetRasterBand(1).WriteArray(curr_img)
            del img
    return index


def pair_generate(scale: int, base_dir: str, seed: int = 27, ):
    """generate pairs for training

    Args:
        scale (int): scale for super-resolution.
        base_dir (str): path of patches.
        seed (int, optional): random seed. Defaults to 27.
    """
    random.seed(seed)

    hr_dir = os.path.join(base_dir, 'HR')
    name_lists_hr = sorted(
        [name for name in os.listdir(hr_dir) if name.endswith(".tif")])
    # name_lists_hr = random.sample(name_lists_hr, 5000)
    lr_dir = os.path.join(base_dir, 'LR')
    # name_lists_lr = sorted(
    #     [name for name in os.listdir(lr_dir) if name.endswith(".TIF")])
    total_num = len(name_lists_hr)
    hr_lists = random.sample(name_lists_hr, total_num)
    replace_size = 192//scale
    lr_lists = [h.replace('Dem_192_192_', 'Dem_'+str(replace_size) +
                          '_'+str(replace_size)+'_') for h in hr_lists]

    # feats
    feat_dir = os.path.join(base_dir, 'Feats')
    feats = []
    for f in os.scandir(feat_dir):
        if not f.is_dir():
            continue
        cur_feat = [os.path.join(f.path, i) for i in hr_lists]
        feats.append(cur_feat)

    # HR-LR pairs for training SR model
    dataset_list = []
    for hr, lr, *sub_feats in zip(hr_lists, lr_lists, *feats):
        dataset_list.append(os.path.join(hr_dir, hr) + " " +
                            os.path.join(lr_dir, lr) + " " + " ".join(sub_feats) + "\n")
    train_list = random.sample(dataset_list, int(0.8*total_num))
    val_list = [i for i in dataset_list if i not in train_list]
    print("num of Train_dataset:", len(train_list))
    print("num of Val_dataset:", len(val_list))

    with open(os.path.join(base_dir, "train_" + str(scale) + "x.txt"), "w") as f:
        for name in train_list:
            f.write(name)

    with open(os.path.join(base_dir, "val_" + str(scale) + "x.txt"), "w") as f:
        for name in val_list:
            f.write(name)


if __name__ == "__main__":
    # tiles generate
    base_dir = "./Data/SAR"
    for root, dirs, files in os.walk(os.path.join(base_dir, "Origin")):
        if len(dirs) or not len(files):
            continue
        start_index = 1
        for file in files:
            if not file.endswith(".tif"):
                continue
            # 相当于把Origin里的文件结构复制到了base_dir
            out_dir = root.replace('Origin/', '')
            mode = root.split('/')[-1]
            print("current img to crop:", file, "mode:", mode)
            start_index = tiles_generate(os.path.join(
            root, file), mode=mode, scale=4, out_dir=out_dir, start_index=start_index)

    # for mode in ["HR", "LR"]:
    #     start_index = 1
    #     cur_dir = os.path.join(base_dir, mode + "_Origin")
    #     for file in os.listdir(cur_dir):
    #         if not file.endswith(".tif"):
    #             continue
    #         start_index = tiles_generate(os.path.join(
    #             cur_dir, file), mode=mode, scale=4, out_dir=base_dir, start_index=start_index)
    # pairs generate
    pair_generate(scale=4, base_dir=base_dir)
