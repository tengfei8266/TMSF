from osgeo import gdal
import os
import numpy as np


base_dir = "./Data/moon"
hr_dir = os.path.join(base_dir, "hr")
lr_dir = os.path.join(base_dir, "lr")
# data_path = "Data/Lunar_global/origin/ldem_875s_5m.lbl"
pixel_values = []
for name in os.listdir(hr_dir):
    if name.endswith(".tif"):
        dataset = gdal.Open(os.path.join(hr_dir,name))
        arr = dataset.GetRasterBand(1).ReadAsArray()
        pixel_values.extend(arr.flatten())
mean = np.mean(pixel_values)
std = np.std(pixel_values)
print(mean, std)
# data_path = "./origin/ldem_1024_75s_60s_030_060.lbl"
# dataset = gdal.Open(data_path)
# driver = gdal.GetDriverByName('GTiff')
# out_path = data_path.split('.lbl')[0].split("ldem_")[-1] + "_polar.tif"
# # out_path = "1024_75_60_030_060.tif"
# save_tif = driver.Create(os.path.join("./processed",out_path),dataset.RasterXSize,dataset.RasterYSize,1,gdal.GDT_Float32)
# save_tif.SetGeoTransform(dataset.GetGeoTransform())
# save_tif.SetProjection(dataset.GetProjection())
# arr = dataset.GetRasterBand(1).ReadAsArray() * 0.5

# save_tif.GetRasterBand(1).WriteArray(arr)

# for 
# 60-30
# -1338.5287042686184 2346.5800338281065
# 40-20
# -1500.5071338188193 2059.089750341637
# 20-10
# -1067.869330247014 1970.9982615790632
# 10-5
# -979.3627856872142 1130.6843199936013
# 5-2.5
# -2253.4084 3844.9346