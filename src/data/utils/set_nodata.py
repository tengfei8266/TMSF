import numpy as np
from osgeo import gdal

path = r"G:\Transfer\DEM_SAR\Predict\Origin\HR\1_ICHighlyDEM1.tif"

out_path = r"D:\Dem\exper_2\Origin\HR\nodata\void_1_unproj.tif"

dataset = gdal.Open(path)
proj = dataset.GetProjection()
transf = dataset.GetGeoTransform()

data = dataset.GetRasterBand(1)

nodata = data.GetNoDataValue()

arr = data.ReadAsArray()

new_arr = np.nan_to_num(arr, nan=-9999)

driver = gdal.GetDriverByName('GTiff')
save_tif = driver.Create(out_path, 6600, 6600, 1, gdal.GDT_Float64)
save_tif.SetGeoTransform(transf)
save_tif.SetProjection(proj)
save_tif.GetRasterBand(1).WriteArray(new_arr)
save_tif.GetRasterBand(1).SetNoDataValue(-9999)
save_tif.FlushCache()
save_tif = None


a = 1