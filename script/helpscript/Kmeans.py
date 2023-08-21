import arcpy
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from arcpy import env
from arcpy.sa import *
import time

work_path = "D:\ArcGISProjects\syraster\SJSG.gdb"
env.workspace = work_path

# 读取DEM数据并转换为NumPy数组
dem_path = "D:\ArcGISProjects\MyProject\LBCL.gdb\SY_DEM_5_DT_3"


slope_path = "D:\ArcGISProjects\MyProject\LBCL.gdb\SY_DEM_5_DT_3_SLOPE"
aspect_path = "D:\ArcGISProjects\MyProject\LBCL.gdb\SY_DEM_5_DT_3_ASP"
curvature_path = "D:\ArcGISProjects\MyProject\LBCL.gdb\SY_DEM_5_DT_3_QL"


# 计算坡度
def get_slope():
    with arcpy.EnvManager(parallelProcessingFactor="0"):
        slope_raster = arcpy.sa.Slope(dem_path, "DEGREE", 1, "PLANAR", "METER")
        slope_raster.save("SY_DEM_5_DT_3_SLOPE")


# 计算坡向
def get_asp():
    with arcpy.EnvManager(parallelProcessingFactor="0"):
        aspect_raster = arcpy.sa.Aspect(
            dem_path, "PLANAR", "METER", "GEODESIC_AZIMUTHS"
        )
        aspect_raster.save("SY_DEM_5_DT_3_ASP")


# 计算曲率
def get_cur():
    with arcpy.EnvManager(parallelProcessingFactor="0"):
        curvature_raster = arcpy.sa.Curvature(dem_path, 1, None, None)
        curvature_raster.save("SY_DEM_5_DT_3_QL")


# get_slope()
# get_asp()
# get_cur()
# print("获取完成")

slope_array = arcpy.RasterToNumPyArray("SY_DEM_5_DT_3_SLOPE")
aspect_array = arcpy.RasterToNumPyArray("SY_DEM_5_DT_3_ASP")
curvature_array = arcpy.RasterToNumPyArray("SY_DEM_5_DT_3_QL")
dem_array = arcpy.RasterToNumPyArray(
    "D:\ArcGISProjects\MyProject\LBCL.gdb\SY_DEM_5_DT_3"
)
# 将DEM、坡度、坡向和曲率数组转换为一维数组，并将它们合并成一个特征数组
dem_flat = dem_array.flatten()
slope_flat = slope_array.flatten()
aspect_flat = aspect_array.flatten()
curvature_flat = curvature_array.flatten()

features = np.column_stack((dem_flat, slope_flat, aspect_flat, curvature_flat))

# 初始化最佳的聚类数和最佳的轮廓系数
best_k = 2
best_score = -1

print("start")
# 遍历不同的聚类数，计算轮廓系数
for k in range(2, 10):
    kmeans = KMeans(n_clusters=k, n_init="auto")
    kmeans.fit(features)
    labels = kmeans.labels_
    score = silhouette_score(features, labels)
    if score > best_score:
        best_k = k
        best_score = score

print("得到最佳")
# 使用最佳的聚类数进行最终的地形分类
kmeans = KMeans(n_clusters=best_k)
kmeans.fit(features)
labels = kmeans.labels_.reshape(dem_array.shape)

# 保存分类结果为栅格数据
out_raster = arcpy.NumPyArrayToRaster(
    labels,
    arcpy.Point(arcpy.env.extent.XMin, arcpy.env.extent.YMin),
    arcpy.env.cellSize,
    arcpy.env.cellSize,
)
out_raster.save("RESULT")
print("完成")
