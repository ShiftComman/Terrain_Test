from osgeo import gdal, ogr, osr
import numpy as np

def rasterize_vector(vector_path, value_field, reference_tif_path, output_tif_path):
    """
    将矢量文件转换为栅格，通过直接采样的方式使其与参考TIF完全对齐
    """
    # 打开参考TIF和矢量文件
    reference_tif = gdal.Open(reference_tif_path)
    vector_ds = ogr.Open(vector_path)
    layer = vector_ds.GetLayer()

    # 获取参考栅格信息
    geotransform = reference_tif.GetGeoTransform()
    projection = reference_tif.GetProjection()
    x_size = reference_tif.RasterXSize
    y_size = reference_tif.RasterYSize

    # 创建输出栅格
    driver = gdal.GetDriverByName("GTiff")
    output_raster = driver.Create(output_tif_path, x_size, y_size, 1, gdal.GDT_Float32)
    output_raster.SetGeoTransform(geotransform)
    output_raster.SetProjection(projection)

    # 初始化输出数组
    output_data = np.full((y_size, x_size), -9999, dtype=np.float32)

    # 创建空间索引
    layer.SetSpatialFilter(None)
    layer.ResetReading()

    # 遍历每个像元中心点
    for y in range(y_size):
        for x in range(x_size):
            # 计算像元中心点坐标
            px = geotransform[0] + (x + 0.5) * geotransform[1]
            py = geotransform[3] + (y + 0.5) * geotransform[5]
            
            # 创建点几何
            point = ogr.Geometry(ogr.wkbPoint)
            point.AddPoint(px, py)
            
            # 设置空间过滤
            layer.SetSpatialFilter(point)
            
            # 获取该点位置的要素值
            feature = layer.GetNextFeature()
            if feature:
                value = feature.GetField(value_field)
                output_data[y, x] = value

    # 写入数据
    band = output_raster.GetRasterBand(1)
    band.SetNoDataValue(-9999)
    band.WriteArray(output_data)
    band.ComputeStatistics(False)

    # 清理
    output_raster = None
    reference_tif = None
    vector_ds = None

    print(f"栅格化完成。输出文件保存在: {output_tif_path}")

if __name__ == "__main__":
    vector_path = r'C:\Users\Runker\Desktop\slope_temp\wc_slope_101_result_smooth_filled.shp'
    value_field = 'gridcode'
    reference_tif_path = r'F:\tif_features\county_feature\wc\dem.tif'
    output_tif_path = r'F:\tif_features\county_feature\wc\slopeclassssss.tif'
    rasterize_vector(vector_path, value_field, reference_tif_path, output_tif_path)