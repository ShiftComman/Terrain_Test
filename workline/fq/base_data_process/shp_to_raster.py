from osgeo import gdal, ogr,osr
import numpy as np
def rasterize_vector(vector_path, value_field, reference_tif_path, output_tif_path, apply_mask=False, mask_value=9999):
    """
    将矢量文件转换为栅格，并使其与参考 TIF 文件完全对齐。
    可选择是否对矢量未覆盖但TIF文件有效的区域应用掩码。

    参数:
    vector_path (str): 输入矢量文件的路径
    value_field (str): 用于栅格化的矢量字段名
    reference_tif_path (str): 参考 TIF 文件的路径
    output_tif_path (str): 输出 TIF 文件的路径
    apply_mask (bool): 是否应用掩码，默认为False
    mask_value (int): 如果应用掩码，用于填充的值，默认为999

    返回:
    None
    """
    # 打开参考 TIF 文件
    reference_tif = gdal.Open(reference_tif_path)

    # 获取参考 TIF 文件的信息
    geotransform = reference_tif.GetGeoTransform()
    projection = reference_tif.GetProjection()
    x_size = reference_tif.RasterXSize
    y_size = reference_tif.RasterYSize

    # 如果需要应用掩码，读取参考 TIF 数据
    if apply_mask:
        reference_data = reference_tif.GetRasterBand(1).ReadAsArray()
        reference_no_data = reference_tif.GetRasterBand(1).GetNoDataValue()

    # 创建新的栅格文件，完全匹配参考 TIF 的属性
    driver = gdal.GetDriverByName("GTiff")
    output_raster = driver.Create(output_tif_path, x_size, y_size, 1, gdal.GDT_Float32)
    output_raster.SetGeoTransform(geotransform)
    output_raster.SetProjection(projection)

    # 初始化输出栅格为 NoData 值
    band = output_raster.GetRasterBand(1)
    no_data_value = -9999  # 可以根据需要更改
    band.SetNoDataValue(no_data_value)
    band.Fill(no_data_value)

    # 打开矢量文件
    vector = ogr.Open(vector_path)
    layer = vector.GetLayer()

    # 确保矢量数据使用与栅格相同的坐标系
    source_srs = layer.GetSpatialRef()
    target_srs = osr.SpatialReference(wkt=projection)
    
    if source_srs.IsSame(target_srs) == 0:
        print("警告：矢量数据和栅格数据的坐标系不同。正在进行坐标转换。")
        # 如果坐标系不同，可以在这里添加转换逻辑

    # 使用 GDAL 栅格化函数
    gdal.RasterizeLayer(
        output_raster, 
        [1], 
        layer, 
        options=[f"ATTRIBUTE={value_field}", "ALL_TOUCHED=TRUE"]
    )

    # 读取栅格数据到 NumPy 数组
    raster_data = band.ReadAsArray()

    # 如果需要应用掩码
    if apply_mask:
        # 将矢量未覆盖但TIF有效的区域设置为掩码值
        mask = (raster_data == no_data_value) & (reference_data != reference_no_data)
        raster_data[mask] = mask_value

    # 将处理后的数据写回栅格
    band.WriteArray(raster_data)

    # 计算统计信息（可选）
    band.ComputeStatistics(False)

    # 清理
    output_raster = None
    reference_tif = None
    vector = None

    print(f"栅格化完成。输出文件保存在: {output_tif_path}")

if __name__ == "__main__":
    vector_path = r'F:\cache_data\shp_file\fq\fq_sd_polygon.shp'  # 矢量文件路径
    value_field = 'DLDM'  # 用于栅格化的矢量字段名
    reference_tif_path = r'F:\tif_features\county_feature\fq\dem.tif'  # 参考 TIF 文件路径
    output_tif_path = r'F:\tif_features\county_feature\fq\dl.tif' # 输出 TIF 文件路径
    rasterize_vector(vector_path, value_field, reference_tif_path, output_tif_path)