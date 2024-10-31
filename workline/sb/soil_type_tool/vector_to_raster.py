import rasterio
import rasterio.features
from rasterio.warp import transform_geom, calculate_default_transform, reproject, Resampling
import fiona
import numpy as np
from shapely.geometry import shape, mapping


def vector_to_aligned_raster(input_vector, reference_raster, output_raster, value_field=None, nodata_value=None):
    """
    将矢量转换为与参考栅格对齐的栅格，并用指定值填充无数据区域
    
    参数:
    input_vector: 输入矢量文件路径
    reference_raster: 参考栅格文件路径
    output_raster: 输出栅格文件路径
    value_field: 用于赋值的字段名，如果为None则所有区域赋值为1
    nodata_value: 无数据区域的填充值
    """
    try:
        # 读取参考栅格
        with rasterio.open(reference_raster) as ref_src:
            # 获取参考栅格的元数据
            ref_meta = ref_src.meta.copy()
            ref_crs = ref_src.crs
            
            # 读取矢量数据
            with fiona.open(input_vector) as src_vector:
                # 检查是否需要进行投影转换
                vector_crs = src_vector.crs
                
                geometries = []
                values = []
                
                # 处理每个要素
                for feature in src_vector:
                    geom = shape(feature['geometry'])
                    
                    # 如果坐标系不同，进行投影转换
                    if vector_crs != ref_crs:
                        geom = shape(transform_geom(
                            vector_crs,
                            ref_crs,
                            mapping(geom)
                        ))
                    
                    geometries.append(geom)
                    
                    # 获取栅格化的值
                    if value_field:
                        value = feature['properties'][value_field]
                        # 尝试将值转换为浮点数
                        try:
                            value = float(value)
                        except ValueError:
                            # 如果无法转换为浮点数，使用唯一的整数标识符
                            value = hash(value) % (2**32)
                    else:
                        value = 1
                    values.append(value)
            
            # 创建输出栅格
            ref_meta.update({
                'dtype': 'float64',  # 使用 float64 以适应更大范围的值
                'nodata': float(nodata_value) if nodata_value is not None else 0.0
            })
            
            with rasterio.open(output_raster, 'w', **ref_meta) as dst:
                # 创建初始栅格，填充无数据值
                if nodata_value is not None:
                    init_raster = np.full((ref_meta['height'], ref_meta['width']), 
                                        nodata_value, 
                                        dtype=ref_meta['dtype'])
                else:
                    init_raster = np.zeros((ref_meta['height'], ref_meta['width']), 
                                         dtype=ref_meta['dtype'])
                
                # 栅格化矢量数据
                burned = rasterio.features.rasterize(
                    zip(geometries, values),
                    out_shape=(ref_meta['height'], ref_meta['width']),
                    transform=ref_meta['transform'],
                    dtype=ref_meta['dtype']
                )
                
                # 将栅格化结果写入文件
                dst.write(burned, 1)
        
        print("转换完成！")
        print(f"输出栅格: {output_raster}")
        
    except Exception as e:
        print(f"发生错误: {str(e)}")
        raise
        

# 使用示例：
if __name__ == "__main__":
    # 设置输入参数
    input_vector = r"F:\cache_data\shp_file\sb\sb_sd_dltb.shp"  # 输入矢量路径
    reference_raster = r"F:\tif_features\county_feature\sb\dem.tif"  # 参考栅格路径
    output_raster = r"F:\tif_features\county_feature\sb\dl.tif"  # 输出栅格路径
    value_field = "DLDMN"  # 可选：用于赋值的字段名
    nodata_value = 9999  # 无数据区域的填充值
    # 执行转换
    vector_to_aligned_raster(
        input_vector,
        reference_raster,
        output_raster,
        value_field,  # 如果不需要指定字段，可以省略此参数
        nodata_value  # 如果不需要指定无数据值，可以省略此参数
    )
