import os
import rasterio
import numpy as np
from pathlib import Path

def multiband_to_single_bands(input_raster_path, output_dir, band_names=None):
    """
    将多波段栅格数据转换为单波段栅格数据
    
    参数:
        input_raster_path: 输入的多波段栅格文件路径
        output_dir: 输出单波段文件的目录
        band_names: 波段名称列表，如果提供，将用于命名输出文件
    """
    os.makedirs(output_dir, exist_ok=True)
    
    with rasterio.open(input_raster_path) as src:
        meta = src.meta.copy()
        meta.update({
            'count': 1,  # 设置为单波段
            'driver': 'GTiff',
            'dtype': src.dtypes[0]
        })
        
        num_bands = src.count
        
        # 获取波段描述信息
        band_descriptions = []
        for i in range(1, num_bands + 1):
            try:
                desc = src.descriptions[i-1]
                # 如果描述为空或None，则使用默认名称
                band_descriptions.append(desc if desc else f"band_{i}")
            except:
                band_descriptions.append(f"band_{i}")
        
        for band_idx in range(1, num_bands + 1):
            try:
                band_data = src.read(band_idx)
                
                # 使用波段描述作为文件名，移除可能的非法字符
                band_name = band_descriptions[band_idx-1]
                # 提取波段名称中的关键字（例如从"MeanBand1pca_1tif"中提取"Mean"）
                band_name = band_name.split('Band')[0] if 'Band' in band_name else band_name
                band_name = band_name.replace('_1tif', '')
                band_name = "".join(c for c in band_name if c.isalnum() or c in ('_', '-'))
                output_filename = f"{band_name}.tif"
                
                output_path = os.path.join(output_dir, output_filename)
                
                # 确保数据类型正确
                band_data = band_data.astype(meta['dtype'])
                
                with rasterio.open(output_path, 'w', **meta) as dst:
                    dst.write(band_data, 1)
                    try:
                        dst.descriptions = (band_descriptions[band_idx-1],)
                    except:
                        pass
                    
                print(f"已保存波段 {band_idx} ({band_name}) 到: {output_filename}")
                
            except Exception as e:
                print(f"处理波段 {band_idx} 时出错: {str(e)}")

if __name__ == "__main__":
    input_raster = r"F:\tif_features\temp\measures\measures.tif"
    output_directory = r"F:\tif_features\temp\measures_one"
    
    try:
        multiband_to_single_bands(input_raster, output_directory)
    except Exception as e:
        print(f"程序执行出错: {str(e)}")
