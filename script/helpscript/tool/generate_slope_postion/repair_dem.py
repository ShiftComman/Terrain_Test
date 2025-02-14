import rasterio
import numpy as np
from rasterio.warp import calculate_default_transform, reproject, Resampling
from scipy.ndimage import uniform_filter
from scipy.ndimage import binary_erosion

# 首先读取小范围高分辨率栅格(5m)来获取目标分辨率的参数
def reproject_raster(small_raster_path, large_raster_path, output_path):
    # 1. 首先读取小栅格来获取目标分辨率的参数
    with rasterio.open(small_raster_path) as small_src:
        small_nodata = small_src.nodata if small_src.nodata is not None else -9999
        
        with rasterio.open(large_raster_path) as large_src:
            large_nodata = large_src.nodata if large_src.nodata is not None else -9999
            
            # 2. 使用小栅格的分辨率计算新的变换参数和尺寸
            dst_transform, dst_width, dst_height = calculate_default_transform(
                large_src.crs,
                large_src.crs,
                large_src.width,
                large_src.height,
                *large_src.bounds,
                resolution=small_src.res
            )
            
            # 3. 创建目标数组
            temp_large = np.full((dst_height, dst_width), large_nodata, dtype='float32')
            temp_small = np.full((dst_height, dst_width), small_nodata, dtype='float32')
            
            # 4. 将大栅格重投影到小栅格的分辨率
            reproject(
                source=rasterio.band(large_src, 1),
                destination=temp_large,
                src_transform=large_src.transform,
                src_crs=large_src.crs,
                dst_transform=dst_transform,
                dst_crs=large_src.crs,
                resampling=Resampling.bilinear,
                init_dest_nodata=False
            )
            
            # 5. 将小栅格重投影到相同的网格
            reproject(
                source=rasterio.band(small_src, 1),
                destination=temp_small,
                src_transform=small_src.transform,
                src_crs=small_src.crs,
                dst_transform=dst_transform,
                dst_crs=large_src.crs,
                resampling=Resampling.bilinear,
                init_dest_nodata=False
            )
            
            # 6. 创建有效值掩码，但排除边缘5个像元
            valid_mask = (temp_small > 0) & \
                        (temp_small != small_nodata) & \
                        (temp_small != large_nodata) & \
                        (np.isfinite(temp_small))
            
            # 创建一个边缘缓冲区掩码
            buffer_size = 5  # 边缘5个像元
            # 先创建一个基础掩码，然后进行腐蚀操作
            base_mask = valid_mask.copy()
            valid_mask = binary_erosion(base_mask, iterations=buffer_size)
            
            # 7. 值替换
            result = temp_large.copy()
            result[valid_mask] = temp_small[valid_mask]

        # 保存结果
        meta = large_src.meta.copy()
        meta.update({
            'dtype': 'float32',
            'nodata': large_nodata,
            'transform': dst_transform,
            'width': dst_width,
            'height': dst_height
        })
        
    with rasterio.open(output_path, 'w', **meta) as dst:
        dst.write(result, 1)
        
if __name__ == "__main__":
    small_raster_path = r'C:\Users\Runker\Desktop\temp\min.tif'
    large_raster_path = r"C:\Users\Runker\Desktop\temp\a_mosaic.tif"
    output_path = r"C:\Users\Runker\Desktop\temp\a_mosaic_new.tif"
    reproject_raster(small_raster_path, large_raster_path, output_path)
