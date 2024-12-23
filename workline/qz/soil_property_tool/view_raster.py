import rasterio
import numpy as np
from rasterio.windows import Window
from scipy import ndimage
from enum import Enum

class OutlierMethod(Enum):
    """异常值处理方法枚举"""
    TRUNCATE = 'truncate'           # 截断到阈值
    MEAN = 'mean'                   # 全局均值替换
    MEDIAN = 'median'               # 全局中值替换
    LOCAL_MEAN = 'local_mean'       # 局部均值替换
    LOCAL_MEDIAN = 'local_median'   # 局部中值替换
    IDW = 'idw'                     # 反距离加权插值

def process_outliers(data_chunk, outliers_mask, valid_mask, method, window_size, 
                    upper_bound, lower_bound, pad=None):
    """
    使用指定方法处理异常值
    
    参数:
        data_chunk: 数据块
        outliers_mask: 异常值掩码
        valid_mask: 有效值掩码
        method: 处理方法
        window_size: 邻域窗口大小
        upper_bound: 上限阈值
        lower_bound: 下限阈值
        pad: padding大小（用于局部方法）
    """
    processed_chunk = data_chunk.copy()
    
    if method == OutlierMethod.TRUNCATE:
        # 截断法：直接将超出范围的值设为阈值
        processed_chunk[data_chunk > upper_bound] = upper_bound
        processed_chunk[data_chunk < lower_bound] = lower_bound
        
    elif method == OutlierMethod.MEAN:
        # 全局均值替换：使用所有有效非异常值的均值
        valid_data = data_chunk[(valid_mask) & ~(outliers_mask)]
        if len(valid_data) > 0:
            mean_val = np.mean(valid_data)
            processed_chunk[outliers_mask] = mean_val
            
    elif method == OutlierMethod.MEDIAN:
        # 全局中值替换：使用所有有效非异常值的中值
        valid_data = data_chunk[(valid_mask) & ~(outliers_mask)]
        if len(valid_data) > 0:
            median_val = np.median(valid_data)
            processed_chunk[outliers_mask] = median_val
            
    elif method in [OutlierMethod.LOCAL_MEAN, OutlierMethod.LOCAL_MEDIAN]:
        # 局部统计替换
        y_indices, x_indices = np.where(outliers_mask)
        for yi, xi in zip(y_indices, x_indices):
            # 提取邻域
            y_start = max(0, yi - pad)
            y_end = min(data_chunk.shape[0], yi + pad + 1)
            x_start = max(0, xi - pad)
            x_end = min(data_chunk.shape[1], xi + pad + 1)
            
            neighborhood = data_chunk[y_start:y_end, x_start:x_end]
            neighborhood_valid = neighborhood[valid_mask[y_start:y_end, x_start:x_end]]
            
            # 排除邻域中的异常值
            valid_neighbors = neighborhood_valid[
                (neighborhood_valid >= lower_bound) & 
                (neighborhood_valid <= upper_bound)
            ]
            
            if len(valid_neighbors) > 0:
                if method == OutlierMethod.LOCAL_MEAN:
                    processed_chunk[yi, xi] = np.mean(valid_neighbors)
                else:  # LOCAL_MEDIAN
                    processed_chunk[yi, xi] = np.median(valid_neighbors)
            else:
                # 如果没有有效邻域值，使用截断值
                processed_chunk[yi, xi] = (upper_bound if data_chunk[yi, xi] > upper_bound 
                                         else lower_bound)
                
    elif method == OutlierMethod.IDW:
        # 反距离加权插值
        y_indices, x_indices = np.where(outliers_mask)
        for yi, xi in zip(y_indices, x_indices):
            y_start = max(0, yi - pad)
            y_end = min(data_chunk.shape[0], yi + pad + 1)
            x_start = max(0, xi - pad)
            x_end = min(data_chunk.shape[1], xi + pad + 1)
            
            # 提取邻域
            neighborhood = data_chunk[y_start:y_end, x_start:x_end]
            neighborhood_valid = valid_mask[y_start:y_end, x_start:x_end]
            neighborhood_normal = ((neighborhood >= lower_bound) & 
                                (neighborhood <= upper_bound))
            
            # 获取有效点的位置和值
            valid_points = neighborhood_valid & neighborhood_normal
            if np.any(valid_points):
                y_local, x_local = np.where(valid_points)
                values = neighborhood[valid_points]
                
                # 计算距离权重
                distances = np.sqrt((y_local - (yi-y_start))**2 + 
                                 (x_local - (xi-x_start))**2)
                weights = 1 / (distances + 1e-6)  # 避免除零
                weights = weights / np.sum(weights)
                
                # 计算加权平均
                processed_chunk[yi, xi] = np.sum(values * weights)
            else:
                # 如果没有有效邻域值，使用截断值
                processed_chunk[yi, xi] = (upper_bound if data_chunk[yi, xi] > upper_bound 
                                         else lower_bound)
    
    return processed_chunk

def analyze_raster(input_raster_path, threshold=3, process=False, output_path=None, 
                  window_size=3, method=OutlierMethod.LOCAL_MEAN):
    """
    分析或处理栅格数据的异常值
    
    参数:
        input_raster_path: 输入栅格文件路径
        threshold: 异常值判定的标准差倍数，默认为3
        process: 是否处理异常值，默认False
        output_path: 输出栅格文件路径（仅在process=True时需要）
        window_size: 邻域窗口大小，默认3x3
        method: 异常值处理方法，默认使用局部均值法
    """
    with rasterio.open(input_raster_path) as src:
        # 获取数据基本信息
        height = src.height
        width = src.width
        meta = src.meta.copy()
        
        # 使用rasterio的statistics方法获取统计信息
        stats = src.statistics(1)
        mean_val = stats.mean
        std_val = stats.std
        min_val = stats.min
        max_val = stats.max
        
        # 计算异常值的阈值
        upper_bound = mean_val + threshold * std_val
        lower_bound = mean_val - threshold * std_val
        
        # 统计异常值
        total_outliers = 0
        total_valid_pixels = 0
        
        # 分块处理并统计，使用更大的块以处理边界
        chunk_size = 1024
        pad = window_size // 2  # 邻域半径
        
        if process and output_path:
            dst = rasterio.open(output_path, 'w', **meta)
        
        for y in range(0, height, chunk_size):
            y_size = min(chunk_size, height - y)
            for x in range(0, width, chunk_size):
                x_size = min(chunk_size, width - x)
                
                # 读取带padding的数据块
                read_window = Window(
                    max(0, x - pad),
                    max(0, y - pad),
                    min(width - max(0, x - pad), x_size + 2 * pad),
                    min(height - max(0, y - pad), y_size + 2 * pad)
                )
                data_chunk = src.read(1, window=read_window)
                
                # 统计有效像元和异常值
                valid_mask = ~np.isnan(data_chunk)
                if src.nodata is not None:
                    valid_mask &= (data_chunk != src.nodata)
                
                # 找出当前块中的异常值
                outliers_mask = (data_chunk > upper_bound) | (data_chunk < lower_bound)
                outliers_mask &= valid_mask
                
                # 统计异常值
                chunk_valid_data = data_chunk[valid_mask]
                total_outliers += np.sum(outliers_mask)
                total_valid_pixels += len(chunk_valid_data)
                
                # 如果需要处理异常值
                if process and output_path:
                    # 创建输出数据
                    processed_chunk = process_outliers(
                        data_chunk, outliers_mask, valid_mask, 
                        method, window_size, upper_bound, lower_bound, pad
                    )
                    
                    # 去除padding，写入结果
                    write_window = Window(
                        x,
                        y,
                        x_size,
                        y_size
                    )
                    out_chunk = processed_chunk[
                        pad if y > 0 else 0:pad + y_size if y + y_size < height else y_size,
                        pad if x > 0 else 0:pad + x_size if x + x_size < width else x_size
                    ]
                    dst.write(out_chunk, 1, window=write_window)
        
        if process and output_path:
            dst.close()
        
        # 计算比例
        outlier_ratio = (total_outliers / total_valid_pixels) * 100 if total_valid_pixels > 0 else 0
        
        print(f"\n统计信息:")
        print(f"均值: {mean_val:.2f}")
        print(f"标准差: {std_val:.2f}")
        print(f"最小值: {min_val:.2f}")
        print(f"最大值: {max_val:.2f}")
        print(f"异常值范围: < {lower_bound:.2f} 或 > {upper_bound:.2f}")
        print(f"\n异常值统计:")
        print(f"异常值像元数: {total_outliers}")
        print(f"有效像元总数: {total_valid_pixels}")
        print(f"异常值占比: {outlier_ratio:.2f}%")
        
        if process and output_path:
            print(f"\n已保存处理后的栅格到: {output_path}")

if __name__ == "__main__":
    # 示例用法
    input_path = r"F:\tif_features\county_feature\qz\dem.tif"
    output_path = r"F:\tif_features\county_feature\qz\dem_cleaned.tif"
    
    # 只查看统计信息
    analyze_raster(input_path)
    
    # 使用不同方法处理异常值
    # 1. 截断法
    # analyze_raster(input_path, process=True, output_path=output_path, 
    #               method=OutlierMethod.TRUNCATE)
    
    # 2. 全局均值替换
    # analyze_raster(input_path, process=True, output_path=output_path, 
    #               method=OutlierMethod.MEAN)
    
    # 3. 局部均值替换（默认）
    # analyze_raster(input_path, process=True, output_path=output_path, 
    #               method=OutlierMethod.LOCAL_MEAN, window_size=3)
    
    # 4. 局部中值替换
    # analyze_raster(input_path, process=True, output_path=output_path, 
    #               method=OutlierMethod.LOCAL_MEDIAN, window_size=5)
    
    # 5. IDW插值
    # analyze_raster(input_path, process=True, output_path=output_path, 
    #               method=OutlierMethod.IDW, window_size=5)
