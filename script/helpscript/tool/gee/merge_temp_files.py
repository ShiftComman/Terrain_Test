import rasterio
import os
from rasterio.merge import merge
from rasterio.warp import calculate_default_transform, reproject, Resampling
import logging
from tqdm import tqdm
from pathlib import Path
import numpy as np
import time

def merge_two_rasters(raster1_path, raster2_path, output_path, logger):
    """使用分块方式合并两个栅格文件"""
    try:
        with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN='EMPTY_DIR',
                         GDAL_TIFF_INTERNAL_MASK=True,
                         GDAL_PAM_ENABLED=False,
                         CPL_DEBUG=False):
            
            with rasterio.open(raster1_path) as src1, rasterio.open(raster2_path) as src2:
                # 获取输出范围
                bounds = [src1.bounds, src2.bounds]
                left = min([bound.left for bound in bounds])
                bottom = min([bound.bottom for bound in bounds])
                right = max([bound.right for bound in bounds])
                top = max([bound.top for bound in bounds])
                
                # 计算输出transform
                resolution = src1.transform[0]  # 假设两个文件分辨率相同
                width = int((right - left) / resolution)
                height = int((top - bottom) / resolution)
                out_transform = rasterio.transform.from_bounds(left, bottom, right, top, width, height)
                
                # 准备输出文件的元数据
                meta = src1.meta.copy()
                meta.update({
                    "driver": "GTiff",
                    "height": height,
                    "width": width,
                    "transform": out_transform,
                    "compress": "DEFLATE",
                    "predictor": 2,
                    "zlevel": 6,
                    "tiled": True,
                    "blockxsize": 256,
                    "blockysize": 256
                })
                
                # 创建输出文件
                with rasterio.open(output_path, "w", **meta) as dest:
                    # 设置分块大小
                    block_size = 2048
                    
                    # 计算需要处理的块数
                    x_blocks = int(np.ceil(width / block_size))
                    y_blocks = int(np.ceil(height / block_size))
                    total_blocks = x_blocks * y_blocks
                    
                    with tqdm(total=total_blocks, desc="合并进度") as pbar:
                        for y in range(0, height, block_size):
                            y_size = min(block_size, height - y)
                            for x in range(0, width, block_size):
                                x_size = min(block_size, width - x)
                                
                                # 为当前块创建窗口
                                window = rasterio.windows.Window(x, y, x_size, y_size)
                                
                                try:
                                    # 读取第一个文件的数据
                                    data1 = src1.read(window=window)
                                    if data1.size == 0:
                                        data1 = np.full((meta['count'], y_size, x_size), meta['nodata'])
                                    
                                    # 读取第二个文件的数据
                                    data2 = src2.read(window=window)
                                    if data2.size == 0:
                                        data2 = np.full((meta['count'], y_size, x_size), meta['nodata'])
                                    
                                    # 确保数据形状一致
                                    if data1.shape != data2.shape:
                                        # 调整数据形状以匹配
                                        target_shape = (meta['count'], y_size, x_size)
                                        if data1.shape != target_shape:
                                            data1 = np.full(target_shape, meta['nodata'])
                                        if data2.shape != target_shape:
                                            data2 = np.full(target_shape, meta['nodata'])
                                    
                                    # 合并数据
                                    mask1 = data1 != meta['nodata']
                                    merged_data = np.copy(data2)  # 创建副本避免修改原始数据
                                    merged_data[mask1] = data1[mask1]
                                    
                                    # 写入数据
                                    dest.write(merged_data, window=window)
                                    
                                except rasterio.errors.RasterioIOError as e:
                                    logger.warning(f"读取窗口数据时出错: {str(e)}")
                                    # 如果读取失败，填充nodata
                                    merged_data = np.full((meta['count'], y_size, x_size), meta['nodata'])
                                    dest.write(merged_data, window=window)
                                
                                pbar.update(1)
                                
    except Exception as e:
        logger.error(f"合并过程中发生错误: {str(e)}")
        raise

def reproject_raster(input_path, output_path, target_crs, logger):
    """重投影单个栅格文件"""
    with rasterio.open(input_path) as src:
        # 计算新的变换和尺寸
        dst_crs = rasterio.crs.CRS.from_string(target_crs)
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds)
        
        # 更新元数据
        meta = src.meta.copy()
        meta.update({
            'crs': dst_crs,
            'transform': transform,
            'width': width,
            'height': height
        })
        
        with rasterio.open(output_path, 'w', **meta) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.nearest)

def merge_temp_files(temp_files, output_path, target_crs, logger):
    """合并临时文件"""
    try:
        temp_outputs = temp_files.copy()
        output_dir = os.path.dirname(output_path)
        temp_dir = os.path.join(output_dir, 'merge_temp')
        os.makedirs(temp_dir, exist_ok=True)
        
        total_iterations = len(temp_files).bit_length()  # 计算需要的合并轮数
        logger.info(f"总计需要 {total_iterations} 轮合并")
        
        # 递归合并临时文件
        logger.info("开始合并临时文件...")
        current_iteration = 1
        
        while len(temp_outputs) > 1:
            new_temp_outputs = []
            logger.info(f"开始第 {current_iteration}/{total_iterations} 轮合并")
            
            # 创建当前轮次的进度条
            pairs = len(temp_outputs) // 2 + (len(temp_outputs) % 2)
            pbar = tqdm(total=pairs, 
                       desc=f"第 {current_iteration} 轮合并进度",
                       unit="对",
                       position=0,
                       leave=True)
            
            for i in range(0, len(temp_outputs), 2):
                if i + 1 < len(temp_outputs):
                    # 合并两个临时文件
                    merged_output = os.path.join(temp_dir, f'merged_{len(new_temp_outputs)}.tif')
                    
                    # 如果文件已存在，先尝试删除
                    if os.path.exists(merged_output):
                        try:
                            # 确保文件没有被其他进程占用
                            with open(merged_output, 'a'):
                                os.remove(merged_output)
                        except (PermissionError, OSError) as e:
                            # 如果无法删除，使用一个新的文件名
                            base, ext = os.path.splitext(merged_output)
                            merged_output = f"{base}_{int(time.time())}{ext}"
                    
                    logger.info(f"合并临时文件 {os.path.basename(temp_outputs[i])} 和 "
                              f"{os.path.basename(temp_outputs[i+1])}")
                    merge_two_rasters(temp_outputs[i], temp_outputs[i+1], merged_output, logger)
                    new_temp_outputs.append(merged_output)
                else:
                    # 如果是单个文件，直接加入下一轮
                    new_temp_outputs.append(temp_outputs[i])
                pbar.update(1)
            
            pbar.close()
            temp_outputs = new_temp_outputs
            logger.info(f"第 {current_iteration} 轮合并完成，当前剩余文件数：{len(temp_outputs)}")
            current_iteration += 1
        
        # 最后一个临时文件就是最终结果
        if temp_outputs:
            if target_crs:
                # 对最终结果进行投影转换
                logger.info(f"正在进行投影转换到 {target_crs}...")
                with tqdm(total=1, desc="投影转换进度", unit="文件") as pbar:
                    reproject_raster(temp_outputs[0], output_path, target_crs, logger)
                    pbar.update(1)
            else:
                # 直接复制到目标位置
                logger.info("正在生成最终输出文件...")
                with tqdm(total=1, desc="生成最终文件", unit="文件") as pbar:
                    with rasterio.open(temp_outputs[0]) as src:
                        with rasterio.open(output_path, 'w', **src.meta) as dst:
                            dst.write(src.read())
                    pbar.update(1)
        
        logger.info(f"处理完成，最终输出文件：{output_path}")
        
    except Exception as e:
        logger.error(f"合并过程中发生错误: {str(e)}")
        raise
    finally:
        # 清理临时合并文件
        logger.info("清理临时合并文件...")
        if os.path.exists(temp_dir):
            with tqdm(os.listdir(temp_dir), desc="清理临时文件", unit="文件") as pbar:
                for f in pbar:
                    try:
                        file_path = os.path.join(temp_dir, f)
                        # 确保文件已经关闭
                        with open(file_path, 'a'):
                            pass
                        os.remove(file_path)
                    except Exception as e:
                        logger.warning(f"删除临时文件时出错: {str(e)}")
            try:
                os.rmdir(temp_dir)
            except Exception as e:
                logger.warning(f"删除临时目录时出错: {str(e)}")

def main(temp_path, output_path):
    # 配置日志
    log_file = r'E:\GuiZhouProvinces\logs\merge_temp.log'
    log_dir = Path(log_file).parent
    log_dir.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        encoding='utf-8'
    )
    logger = logging.getLogger(__name__)
    
    # 临时文件列表（请替换为实际的文件路径）
    temp_files = [os.path.join(temp_path, f) for f in os.listdir(temp_path) if f.endswith('.tif')]
    
    target_crs = 'EPSG:4326'  # 如果不需要投影转换，设置为 None
    
    logger.info("开始处理临时文件合并")
    try:
        merge_temp_files(temp_files, output_path, target_crs, logger)
    except Exception as e:
        logger.error(f"处理过程中发生错误: {str(e)}")
        raise
    finally:
        logger.info("处理完成")

if __name__ == "__main__":
    temp_path = r'E:\GuiZhouProvinces\s2\temp_mosaic'
    output_path = r'E:\GuiZhouProvinces\s2_merge\final_mosaic.tif'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    main(temp_path, output_path) 