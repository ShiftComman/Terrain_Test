import rasterio
import numpy as np
from rasterio.merge import merge
from rasterio.warp import calculate_default_transform, reproject, Resampling
import os
from rasterio.enums import ColorInterp
import logging
from tqdm import tqdm
from pathlib import Path


def merge_rasters(raster_path, output_path, logger, target_crs=None):
    raster_paths = [os.path.join(raster_path, f) for f in os.listdir(raster_path) if f.endswith('.tif')]
    
    try:
        with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN='EMPTY_DIR',
                          GDAL_TIFF_INTERNAL_MASK=True,
                          GDAL_PAM_ENABLED=False,
                          CPL_DEBUG=False):
            
            # 打开所有栅格文件
            src_files = [rasterio.open(raster) for raster in raster_paths]
            
            # 获取第一个文件的元数据
            meta = src_files[0].meta.copy()
            
            # 执行合并（使用原始投影）
            logger.info(f"Merging {len(src_files)} rasters in original projection...")
            mosaic, out_trans = merge(src_files)
            
            logger.info(f"Merged raster shape: {mosaic.shape}")
            
            # 更新元数据
            meta.update({
                "driver": "GTiff",
                "height": mosaic.shape[1],
                "width": mosaic.shape[2],
                "transform": out_trans,
                "count": mosaic.shape[0],
                "compress": "DEFLATE",
                "predictor": 2,
                "zlevel": 6
            })
            
            # 如果目标CRS与原始CRS不同，则进行重投影
            if target_crs != meta['crs'].to_string():
                logger.info(f"Reprojecting from {meta['crs'].to_string()} to {target_crs}")
                
                # 计算新的变换和尺寸
                dst_crs = rasterio.crs.CRS.from_string(target_crs)
                dst_transform, dst_width, dst_height = calculate_default_transform(
                    meta['crs'], dst_crs, meta['width'], meta['height'], *rasterio.transform.array_bounds(meta['height'], meta['width'], out_trans)
                )
                
                # 更新元数据
                meta.update({
                    'crs': dst_crs,
                    'transform': dst_transform,
                    'width': dst_width,
                    'height': dst_height
                })
                
                # 创建目标数组
                dst_mosaic = np.zeros((meta['count'], dst_height, dst_width), dtype=meta['dtype'])
                
                # 进行重投影
                for i in tqdm(range(meta['count']), desc="Reprojecting bands"):
                    reproject(
                        source=mosaic[i],
                        destination=dst_mosaic[i],
                        src_transform=out_trans,
                        src_crs=src_files[0].crs,
                        dst_transform=dst_transform,
                        dst_crs=dst_crs,
                        resampling=Resampling.nearest
                    )
                
                mosaic = dst_mosaic
            
            # 写入最终的栅格
            logger.info(f"Writing output to {output_path}")
            with rasterio.open(output_path, "w", **meta) as dest:
                dest.write(mosaic)
                # 设置颜色解释
                color_interps = [ColorInterp.undefined] * meta['count']
                dest.colorinterp = color_interps
        
        logger.info(f"Rasters processed successfully. Output saved to {output_path}")
        logger.info(f"Number of bands in output: {mosaic.shape[0]}")
        logger.info(f"Output CRS: {meta['crs'].to_string()}")
        
    except Exception as e:
        logger.error(f"Error processing rasters: {str(e)}")
        raise
    finally:
        # 关闭所有打开的文件
        for src in src_files:
            src.close()

def natural_sort_key(s):
    """实现自然排序的key函数"""
    import re
    def atoi(text):
        return int(text) if text.isdigit() else text
    return [atoi(c) for c in re.split(r'(\d+)', s)]

def merge_rasters_in_batches(raster_path, output_path, logger, target_crs=None, batch_size=10):
    """分批次合并栅格文件
    
    Args:
        raster_path: 栅格文件所在目录
        output_path: 输出文件路径
        logger: 日志记录器
        target_crs: 目标坐标系
        batch_size: 每批处理的文件数量
    """
    # 使用自然排序
    raster_files = sorted([os.path.join(raster_path, f) for f in os.listdir(raster_path) if f.endswith('.tif')],
                         key=natural_sort_key)
    
    total_files = len(raster_files)
    
    if total_files == 0:
        logger.error("未找到任何TIF文件")
        return
    
    logger.info(f"总共找到 {total_files} 个栅格文件")
    
    # 创建临时文件夹
    temp_dir = os.path.join(os.path.dirname(output_path), 'temp_mosaic')
    os.makedirs(temp_dir, exist_ok=True)
    
    try:
        current_batch = 1
        temp_outputs = []
        
        # 计算总批次数
        total_batches = (total_files + batch_size - 1) // batch_size
        
        # 使用tqdm创建进度条
        batch_progress = tqdm(range(0, total_files, batch_size), 
                            desc="批次处理进度", 
                            total=total_batches,
                            unit="批")
        
        # 分批处理文件
        for i in batch_progress:
            batch_files = raster_files[i:i + batch_size]
            temp_output = os.path.join(temp_dir, f'temp_mosaic_{current_batch}.tif')
            
            batch_progress.set_description(f"处理第 {current_batch}/{total_batches} 批 ({len(batch_files)} 个文件)")
            logger.info(f"处理第 {current_batch} 批 ({len(batch_files)} 个文件)")
            
            with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN='EMPTY_DIR',
                            GDAL_TIFF_INTERNAL_MASK=True,
                            GDAL_PAM_ENABLED=False,
                            CPL_DEBUG=False):
                
                # 打开当前批次的所有文件
                src_files = [rasterio.open(f) for f in batch_files]
                
                # 获取第一个文件的元数据
                meta = src_files[0].meta.copy()
                
                # 执行合并
                mosaic, out_trans = merge(src_files)
                
                # 更新元数据
                meta.update({
                    "driver": "GTiff",
                    "height": mosaic.shape[1],
                    "width": mosaic.shape[2],
                    "transform": out_trans,
                    "count": mosaic.shape[0],
                    "compress": "DEFLATE",
                    "predictor": 2,
                    "zlevel": 6
                })
                
                # 写入临时文件
                with rasterio.open(temp_output, "w", **meta) as dest:
                    dest.write(mosaic)
                
                # 关闭所有打开的文件
                for src in src_files:
                    src.close()
            
            temp_outputs.append(temp_output)
            current_batch += 1
        
        # 合并所有临时文件
        logger.info("合并临时文件...")
        merge_rasters(temp_dir, output_path, logger, target_crs)
        
    finally:
        # 清理临时文件
        logger.info("清理临时文件...")
        pass
        # for temp_file in temp_outputs:
        #     try:
        #         if os.path.exists(temp_file):
        #             os.remove(temp_file)
        #     except Exception as e:
        #         logger.warning(f"删除临时文件时出错: {str(e)}")
        # try:
        #     if os.path.exists(temp_dir):
        #         os.rmdir(temp_dir)
        # except Exception as e:
        #     logger.warning(f"删除临时目录时出错: {str(e)}")

def main(input_dir:str, output_path:str, target_crs:str, log_file:str, batch_size:int=100):
    # 配置日志
    if log_file:
        log_dir = Path(log_file).parent
        log_dir.mkdir(parents=True, exist_ok=True)
        logging.basicConfig(filename=log_file, level=logging.INFO, 
                          format='%(asctime)s - %(levelname)s - %(message)s', encoding='utf-8')
    else:
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', encoding='utf-8')
    
    logger = logging.getLogger(__name__)
    logger.info("开始处理栅格数据")
    try:
        merge_rasters_in_batches(input_dir, output_path, logger, target_crs, batch_size)
    except Exception as e:
        logger.error(f"处理栅格数据时发生错误: {str(e)}")
        raise
    finally:
        logger.info("栅格数据处理完成")

if __name__ == "__main__":
    input_dir = r'E:\GuiZhouProvinces\s2'
    output_path = r'E:\GuiZhouProvinces\s2\mosaics.tif'
    target_crs = 'EPSG:4326'
    log_file = r'E:\GuiZhouProvinces\logs\mosaic.log'
    batch_size = 484  # 每批处理11个文件
    main(input_dir, output_path, target_crs, log_file, batch_size)