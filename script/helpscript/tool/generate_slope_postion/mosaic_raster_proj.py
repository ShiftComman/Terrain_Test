import rasterio
import numpy as np
from rasterio.merge import merge
from rasterio.warp import calculate_default_transform, reproject, Resampling
import os
from rasterio.enums import ColorInterp
import logging
from tqdm import tqdm
from pathlib import Path
from osgeo import gdal  # 添加 GDAL 导入

def convert_bil_to_tiff(bil_path, temp_dir):
    """将 BIL 文件转换为临时 GeoTIFF 文件"""
    temp_tiff = os.path.join(temp_dir, os.path.basename(bil_path).replace('.bil', '_temp.tif'))
    ds = gdal.Open(bil_path)
    if ds is None:
        raise ValueError(f"GDAL 无法打开文件: {bil_path}")
    
    gdal.Translate(temp_tiff, ds, format='GTiff')
    ds = None  # 关闭数据集
    return temp_tiff

def merge_rasters(raster_path, output_path, logger, target_crs=None):
    src_files = []
    temp_files = []  # 存储临时文件路径
    temp_dir = os.path.join(os.path.dirname(output_path), 'temp')
    os.makedirs(temp_dir, exist_ok=True)
    
    try:
        # 递归获取所有.tif和.bil格式的栅格文件
        raster_paths = []
        for root, dirs, files in os.walk(raster_path):
            for f in files:
                if f.lower().endswith(('.tif', '.bil')):
                    raster_paths.append(os.path.join(root, f))
        
        if not raster_paths:
            logger.error("未找到.tif或.bil格式的栅格文件")
            raise ValueError("目录中没有找到可用的栅格文件")
        
        # 处理每个栅格文件，并将投影转换为指定的CRS
        for raster in raster_paths:
            try:
                # 检查文件扩展名，如果是BIL文件，则进行转换
                if raster.lower().endswith('.bil'):
                    logger.info(f"转换 BIL 文件: {raster}")
                    temp_tiff = convert_bil_to_tiff(raster, temp_dir)
                    temp_files.append(temp_tiff)
                    src = rasterio.open(temp_tiff)
                else:
                    src = rasterio.open(raster)

                # 检查并转换投影
                if src.crs.to_string() != target_crs:
                    logger.info(f"将 {raster} 从 {src.crs.to_string()} 转换为 {target_crs}")
                    reprojected_path = os.path.join(temp_dir, f"reprojected_{os.path.basename(raster)}")
                    reproject_raster(src, reprojected_path, target_crs)
                    temp_files.append(reprojected_path)
                    src = rasterio.open(reprojected_path)

                src_files.append(src)
                logger.info(f"成功处理文件: {raster}")
            except Exception as e:
                logger.error(f"处理文件 {raster} 时出错: {str(e)}")
                raise

        # 获取第一个文件的元数据
        meta = src_files[0].meta.copy()
        
        # 执行合并（使用指定输入投影）
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
        
        # 写入最终的栅格
        logger.info(f"写入输出文件到 {output_path}")
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
        # 清理临时文件
        for temp_file in temp_files:
            try:
                os.remove(temp_file)
                logger.info(f"删除临时文件: {temp_file}")
            except Exception as e:
                logger.warning(f"删除临时文件失败: {temp_file}, 错误: {str(e)}")
        try:
            os.rmdir(temp_dir)
        except:
            pass

def reproject_raster(src, dst_path, dst_crs):
    """将栅格重投影到指定的CRS"""
    transform, width, height = calculate_default_transform(
        src.crs, dst_crs, src.width, src.height, *src.bounds)
    
    kwargs = src.meta.copy()
    kwargs.update({
        'crs': dst_crs,
        'transform': transform,
        'width': width,
        'height': height
    })

    with rasterio.open(dst_path, 'w', **kwargs) as dst:
        for i in range(1, src.count + 1):
            reproject(
                source=rasterio.band(src, i),
                destination=rasterio.band(dst, i),
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=transform,
                dst_crs=dst_crs,
                resampling=Resampling.nearest)

def main(input_dir:str,output_path:str,target_crs:str,log_file:str):
    # 配置日志
    if log_file:
        log_dir = Path(log_file).parent
        log_dir.mkdir(parents=True, exist_ok=True)
        logging.basicConfig(filename=log_file, level=logging.INFO, 
                            format='%(asctime)s - %(levelname)s - %(message)s',encoding='utf-8')
    else:
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',encoding='utf-8')
    
    logger = logging.getLogger(__name__)
    logger.info("开始处理栅格数据")
    try:
        merge_rasters(input_dir, output_path, logger, target_crs)
    except Exception as e:
        logger.error(f"处理栅格数据时发生错误: {str(e)}")
        raise
    finally:
        logger.info("栅格数据处理完成")
if __name__ == "__main__":
    input_dir = r'F:\ArcgisData\m5\fq'
    output_path = r'F:\ArcgisData\m5\fq\fq_mosaic_dem.tif'
    target_crs = 'EPSG:4545'
    log_file = r'F:\ArcgisData\m5\fq\fq_mosaic_dem.log'
    main(input_dir,output_path,target_crs,log_file)
