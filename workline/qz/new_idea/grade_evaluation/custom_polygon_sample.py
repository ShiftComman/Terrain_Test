import os
import geopandas as gpd
import rasterio
import pandas as pd
import numpy as np
from rasterstats import zonal_stats
import logging
from datetime import datetime
import sys
from shapely.geometry import shape
from rasterio.mask import mask
import multiprocessing
import psutil
from tqdm import tqdm
import time
from pathlib import Path
from typing import Dict, List, Union

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def setup_logger(name, log_file, level=logging.INFO):
    """设置日志记录器"""
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    log_dir = Path(log_file).parent
    log_dir.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(log_file)
    handler.setFormatter(formatter)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    logger.addHandler(console_handler)
    return logger

logger = setup_logger('custom_zonal_raster', Path('logs/custom_zonal_raster.log'))

def process_single_raster(args):
    vector_data, raster_path, stats_methods = args  # 移除 pbar 参数
    try:
        raster_name = Path(raster_path).name
        
        with rasterio.open(raster_path) as src:
            logger.info(f"栅格文件打开成功: {raster_path}")
            logger.info(f"栅格形状: {src.shape}, 数据类型: {src.dtypes[0]}")
            
            raster_data = src.read(1, masked=True)
            affine = src.transform
            
            stats_results = zonal_stats(vector_data, 
                                      raster_data, 
                                      affine=affine,
                                      stats=stats_methods,
                                      all_touched=True,
                                      nodata=src.nodata if src.nodata is not None else -999)
            
        result_df = pd.DataFrame(stats_results)
        raster_name = Path(raster_path).stem.replace(' ', '_').replace('-', '_')
        new_columns = {col: f"{raster_name}_{col}" for col in result_df.columns}
        result_df.rename(columns=new_columns, inplace=True)
        
        return result_df, None  # 返回结果和错误状态
    except Exception as e:
        logger.error(f"处理 {raster_path} 时出错: {str(e)}")
        return None, str(e)  # 返回错误状态

def process_rasters_custom(
    polygon_path: Union[str, Path],
    raster_names: List[str],  # 完整路径列表
    output_path: Union[str, Path],
    custom_stats: Dict[str, List[str]] = None,
    default_stats: List[str] = ['mean'],
    output_format: str = 'csv',
    simplify: bool = True,
    simplify_tolerance: float = 0.001,
    use_multiprocessing: bool = True,
    num_workers: int = None
):
    """
    自定义栅格处理函数
    
    参数:
        polygon_path: 多边形矢量文件路径
        raster_names: 栅格文件完整路径列表
        output_path: 输出文件路径
        custom_stats: 特定栅格的自定义统计方法，格式为 {
            "栅格文件名1": ["mean", "max", "min"],
            "栅格文件名2": ["count", "majority"]
        }
        default_stats: 默认统计方法
        output_format: 输出格式 ('csv' 或 'shp')
        simplify: 是否简化几何形状
        simplify_tolerance: 简化容差
        use_multiprocessing: 是否使用多进程
        num_workers: 进程数量
    """
    start_time = time.time()
    custom_stats = custom_stats or {}
    
    # 创建总进度条
    total_pbar = tqdm(total=6, desc="总体进度", position=0)
    
    logger.info(f"配置包含 {len(raster_names)} 个栅格文件")
    total_pbar.update(1)
    
    # 修改验证栅格文件存在的部分
    missing_files = []
    for raster_path in raster_names:
        if not Path(raster_path).exists():
            missing_files.append(str(raster_path))
    
    if missing_files:
        raise FileNotFoundError(f"以下栅格文件不存在:\n" + "\n".join(missing_files))
    
    total_pbar.update(1)
    
    logger.info(f"读取并预处理矢量数据: {polygon_path}")
    gdf = gpd.read_file(polygon_path, fid_as_index=True)
    if simplify:
        original_area = gdf.area.sum()
        gdf['geometry'] = gdf.geometry.simplify(tolerance=simplify_tolerance)
        simplified_area = gdf.area.sum()
        area_change = (simplified_area - original_area) / original_area * 100
        logger.info(f"几何简化完成。面积变化: {area_change:.2f}%")
    
    total_pbar.update(1)
    
    # 创建栅格处理进度条
    raster_pbar = tqdm(total=len(raster_names), desc="栅格处理进度", position=1, leave=True)
    
    process_args = []
    for raster_path in raster_names:
        raster_name = Path(raster_path).stem
        stats_methods = custom_stats.get(raster_name, default_stats)
        process_args.append((gdf, raster_path, stats_methods))
    
    results = []
    errors = []
    
    if use_multiprocessing:
        if num_workers is None:
            available_memory = psutil.virtual_memory().available
            num_workers = min(multiprocessing.cpu_count(), max(1, int(available_memory / (4 * 1024 * 1024 * 1024))))
        logger.info(f"使用 {num_workers} 个工作进程进行并行处理")
        
        with multiprocessing.Pool(processes=num_workers) as pool:
            for result, error in pool.imap(process_single_raster, process_args):
                if error is None:
                    results.append(result)
                else:
                    errors.append(error)
                raster_pbar.update(1)
                raster_pbar.set_description(f"处理进度 (成功: {len(results)}, 失败: {len(errors)})")
    else:
        logger.info("使用单进程处理")
        for args in process_args:
            result, error = process_single_raster(args)
            if error is None:
                results.append(result)
            else:
                errors.append(error)
            raster_pbar.update(1)
            raster_pbar.set_description(f"处理进度 (成功: {len(results)}, 失败: {len(errors)})")
    
    if errors:
        logger.warning(f"处理过程中出现 {len(errors)} 个错误:")
        for error in errors:
            logger.warning(error)
    
    total_pbar.update(1)
    
    logger.info("合并所有结果")
    df_merged = pd.concat([gdf.reset_index().rename(columns={'index': 'FID'})] + results, axis=1)
    
    logger.info("添加中心点坐标和边界框信息")
    df_merged['Centroid_X'] = gdf.geometry.centroid.x
    df_merged['Centroid_Y'] = gdf.geometry.centroid.y
    bounds = gdf.bounds
    df_merged['XMin'], df_merged['YMin'], df_merged['XMax'], df_merged['YMax'] = bounds.minx, bounds.miny, bounds.maxx, bounds.maxy
    # 计算面积
    df_merged['project_Area'] = gdf.area
    total_pbar.update(1)
    
    logger.info(f"保存数据到: {output_path}")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if output_format.lower() == 'csv':
        df_merged.to_csv(output_path, index=False, encoding='utf-8')
        logger.info("数据已保存为CSV格式")
    elif output_format.lower() == 'shp':
        gdf_result = gpd.GeoDataFrame(df_merged, geometry=gdf.geometry)
        gdf_result.to_file(output_path, driver='ESRI Shapefile', encoding='utf-8')
        logger.info("数据已保存为Shapefile格式")
    else:
        logger.error(f"不支持的输出格式: {output_format}")
        raise ValueError(f"不支持的输出格式: {output_format}")

    total_pbar.update(1)
    
    end_time = time.time()
    total_time = end_time - start_time
    logger.info(f"处理完成，总耗时: {total_time:.2f} 秒")
    
    # 关闭进度条
    total_pbar.close()
    raster_pbar.close()

def generate_raster_paths(folder_config: Dict[str, Dict]):
    """
    根据文件夹配置生成栅格路径
    
    参数:
        folder_config: 包含文件夹路径和对应栅格名称的字典
        格式为: {
            'category_name': {
                'folder': Path对象,
                'raster_names': List[str]
            }
        }
    
    返回:
        raster_paths: 所有栅格文件的完整路径列表
        custom_stats: 自定义统计方法配置
    """
    raster_paths = []
    custom_stats = {}
    
    # 统计方法配置
    stats_methods = {
        'categorical': ['majority'],  # 分类数据使用众数
        'continuous': ['mean']        # 连续数据使用平均值
    }
    
    # 分类数据列表
    categorical_rasters = ['irrigation_level', 'drainage_level', 'slopepostion', 'TRZD_prediction']
    
    for category, config in folder_config.items():
        folder = config['folder']
        raster_names = config['raster_names']
        
        for raster_name in raster_names:
            raster_path = folder / f"{raster_name}.tif"
            if raster_path.exists():
                raster_paths.append(str(raster_path))
                
                # 根据栅格类型设置统计方法
                if any(cat in raster_name for cat in categorical_rasters):
                    custom_stats[raster_name] = stats_methods['categorical']
                else:
                    custom_stats[raster_name] = stats_methods['continuous']
            else:
                logger.warning(f"栅格文件不存在: {raster_path}")
    
    return raster_paths, custom_stats

if __name__ == "__main__":
    # 基础路径配置
    polygon_path = Path(r"G:\soil_property_result\qzs\grade_evaluation\shp\crop_dissolve.shp")
    output_path = Path(r"G:\soil_property_result\qzs\grade_evaluation\table\grade_evaluation_sample.csv")
    
    # 配置不同类型栅格的文件夹和文件名
    folder_config = {
        'nutrient': {
            'folder': Path(r"G:\soil_property_result\qzs\soil_property_predict"),
            'raster_names': [
                'PH_prediction', 'OM_prediction', 'AK_prediction', 'AP_prediction',
                'TRRZ_prediction', 'GZCHD_prediction', 'YXTCHD_prediction',
                'CD_prediction', 'HG_prediction', 'AS2_prediction', 'PB_prediction', 
                'CR_prediction'
            ]
        },
        'terrain': {
            'folder': Path(r"G:\tif_features\county_feature\qz"),
            'raster_names': ['slopepostion', 'dem']
        },
        'irrigation_drainage': {
            'folder': Path(r"G:\soil_property_result\qzs\irrigation_drainage_generate"),
            'raster_names': ['irrigation_level_2025022411', 'drainage_level_2025022411']
        },
        'texture': {
            'folder': Path(r"G:\soil_property_result\qzs\soil_property_class_predict"),
            'raster_names': ['TRZD_prediction']
        }
    }
    
    # 生成栅格路径和统计方法配置
    raster_paths, custom_stats = generate_raster_paths(folder_config)
    
    # 验证是否有找到栅格文件
    if not raster_paths:
        logger.error("未找到任何栅格文件")
        sys.exit(1)
    
    logger.info(f"找到 {len(raster_paths)} 个栅格文件")
    
    # 运行处理
    process_rasters_custom(
        polygon_path=polygon_path,
        raster_names=raster_paths,  # 直接传入完整路径列表
        output_path=output_path,
        custom_stats=custom_stats,
        default_stats=['mean'],
        output_format='csv',
        simplify=True,
        simplify_tolerance=0.001,
        use_multiprocessing=True,
        num_workers=6
    ) 