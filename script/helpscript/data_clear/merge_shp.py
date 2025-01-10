import geopandas as gpd
import pandas as pd
import concurrent.futures
import numpy as np
import fiona
import os
import logging
from pathlib import Path
from dotenv import load_dotenv
from tqdm import tqdm

# 设置logger
logger = logging.getLogger(__name__)

def read_shapefile_chunked(file_path, chunksize=10000):
    """
    分块读取单个shapefile文件
    """
    try:
        # 直接读取文件，不进行分块（因为分块导致了类型错误）
        gdf = gpd.read_file(file_path)
        
        # 确保所有列都是正确的数据类型
        for col in gdf.columns:
            if col != 'geometry':
                # 尝试转换为数值类型，如果失败则保持为字符串
                try:
                    gdf[col] = pd.to_numeric(gdf[col], errors='coerce')
                except:
                    gdf[col] = gdf[col].astype(str)
        
        return gdf
    except Exception as e:
        logger.error(f"读取文件 {file_path} 时出错: {str(e)}")
        return None

def process_shapefile(file_path):
    """
    处理单个shapefile文件
    """
    try:
        gdf = read_shapefile_chunked(file_path)
        if gdf is not None:
            # 确保返回的是GeoDataFrame
            if not isinstance(gdf, gpd.GeoDataFrame):
                logger.error(f"文件 {file_path} 返回了非GeoDataFrame对象")
                return None
            return gdf
    except Exception as e:
        logger.error(f"处理文件 {file_path} 时出错: {str(e)}")
    return None

def merge_shapefiles(input_dir, output_file, max_workers=4):
    """
    合并指定目录下的所有shp文件
    """
    logger.info(f"开始合并shapefile文件，源目录: {input_dir}")
    
    # 获取所有shp文件
    shp_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith('.shp')]
    total_files = len(shp_files)
    
    if not shp_files:
        logger.warning(f"在 {input_dir} 目录下没有找到shp文件")
        return
    
    logger.info(f"共发现 {total_files} 个shp文件待处理")
    
    # 并行处理文件
    gdfs = []
    with tqdm(total=total_files, desc="处理进度", unit="文件") as pbar:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_file = {executor.submit(process_shapefile, f): f for f in shp_files}
            
            for future in concurrent.futures.as_completed(future_to_file):
                file_path = future_to_file[future]
                try:
                    gdf = future.result()
                    if gdf is not None and isinstance(gdf, gpd.GeoDataFrame):
                        gdfs.append(gdf)
                    pbar.set_postfix({"当前文件": os.path.basename(file_path)}, refresh=True)
                    pbar.update(1)
                except Exception as e:
                    logger.error(f"处理文件 {file_path} 失败: {str(e)}")

    if not gdfs:
        logger.error("没有成功读取任何shapefile文件")
        return
    
    logger.info(f"成功读取 {len(gdfs)}/{total_files} 个文件，开始合并...")
    
    try:
        # 确保所有GeoDataFrame具有相同的列
        common_columns = set.intersection(*[set(gdf.columns) for gdf in gdfs])
        gdfs = [gdf[list(common_columns)] for gdf in gdfs]
        
        # 合并所有GeoDataFrame
        logger.info("开始合并文件...")
        final_gdf = pd.concat(gdfs, ignore_index=True)
        
        # 确保结果是GeoDataFrame
        if not isinstance(final_gdf, gpd.GeoDataFrame):
            final_gdf = gpd.GeoDataFrame(final_gdf, geometry='geometry')
        
        # 保存合并后的文件
        logger.info("正在保存合并后的文件...")
        final_gdf.to_file(output_file, encoding='utf-8')
        logger.info(f"合并完成，文件已保存至: {output_file}")
        
    except Exception as e:
        logger.error(f"合并或保存文件时出错: {str(e)}")
        raise

if __name__ == "__main__":
    # 获取环境变量
    load_dotenv()
    shp_path = os.getenv('SHP_PATH')
    log_data_path = os.getenv('LOG_DATA_PATH')
    log_file = os.path.join(log_data_path, 'merge_shp.log')
    
    # 设置日志
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # 添加控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # 添加文件处理器
    if log_file:
        log_dir = Path(log_file).parent
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    # 设置输出文件路径
    output_file = os.path.join(shp_path, 'china_merged_multiple.shp')
    
    # 执行合并操作
    merge_shapefiles(shp_path, output_file)