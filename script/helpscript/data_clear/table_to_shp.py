import pandas as pd
import json
import geopandas as gpd
from shapely.geometry import Point
import os
import fiona
import logging
from pathlib import Path
from dotenv import load_dotenv

# 文件开头添加 logger 定义
logger = logging.getLogger(__name__)

def load_excel_dtype(dtype_dict_path):
    with open(dtype_dict_path, 'r', encoding='utf-8') as file:
        dtype_dict = json.load(file)
    return dtype_dict

def table_to_shp(input_file, output_file, lon_col, lat_col, dtype_dict_path, input_crs=4326, output_crs=None):
    logger.info(f"开始表格转shapefile过程: {input_file} -> {output_file}")
    # 设置fiona支持utf-8编码
    fiona.supported_drivers['ESRI Shapefile'] = 'rw'

    # 读取数据类型字典
    dtype_dict = load_excel_dtype(dtype_dict_path)
    
    # 先不指定数据类型读取文件
    _, file_extension = os.path.splitext(input_file)
    if file_extension.lower() == '.xlsx':
        df = pd.read_excel(input_file)
    elif file_extension.lower() == '.csv':
        df = pd.read_csv(input_file)
    else:
        raise ValueError("不支持的文件格式。请使用.xlsx或.csv文件。")
    
    # 对需要转换为整数的列进行处理
    for col, dtype in dtype_dict.items():
        if dtype == 'int64' and col in df.columns:
            # 先将NaN填充为0，然后转换为整数
            df[col] = df[col].fillna(0).astype('int64')
        else:
            df[col] = df[col].astype(dtype)
    
    # 创建几何列
    geometry = [Point(xy) for xy in zip(df[lon_col], df[lat_col])]
    
    # 创建GeoDataFrame
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs=f"EPSG:{input_crs}")
    
    # 如果需要,转换坐标系
    if input_crs != output_crs:
        gdf = gdf.to_crs(epsg=output_crs)
    
    # 保存为shapefile,使用utf-8编码
    gdf.to_file(output_file, driver="ESRI Shapefile", encoding='utf-8')

    logger.info(f"Shapefile已保存至: {output_file}")


# 
if __name__ == "__main__":
    # 获取环境变量
    load_dotenv()
    base_data_path = os.getenv('SAVE_PATH')
    shp_path = os.getenv('SHP_PATH')
    log_file = os.path.join(base_data_path, 'logs', 'table_to_shp.log')
    dtype_dict_path = os.getenv('DTYPE_ALL_PATH')
    # 设置日志
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # 添加控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # 如果指定了日志文件，添加文件处理器
    if log_file:
        log_dir = Path(log_file).parent
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    for file in os.listdir(base_data_path):
        if file.endswith('.csv'):
            input_file = os.path.join(base_data_path, file)
            output_file = os.path.join(shp_path, file.replace('.csv', '.shp')) 
            lon_col = 'dwjd'
            lat_col = 'dwwd'
            input_crs = 4326
            output_crs = 4490
            table_to_shp(input_file, output_file, lon_col, lat_col, dtype_dict_path, input_crs, output_crs)