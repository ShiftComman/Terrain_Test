import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import os
import fiona
import logging
from pathlib import Path


def table_to_shp(input_file, output_file, lon_col, lat_col, input_crs=4326, output_crs=4544):
    # 设置fiona支持utf-8编码
    fiona.supported_drivers['ESRI Shapefile'] = 'rw'

    # 读取输入文件
    _, file_extension = os.path.splitext(input_file)
    if file_extension.lower() == '.xlsx':
        df = pd.read_excel(input_file)
    elif file_extension.lower() == '.csv':
        df = pd.read_csv(input_file, encoding='utf-8')
    else:
        raise ValueError("不支持的文件格式。请使用.xlsx或.csv文件。")

    # 创建几何列
    geometry = [Point(xy) for xy in zip(df[lon_col], df[lat_col])]
    
    # 创建GeoDataFrame
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs=f"EPSG:{input_crs}")
    
    # 如果需要,转换坐标系
    if input_crs != output_crs:
        gdf = gdf.to_crs(epsg=output_crs)
    
    # 保存为shapefile,使用utf-8编码
    gdf.to_file(output_file, driver="ESRI Shapefile", encoding='utf-8')

    print(f"Shapefile已保存至: {output_file}")

def main(input_file, output_file, lon_col, lat_col, log_file, input_crs=4326, output_crs=4544):
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
    
    logger.info("开始表格转shapefile过程")
    try:
        table_to_shp(input_file, output_file, lon_col, lat_col, input_crs, output_crs)
        logger.info("表格转shapefile过程完成")
    except Exception as e:
        logger.error(f"表格转shapefile过程发生错误: {e}")

# 
if __name__ == "__main__":
    input_file = r"F:\soil_mapping\gl\figures\data_clean\cleaned_data.csv"
    output_file = r"F:\soil_mapping\gl\figures\data_clean\gl_result_point.shp"
    lon_col = 'dwjd'
    lat_col = 'dwwd'
    input_crs = 4326
    output_crs = 4544
    log_file = r"F:\soil_mapping\gl\soil-mapping\logs\table_to_shp.log"
    main(input_file, output_file, lon_col, lat_col, log_file, input_crs, output_crs)