import os
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, Polygon
import rasterio
from rasterio.mask import mask
import numpy as np
import logging
import unidecode
import fiona

fiona.drvsupport.supported_drivers['ESRI Shapefile'] = 'rw'

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 创建渔网
def create_net(output_file, extent_file, cell_size):
    gdf = gpd.read_file(extent_file)
    bounds = gdf.total_bounds
    x_min, y_min, x_max, y_max = bounds
    
    x_coords = np.arange(x_min, x_max, cell_size)
    y_coords = np.arange(y_min, y_max, cell_size)
    
    polygons = []
    for x in x_coords:
        for y in y_coords:
            polygons.append(Polygon([(x, y), (x+cell_size, y), (x+cell_size, y+cell_size), (x, y+cell_size)]))
    
    fishnet = gpd.GeoDataFrame({'geometry': polygons}, crs=gdf.crs)
    fishnet.to_file(output_file)

# 面转点
def polygon_point(in_feature, out_feature):
    gdf = gpd.read_file(in_feature)
    gdf['geometry'] = gdf['geometry'].centroid
    gdf.to_file(out_feature)

# 筛选点
def select_point(input_polygon, input_point, output_point):
    polygon_gdf = gpd.read_file(input_polygon)
    point_gdf = gpd.read_file(input_point)
    
    selected_points = gpd.sjoin(point_gdf, polygon_gdf, how="inner", op="within")
    selected_points.to_file(output_point)


def create_point(input_layer):
    """生成虚拟点"""
    logging.info(f"开始处理输入图层: {input_layer}")
    
    # 定义文件名称
    fish_net_name = "grid.shp"
    net_to_point_name = "inside_point.shp"
    select_point_name = "select_point.shp"
    result_point_name = "join_point.shp"
    
    # 渔网构建
    logging.info(f"创建渔网: {fish_net_name}")
    create_net(fish_net_name, input_layer, 300)
    
    # 要素转点
    logging.info(f"将渔网转换为点: {net_to_point_name}")
    gdf = gpd.read_file(fish_net_name, encoding='utf-8')
    gdf['geometry'] = gdf['geometry'].centroid
    gdf.to_file(net_to_point_name, encoding='utf-8')
    
    # 按位置选择
    logging.info(f"选择点: {select_point_name}")
    input_gdf = gpd.read_file(input_layer, encoding='utf-8')
    points_gdf = gpd.read_file(net_to_point_name, encoding='utf-8')
    selected_points = gpd.sjoin(points_gdf, input_gdf, how="inner", predicate="within")
    
    # 处理字段名
    selected_points.columns = [truncate_field_name(col) for col in selected_points.columns]
    
    # 添加字段
    logging.info("添加字段")
    field_list = ['TL', 'YL', 'TS', 'TZ']
    for one_field in field_list:
        selected_points[one_field] = ''
    
    logging.info(f"保存选择的点: {select_point_name}")
    selected_points.to_file(select_point_name, encoding='utf-8')
    
    # 空间连接赋予地类属性
    logging.info("进行空间连接")
    fields_mapping = {
        "TL": "土类",
        "YL": "亚类",
        "TS": "土属",
        "TZ": "土种"
    }
    
    input_gdf = gpd.read_file(input_layer, encoding='utf-8')
    selected_points = gpd.read_file(select_point_name, encoding='utf-8')
    
    joined = gpd.sjoin(selected_points, input_gdf, how="left", predicate="within")
    
    for target_field, source_field in fields_mapping.items():
        joined[target_field] = joined[source_field]
    
    joined = joined[list(fields_mapping.keys()) + ['geometry']]
    
    logging.info(f"保存结果: {result_point_name}")
    joined.to_file(result_point_name, encoding='utf-8')
    
    # 删除多余数据
    logging.info("清理临时文件")
    temp_files = [fish_net_name, net_to_point_name, select_point_name]
    for base_name in temp_files:
        delete_shp_files(base_name)
        logging.info(f"已删除: {base_name}")
    
    logging.info("处理完成")
    return joined  # 返回生成的点数据

def truncate_field_name(name, max_length=10):
    """截断字段名并确保唯一性"""
    truncated = unidecode.unidecode(name)[:max_length].strip()
    return truncated

def create_polygon_point(input_layer):
    """从面要素生成点并提取属性"""
    logging.info(f"开始处理输入图层: {input_layer}")
    
    result_point_name = "polygon_points.shp"
    
    # 读取输入图层并转换为点
    logging.info("将面要素转换为点")
    gdf = gpd.read_file(input_layer, encoding='utf-8')
    gdf['geometry'] = gdf['geometry'].centroid
    
    # 处理字段名
    gdf.columns = [truncate_field_name(col) for col in gdf.columns]
    
    # 添加字段
    logging.info("添加字段")
    field_list = ['TL', 'YL', 'TS', 'TZ']
    for one_field in field_list:
        gdf[one_field] = ''
    
    # 空间连接赋予地类属性
    logging.info("进行空间连接")
    fields_mapping = {
        "TL": "土类",
        "YL": "亚类",
        "TS": "土属",
        "TZ": "土种"
    }
    
    input_gdf = gpd.read_file(input_layer, encoding='utf-8')
    
    joined = gpd.sjoin(gdf, input_gdf, how="left", predicate="within")
    
    for target_field, source_field in fields_mapping.items():
        if source_field in joined.columns:
            joined[target_field] = joined[source_field]
        else:
            logging.warning(f"字段 '{source_field}' 不存在于输入图层中")
    
    result_points = joined[list(fields_mapping.keys()) + ['geometry']]
    
    logging.info(f"保存结果: {result_point_name}")
    result_points.to_file(result_point_name, encoding='utf-8')
    
    logging.info("处理完成")
    return result_points

def delete_shp_files(file_name):
    """删除与给定文件名相关的所有 shp 文件"""
    base_name = file_name.rsplit('.', 1)[0]
    for ext in ['.shp', '.shx', '.dbf', '.prj']:
        file_path = f"{base_name}{ext}"
        if os.path.exists(file_path):
            os.remove(file_path)
            logging.info(f"已删除: {file_path}")

def main(grid_input, polygon_input, output_file, grid_marker, polygon_marker):
    # 设置工作空间
    workspace = r'F:\cache_data\shp_file\sb'
    os.chdir(workspace)
    logging.info(f"当前工作目录: {os.getcwd()}")
    
    # 生成网格点
    grid_points = create_point(grid_input)
    if grid_points is not None:
        grid_points['marker'] = grid_marker
    else:
        logging.error("网格点生成失败")
        return
    
    # 生成面内部点
    polygon_points = create_polygon_point(polygon_input)
    if polygon_points is not None:
        polygon_points['marker'] = polygon_marker
    else:
        logging.error("面内部点生成失败")
        return
    
    # 合并两组点
    combined_points = pd.concat([grid_points, polygon_points], ignore_index=True)
    
    # 为结果文件添加四个字段
    for one_field in ['NEW_TL', 'NEW_YL', 'NEW_TS', 'NEW_TZ']:
        combined_points[one_field] = ''
    
    # 保存结果
    combined_points.to_file(output_file, encoding='utf-8')
    logging.info(f"合并结果已保存至: {output_file}")
    
    # 删除中间文件
    intermediate_files = ['join_point.shp', 'polygon_points.shp']
    for file in intermediate_files:
        delete_shp_files(file)
    
    logging.info("已删除所有中间文件")

if __name__ == "__main__":
    grid_input = "sb_ep_polygon.shp"
    polygon_input = "sb_ep_polygon.shp"
    output_file = "sb_filter_points.shp"
    grid_marker = "grid"
    polygon_marker = "polygon"
    main(grid_input, polygon_input, output_file, grid_marker, polygon_marker)
