from shapely.geometry import Polygon, MultiPolygon, LineString
from shapely.ops import split
import geopandas as gpd
import numpy as np
from tqdm import tqdm
import logging
import time
import os
import glob
import pandas as pd

logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_optimal_split_line(poly):
    """优化的分割线生成函数"""
    minx, miny, maxx, maxy = poly.bounds
    width = maxx - minx
    height = maxy - miny
    centroid = poly.centroid
    
    split_lines = []
    angles = [0, 45, 90, 135]
    
    for angle in angles:
        rad = np.radians(angle)
        length = max(width, height) * 1.5
        dx = length * np.cos(rad)
        dy = length * np.sin(rad)
        
        line = LineString([
            (centroid.x - dx, centroid.y - dy),
            (centroid.x + dx, centroid.y + dy)
        ])
        split_lines.append(line)
    
    best_line = None
    best_difference = float('inf')
    
    for line in split_lines:
        try:
            result = split(poly, line)
            if len(list(result.geoms)) == 2:
                areas = [p.area for p in result.geoms]
                difference = abs(areas[0] - areas[1])
                if difference < best_difference:
                    best_difference = difference
                    best_line = line
        except Exception:
            continue
    
    return best_line if best_line is not None else split_lines[0]

def split_polygon_by_area(polygon, max_area, min_area=None):
    """改进的多边形分割函数"""
    if min_area is None:
        min_area = max_area * 0.1
    
    def split_if_needed(poly):
        if poly.area <= max_area:
            return [poly]
        
        split_line = get_optimal_split_line(poly)
        
        try:
            result = split(poly, split_line)
            split_parts = list(result.geoms)
            
            if len(split_parts) < 2:
                return [poly]
            
            split_polygons = []
            for p in split_parts:
                if isinstance(p, (Polygon, MultiPolygon)):
                    if p.area > min_area:
                        split_polygons.extend(split_if_needed(p))
                    else:
                        split_polygons.append(p)
            
            return split_polygons
        except Exception as e:
            logger.warning(f"分割失败: {str(e)}")
            return [poly]
    
    if isinstance(polygon, MultiPolygon):
        result = []
        for poly in polygon.geoms:
            result.extend(split_if_needed(poly))
        return result
    
    return split_if_needed(polygon)

def process_shapefile(shp_path, max_area, output_path=None, batch_size=100):
    """改进的shapefile处理函数"""
    logger.info(f"开始处理文件: {shp_path}")
    logger.info(f"设定的最大面积阈值: {max_area}")
    
    # 创建临时文件夹
    temp_dir = os.path.join(os.path.dirname(shp_path), "temp_split")
    os.makedirs(temp_dir, exist_ok=True)
    
    logger.info("正在读取shapefile...")
    gdf = gpd.read_file(shp_path, encoding='utf-8')
    logger.info(f"成功读取 {len(gdf)} 个多边形")
    
    # 存储分割后的结果
    split_geometries = []
    split_attributes = []
    temp_files = []
    
    logger.info("开始分割多边形...")
    for idx, row in tqdm(gdf.iterrows(), total=len(gdf), desc="处理多边形"):
        logger.info(f"正在处理第 {idx+1}/{len(gdf)} 个多边形")
        logger.info(f"原始面积: {row.geometry.area:.2f}")
        
        start_time = time.time()
        split_polys = split_polygon_by_area(row.geometry, max_area)
        logger.info(f"分割耗时: {time.time() - start_time:.2f}秒")
        logger.info(f"分割后得到 {len(split_polys)} 个子多边形")
        
        # 为每个分割后的多边形复制原始属性
        for poly in split_polys:
            split_geometries.append(poly)
            split_attributes.append(row.drop('geometry'))
        
        # 批量保存
        if len(split_geometries) >= batch_size:
            temp_gdf = gpd.GeoDataFrame(
                split_attributes,
                geometry=split_geometries,
                crs=gdf.crs
            )
            temp_file = os.path.join(temp_dir, f'temp_split_{len(temp_files)}.shp')
            temp_gdf.to_file(temp_file, encoding='utf-8')
            temp_files.append(temp_file)
            logger.info(f"保存临时文件: {temp_file}")
            
            # 清空临时存储
            split_geometries = []
            split_attributes = []
    
    # 保存最后的批次
    if split_geometries:
        temp_gdf = gpd.GeoDataFrame(
            split_attributes,
            geometry=split_geometries,
            crs=gdf.crs
        )
        temp_file = os.path.join(temp_dir, f'temp_split_{len(temp_files)}.shp')
        temp_gdf.to_file(temp_file, encoding='utf-8')
        temp_files.append(temp_file)
    
    # 合并所有临时文件
    logger.info("合并临时文件...")
    gdfs = []
    for temp_file in temp_files:
        try:
            gdf = gpd.read_file(temp_file)
            gdfs.append(gdf)
        except Exception as e:
            logger.warning(f"读取临时文件失败: {str(e)}")
            continue
    
    if gdfs:
        combined_gdf = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True), crs=gdf.crs)
    else:
        logger.warning("没有有效的临时文件可以合并")
        return None
    
    # 保存最终结果
    if output_path is None:
        output_path = shp_path.replace('.shp', '_split.shp')
    
    logger.info(f"保存最终结果到: {output_path}")
    combined_gdf.to_file(output_path, encoding='utf-8')
    
    # 清理临时文件
    logger.info("清理临时文件...")
    for temp_file in temp_files:
        try:
            os.remove(temp_file)
            # 删除相关的其他文件(.dbf, .prj, .shx)
            for ext in ['.dbf', '.prj', '.shx']:
                temp_file_ext = temp_file.replace('.shp', ext)
                if os.path.exists(temp_file_ext):
                    os.remove(temp_file_ext)
        except Exception as e:
            logger.warning(f"清理临时文件失败: {str(e)}")
    
    try:
        os.rmdir(temp_dir)
    except Exception as e:
        logger.warning(f"删除临时文件夹失败: {str(e)}")
    
    logger.info("处理完成")
    return combined_gdf

def example_usage():
    """使用示例"""
    input_shp = r"C:\Users\Runker\Desktop\ele_sb\clip_test.shp"
    max_area = 20000
    output_shp = r"C:\Users\Runker\Desktop\ele_sb\clip_test_splits.shp"
    
    logger.info("开始示例运行")
    result = process_shapefile(
        input_shp, 
        max_area, 
        output_shp, 
        batch_size=100  # 增加批处理大小
    )
    
    logger.info(f"原始多边形数量: {len(gpd.read_file(input_shp))}")
    logger.info(f"分割后多边形数量: {len(result)}")

if __name__ == "__main__":
    example_usage()