from shapely.geometry import Polygon, MultiPolygon, LineString, Point
from shapely.ops import split, unary_union
import geopandas as gpd
import numpy as np
from scipy.interpolate import CubicSpline
from tqdm import tqdm
import logging
import time
import os
import pandas as pd
import random
import argparse
import shutil
from enum import Enum
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from itertools import islice

logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SplitMethod(Enum):
    CURVE = "curve"      # 自然曲线切割
    MIDLINE = "midline"  # 中线切割
    LINE = "line"        # 直线切割

# === 曲线切割方法 ===
def create_natural_curve(start_point, end_point, control_points_count=3, noise_factor=0.3):
    dx = end_point[0] - start_point[0]
    dy = end_point[1] - start_point[1]
    distance = np.sqrt(dx**2 + dy**2)
    
    control_points = []
    for i in range(control_points_count):
        t = (i + 1) / (control_points_count + 1)
        base_x = start_point[0] + dx * t
        base_y = start_point[1] + dy * t
        
        perpendicular_x = -dy / distance
        perpendicular_y = dx / distance
        
        noise = (random.random() - 0.5) * 2 * noise_factor * distance
        
        control_x = base_x + perpendicular_x * noise
        control_y = base_y + perpendicular_y * noise
        
        control_points.append((control_x, control_y))
    
    all_points = [start_point] + control_points + [end_point]
    t = np.linspace(0, 1, 100)
    x = np.array([p[0] for p in all_points])
    y = np.array([p[1] for p in all_points])
    
    cs_x = CubicSpline(np.linspace(0, 1, len(all_points)), x)
    cs_y = CubicSpline(np.linspace(0, 1, len(all_points)), y)
    
    curve_points = [(float(cs_x(t_)), float(cs_y(t_))) for t_ in t]
    return LineString(curve_points)

def get_split_curves(polygon, num_curves=3, noise_factor=0.3):
    minx, miny, maxx, maxy = polygon.bounds
    width = maxx - minx
    height = maxy - miny
    
    curves = []
    if width > height:
        x_positions = np.linspace(minx, maxx, num_curves + 2)[1:-1]
        for x in x_positions:
            start_point = (x, miny - height * 0.1)
            end_point = (x, maxy + height * 0.1)
            curve = create_natural_curve(start_point, end_point, noise_factor=noise_factor)
            curves.append(curve)
    else:
        y_positions = np.linspace(miny, maxy, num_curves + 2)[1:-1]
        for y in y_positions:
            start_point = (minx - width * 0.1, y)
            end_point = (maxx + width * 0.1, y)
            curve = create_natural_curve(start_point, end_point, noise_factor=noise_factor)
            curves.append(curve)
    
    return curves

def split_polygon_by_curves(polygon, max_area, noise_factor=0.3):
    if polygon.area <= max_area:
        return [polygon]
    
    num_parts = int(np.ceil(polygon.area / max_area))
    num_curves = max(1, int(np.ceil(np.log2(num_parts))))
    
    curves = get_split_curves(polygon, num_curves, noise_factor)
    current_polygons = [polygon]
    
    for curve in curves:
        next_polygons = []
        for poly in current_polygons:
            if poly.area <= max_area:
                next_polygons.append(poly)
                continue
            
            try:
                split_result = split(poly, curve)
                split_parts = list(split_result.geoms) if hasattr(split_result, 'geoms') else [split_result]
                
                if all(part.area < poly.area * 0.1 for part in split_parts):
                    next_polygons.append(poly)
                else:
                    next_polygons.extend(split_parts)
            except Exception as e:
                logger.warning(f"分割失败: {str(e)}")
                next_polygons.append(poly)
        
        current_polygons = next_polygons
    
    final_polygons = []
    for poly in current_polygons:
        if poly.area > max_area * 1.2:
            final_polygons.extend(split_polygon_by_curves(poly, max_area, noise_factor))
        else:
            final_polygons.append(poly)
    
    return final_polygons

# === 中线切割方法 ===
def get_optimal_split_line(poly):
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

def split_polygon_by_midline(polygon, max_area, min_area=None):
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

# === 直线切割方法 ===
def split_polygon_by_line(polygon, max_area):
    def get_split_line(poly):
        minx, miny, maxx, maxy = poly.bounds
        width = maxx - minx
        height = maxy - miny
        
        if width > height:
            mid = minx + width / 2
            return LineString([(mid, miny-1), (mid, maxy+1)])
        else:
            mid = miny + height / 2
            return LineString([(minx-1, mid), (maxx+1, mid)])
    
    def split_if_needed(poly):
        if poly.area <= max_area:
            return [poly]
        
        split_line = get_split_line(poly)
        
        try:
            result = split(poly, split_line)
            split_parts = list(result.geoms)
        except Exception:
            return [poly]
        
        split_polygons = []
        for p in split_parts:
            if isinstance(p, (Polygon, MultiPolygon)):
                split_polygons.extend(split_if_needed(p))
        
        return split_polygons
    
    if isinstance(polygon, MultiPolygon):
        result = []
        for poly in polygon.geoms:
            result.extend(split_if_needed(poly))
        return result
    
    return split_if_needed(polygon)

# === 主处理函数 ===
def chunk_iterator(gdf, chunk_size):
    """生成器函数，逐块yield数据"""
    start = 0
    while start < len(gdf):
        yield gdf.iloc[start:start + chunk_size]
        start += chunk_size

def process_shapefile(shp_path, filter_area, max_area, output_path=None, method="curve", 
                     noise_factor=0.2, chunk_size=2000):
    """处理shapefile的主函数"""
    logger.info(f"开始处理文件: {shp_path}")
    
    # 首先只读取几何和面积信息
    gdf = gpd.read_file(shp_path, encoding='utf-8')
    # 添加面积字段
    gdf['area'] = gdf.geometry.area
    # 获取总面积
    total_area = gdf['area'].sum()
    logger.info(f"总面积: {total_area:.2f}")
    
    # 分离需要切割和不需要切割的多边形的索引
    to_split_idx = gdf[gdf.geometry.area > filter_area].index
    to_keep_idx = gdf[gdf.geometry.area <= filter_area].index
    
    logger.info(f"需要切割的多边形数量: {len(to_split_idx)}")
    logger.info(f"无需切割的多边形数量: {len(to_keep_idx)}")
    
    if len(to_split_idx) == 0:
        return gdf
    
    # 设置输出路径
    if output_path is None:
        output_path = shp_path.replace('.shp', f'_split_{method}.shp')
    
    # 选择切割方法
    split_function = {
        "curve": lambda poly: split_polygon_by_curves(poly, max_area, noise_factor),
        "midline": lambda poly: split_polygon_by_midline(poly, max_area),
        "line": lambda poly: split_polygon_by_line(poly, max_area)
    }[method]
    
    # 获取需要处理的数据
    to_split = gdf.loc[to_split_idx]
    to_keep = gdf.loc[to_keep_idx] if len(to_keep_idx) > 0 else None
    del gdf  # 释放原始数据的内存
    
    # 设置进程数和chunk大小
    num_cores = max(1, mp.cpu_count() - 4)  # 保留4个核心给系统
    total_polygons = len(to_split)
    
    # 根据数据量调整处理策略
    if total_polygons <= chunk_size:
        # 数据量小，直接处理
        logger.info("数据量较小，使用单进程处理")
        split_geometries = []
        split_attributes = []
        
        for idx, row in to_split.iterrows():
            try:
                split_polys = split_function(row.geometry)
                for poly in split_polys:
                    if isinstance(poly, (Polygon, MultiPolygon)) and not poly.is_empty:
                        split_geometries.append(poly)
                        split_attributes.append(row.drop('geometry'))
            except Exception as e:
                logger.warning(f"处理多边形 {idx} 时出错: {str(e)}")
                continue
        
        # 创建结果GeoDataFrame
        if split_geometries:
            result_gdf = gpd.GeoDataFrame(
                split_attributes,
                geometry=split_geometries,
                crs=to_split.crs
            )
            
            # 如果有未切割的多边形，合并它们
            if to_keep is not None:
                result_gdf = pd.concat([result_gdf, to_keep], ignore_index=True)
            
            # 保存结果
            result_gdf.to_file(output_path, encoding='utf-8')
        # 输出分割后文件的数量
        logger.info(f"分割后文件的数量: {len(result_gdf)}")
        # 输出分割后矢量的总面积
        result_gdf['area'] = result_gdf.geometry.area
        split_total_area = result_gdf['area'].sum()
        logger.info(f"分割后矢量的总面积: {split_total_area:.2f}")
        # 输出分割后矢量与原始矢量面积的差异
        area_diff = abs(split_total_area - total_area) / total_area
        if area_diff > 0.01:
            logger.warning(f"分割后矢量与原始矢量面积差异较大: {area_diff:.2%}")
        else:
            logger.info(f"分割后矢量与原始矢量面积一致: {area_diff:.2%}")
        
    else:
        # 数据量大，使用多进程处理
        chunk_size = max(1, min(chunk_size, total_polygons // (num_cores * 2)))
        logger.info(f"使用 {num_cores} 个CPU核心进行并行处理")
        logger.info(f"每个数据块大小: {chunk_size}")
        
        # 如果有未切割的多边形，先保存它们
        if to_keep is not None:
            to_keep.to_file(output_path, encoding='utf-8')
            mode = 'a'
        else:
            mode = 'w'
        
        # 分块处理并直接追加到输出文件
        for i, chunk in enumerate(chunk_iterator(to_split, chunk_size)):
            logger.info(f"处理数据块 {i+1}/{(total_polygons-1)//chunk_size + 1}")
            
            split_geometries = []
            split_attributes = []
            
            for idx, row in chunk.iterrows():
                try:
                    split_polys = split_function(row.geometry)
                    for poly in split_polys:
                        if isinstance(poly, (Polygon, MultiPolygon)) and not poly.is_empty:
                            split_geometries.append(poly)
                            split_attributes.append(row.drop('geometry'))
                except Exception as e:
                    logger.warning(f"处理多边形 {idx} 时出错: {str(e)}")
                    continue
            
            # 创建临时GeoDataFrame并保存
            if split_geometries:
                temp_gdf = gpd.GeoDataFrame(
                    split_attributes,
                    geometry=split_geometries,
                    crs=to_split.crs
                )
                temp_gdf.to_file(output_path, mode=mode, encoding='utf-8')
            
            # 清理内存
            del split_geometries
            del split_attributes
            
            mode = 'a'  # 后续都使用追加模式
    
    logger.info("处理完成")
    # 输出分割后文件的数量
    logger.info(f"分割后文件的数量: {len(temp_gdf)}")
    # 输出分割后矢量的总面积
    result_gdf = gpd.read_file(output_path, encoding='utf-8')
    result_gdf['area'] = result_gdf.geometry.area
    split_total_area = result_gdf['area'].sum()   
    logger.info(f"分割后矢量的总面积: {split_total_area:.2f}")
    # 输出分割后矢量与原始矢量面积的差异
    area_diff = abs(split_total_area - total_area) / total_area
    if area_diff > 0.01:
        logger.warning(f"分割后矢量与原始矢量面积差异较大: {area_diff:.2%}")
    else:
        logger.info(f"分割后矢量与原始矢量面积一致: {area_diff:.2%}")

def main(shp_path, filter_area, max_area, method="curve", noise_factor=0.3,chunk_size=2000):
    """
    主函数
    Args:
        shp_path: shapefile文件路径
        filter_area: 需要切割的面积阈值
        max_area: 切割后的最大面积
        method: 切割方法，可选 "curve", "midline", "line"
        noise_factor: 曲线切割时的噪声因子
    """
    if method in ["curve", "midline", "line"]:
        process_shapefile(
            shp_path=shp_path,
            filter_area=filter_area,
            max_area=max_area,
            method=method,
            noise_factor=noise_factor if method == "curve" else None,
            chunk_size=chunk_size
        )
    
if __name__ == "__main__":
    shp_path = r"C:\Users\Runker\Desktop\ele_ky\ky_merge_data_single_2.shp"
    filter_area = 25000  # 需要切割的面积阈值
    max_area = 10000    # 切割后的最大面积
    method = "curve"
    noise_factor = 0.3
    chunk_size = 2000
    
    main(shp_path, filter_area, max_area, method, noise_factor,chunk_size)