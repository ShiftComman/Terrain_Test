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

logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_natural_curve(start_point, end_point, control_points_count=3, noise_factor=0.3):
    """
    创建自然曲线
    
    参数:
    start_point: 起点坐标 (x, y)
    end_point: 终点坐标 (x, y)
    control_points_count: 控制点数量
    noise_factor: 曲线扭曲程度 (0-1)
    """
    # 基本方向向量
    dx = end_point[0] - start_point[0]
    dy = end_point[1] - start_point[1]
    distance = np.sqrt(dx**2 + dy**2)
    
    # 生成控制点
    control_points = []
    for i in range(control_points_count):
        # 基准点
        t = (i + 1) / (control_points_count + 1)
        base_x = start_point[0] + dx * t
        base_y = start_point[1] + dy * t
        
        # 添加随机扰动
        # 扰动方向垂直于基本方向
        perpendicular_x = -dy / distance
        perpendicular_y = dx / distance
        
        # 随机扰动幅度
        noise = (random.random() - 0.5) * 2 * noise_factor * distance
        
        control_x = base_x + perpendicular_x * noise
        control_y = base_y + perpendicular_y * noise
        
        control_points.append((control_x, control_y))
    
    # 所有点（包括起点和终点）
    all_points = [start_point] + control_points + [end_point]
    
    # 用参数方程生成更多点以创建平滑曲线
    t = np.linspace(0, 1, 100)
    x = np.array([p[0] for p in all_points])
    y = np.array([p[1] for p in all_points])
    
    # 使用三次样条插值
    cs_x = CubicSpline(np.linspace(0, 1, len(all_points)), x)
    cs_y = CubicSpline(np.linspace(0, 1, len(all_points)), y)
    
    # 生成曲线点
    curve_points = [(float(cs_x(t_)), float(cs_y(t_))) for t_ in t]
    
    return LineString(curve_points)

def get_split_curves(polygon, num_curves=3, noise_factor=0.3):
    """生成用于分割的曲线"""
    minx, miny, maxx, maxy = polygon.bounds
    width = maxx - minx
    height = maxy - miny
    
    curves = []
    # 根据多边形的形状决定分割方向
    if width > height:
        # 垂直分割
        x_positions = np.linspace(minx, maxx, num_curves + 2)[1:-1]
        for x in x_positions:
            start_point = (x, miny - height * 0.1)  # 延伸出多边形边界
            end_point = (x, maxy + height * 0.1)
            curve = create_natural_curve(start_point, end_point, 
                                      control_points_count=3, 
                                      noise_factor=noise_factor)
            curves.append(curve)
    else:
        # 水平分割
        y_positions = np.linspace(miny, maxy, num_curves + 2)[1:-1]
        for y in y_positions:
            start_point = (minx - width * 0.1, y)
            end_point = (maxx + width * 0.1, y)
            curve = create_natural_curve(start_point, end_point,
                                      control_points_count=3,
                                      noise_factor=noise_factor)
            curves.append(curve)
    
    return curves

def split_polygon_by_curves(polygon, max_area, noise_factor=0.3):
    """使用曲线分割多边形"""
    if polygon.area <= max_area:
        return [polygon]
    
    # 计算需要的分割次数
    num_parts = int(np.ceil(polygon.area / max_area))
    num_curves = max(1, int(np.log2(num_parts)))  # 使用对数关系确定曲线数量
    
    # 获取分割曲线
    curves = get_split_curves(polygon, num_curves, noise_factor)
    
    # 逐条曲线分割
    current_polygons = [polygon]
    for curve in curves:
        next_polygons = []
        for poly in current_polygons:
            try:
                # 分割并添加结果
                split_result = split(poly, curve)
                split_parts = list(split_result.geoms) if hasattr(split_result, 'geoms') else [split_result]
                next_polygons.extend(split_parts)
            except Exception as e:
                logger.warning(f"分割失败: {str(e)}")
                next_polygons.append(poly)
        current_polygons = next_polygons
    
    # 进一步分割大于max_area的部分
    final_polygons = []
    for poly in current_polygons:
        if poly.area > max_area:
            final_polygons.extend(split_polygon_by_curves(poly, max_area, noise_factor))
        else:
            final_polygons.append(poly)
    
    return final_polygons

def process_shapefile(shp_path, max_area, output_path=None, batch_size=100, noise_factor=0.3):
    """处理shapefile的主函数"""
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
        split_polys = split_polygon_by_curves(row.geometry, max_area, noise_factor)
        logger.info(f"分割耗时: {time.time() - start_time:.2f}秒")
        logger.info(f"分割后得到 {len(split_polys)} 个子多边形")
        
        # 为每个分割后的多边形复制原始属性
        for poly in split_polys:
            if isinstance(poly, (Polygon, MultiPolygon)) and not poly.is_empty:
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
    output_shp = r"C:\Users\Runker\Desktop\ele_sb\clip_test_splitss.shp"
    
    logger.info("开始示例运行")
    result = process_shapefile(
        input_shp, 
        max_area, 
        output_shp, 
        batch_size=100,
        noise_factor=0.3  # 控制曲线的扭曲程度，0-1之间
    )
    
    logger.info(f"原始多边形数量: {len(gpd.read_file(input_shp))}")
    logger.info(f"分割后多边形数量: {len(result)}")

if __name__ == "__main__":
    example_usage()