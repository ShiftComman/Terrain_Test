from shapely.geometry import Polygon, MultiPolygon, box, LineString
from shapely.ops import split
import geopandas as gpd
import numpy as np

def split_polygon_by_area(polygon, max_area):
    """
    将复杂多边形按照指定最大面积进行分割
    
    参数:
    polygon: shapely.geometry.Polygon 或 MultiPolygon 对象
    max_area: float, 每个子多边形的最大面积
    
    返回:
    list of Polygon: 分割后的多边形列表
    """
    
    def get_split_line(poly):
        # 获取多边形的边界框
        minx, miny, maxx, maxy = poly.bounds
        
        # 计算长边，决定切分方向
        width = maxx - minx
        height = maxy - miny
        
        if width > height:
            # 垂直切分
            mid = minx + width / 2
            return LineString([(mid, miny-1), (mid, maxy+1)])
        else:
            # 水平切分
            mid = miny + height / 2
            return LineString([(minx-1, mid), (maxx+1, mid)])
    
    def split_if_needed(poly):
        if poly.area <= max_area:
            return [poly]
        
        # 获取切分线
        split_line = get_split_line(poly)
        
        # 执行切分并转换结果为列表
        try:
            result = split(poly, split_line)
            # 将GeometryCollection转换为列表
            split_parts = list(result.geoms)
        except Exception:
            # 如果切分失败，返回原始多边形
            return [poly]
        
        # 递归处理切分后的每个部分
        split_polygons = []
        for p in split_parts:
            if isinstance(p, (Polygon, MultiPolygon)):
                split_polygons.extend(split_if_needed(p))
        
        return split_polygons
    
    # 处理MultiPolygon情况
    if isinstance(polygon, MultiPolygon):
        result = []
        for poly in polygon.geoms:
            result.extend(split_if_needed(poly))
        return result
    
    # 处理单个Polygon情况
    return split_if_needed(polygon)

def process_shapefile(shp_path, max_area, output_path=None):
    """
    处理输入的shapefile文件，将其中的多边形按照指定面积进行分割
    
    参数:
    shp_path: str, 输入shapefile的路径
    max_area: float, 每个子多边形的最大面积
    output_path: str, 可选，输出shapefile的路径。如果不指定，将在输入文件名后添加'_split'
    
    返回:
    GeoDataFrame: 包含分割后多边形的GeoDataFrame
    """
    # 读取shapefile
    gdf = gpd.read_file(shp_path,encoding='utf-8')
    

    # 存储分割后的结果
    split_geometries = []
    split_attributes = []
    
    # 处理每个多边形
    for idx, row in gdf.iterrows():
        split_polys = split_polygon_by_area(row.geometry, max_area)
        
        # 为每个分割后的多边形复制原始属性
        for poly in split_polys:
            split_geometries.append(poly)
            split_attributes.append(row.drop('geometry'))
    
    # 创建新的GeoDataFrame
    result_gdf = gpd.GeoDataFrame(
        split_attributes, 
        geometry=split_geometries,
        crs=gdf.crs
    )
    
    # 保存结果
    if output_path is None:
        output_path = shp_path.replace('.shp', '_split.shp')
    
    # 保存文件
    result_gdf.to_file(output_path, encoding='utf-8')
    
    
    return result_gdf

def example_usage():
    """
    使用示例
    """
    # 输入shapefile路径
    input_shp = r"C:\Users\Runker\Desktop\ele_sb\clip_test.shp"
    
    # 设置最大面积（单位取决于输入数据的坐标系统）
    max_area = 20000
    
    # 输出路径（可选）
    output_shp = r"C:\Users\Runker\Desktop\ele_sb\clip_test_split.shp"
    
    # 处理shapefile
    result = process_shapefile(input_shp, max_area, output_shp)
    
    # 打印结果统计信息
    print(f"原始多边形数量: {len(gpd.read_file(input_shp))}")
    print(f"分割后多边形数量: {len(result)}")

if __name__ == "__main__":
    example_usage()