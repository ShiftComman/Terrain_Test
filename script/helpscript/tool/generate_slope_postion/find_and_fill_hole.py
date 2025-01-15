import geopandas as gpd
import pandas as pd
from shapely.geometry import box
from shapely.ops import unary_union
from tqdm import tqdm

def fill_holes_by_envelope_with_attributes(in_shp, out_filled_shp):
    """
    通过包络矩形方法填充空洞，并继承相邻要素的属性
    """
    print("正在读取输入数据...")
    gdf = gpd.read_file(in_shp)
    
    # 确保所有列名都是字符串类型
    gdf.columns = [str(col) for col in gdf.columns]
    
    # 获取所有几何体的并集
    union_geom = unary_union(gdf.geometry)
    
    # 获取包络矩形
    envelope = box(*union_geom.bounds)
    
    # 获取空洞
    holes = envelope.difference(union_geom)
    
    # 如果holes是MultiPolygon，拆分为单独的Polygon
    if holes.geom_type == 'MultiPolygon':
        hole_parts = list(holes.geoms)
    else:
        hole_parts = [holes]
    
    # 为每个空洞找到相邻的要素并继承其属性
    filled_features = []
    print(f"开始处理 {len(hole_parts)} 个空洞...")
    for hole in tqdm(hole_parts, desc="填充空洞"):
        buffer_hole = hole.buffer(0.0001)
        
        for idx, row in gdf.iterrows():
            if buffer_hole.intersects(row.geometry):
                new_data = {str(k): v for k, v in row.items()}
                new_data['geometry'] = hole
                filled_features.append(new_data)
                break
    
    # 合并原始数据和填充的空洞
    if filled_features:
        # 创建新的GeoDataFrame，确保列名一致
        holes_gdf = gpd.GeoDataFrame(filled_features, crs=gdf.crs)
        result = pd.concat([gdf, holes_gdf], ignore_index=True)
        result = gpd.GeoDataFrame(result, crs=gdf.crs)
    else:
        result = gdf.copy()
    
    print("正在保存结果...")
    result.to_file(out_filled_shp)
    print("处理完成！")

if __name__ == "__main__":
    in_shp = r'F:\tif_features\county_feature\qz\slopepostion_smooth.shp'
    out_filled_shp = in_shp.replace(".shp", "_filled.shp")
    fill_holes_by_envelope_with_attributes(in_shp, out_filled_shp)