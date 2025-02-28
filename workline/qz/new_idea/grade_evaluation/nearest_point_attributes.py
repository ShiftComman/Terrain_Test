import pandas as pd
import geopandas as gpd
import numpy as np
from scipy.spatial import cKDTree
from shapely.geometry import Point
from typing import Union, List, Dict
import os

def get_nearest_point_attributes(
    input_df: Union[str, pd.DataFrame],
    lon_col: str,
    lat_col: str,
    point_shp: str,
    fields: Union[str, List[str]],
    exclude_values: Dict[str, List] = None,
    search_radius: float = None,
    output_csv: str = None
) -> str:
    """
    使用KDTree快速查找最近点并获取属性值，可以排除指定的属性值
    
    参数:
    input_df: 输入的DataFrame或CSV文件路径
    lon_col: 经度列的名称
    lat_col: 纬度列的名称
    point_shp: 输入的点shp文件路径（包含属性数据）
    fields: 需要获取的字段名称，可以是单个字段名或字段名列表
    exclude_values: 需要排除的值的字典，格式为 {字段名: [排除值列表]}
    search_radius: 搜索半径（米），默认None表示不限制
    output_csv: 输出CSV文件路径
    
    返回:
    输出CSV文件的路径或DataFrame
    """
    try:
        # 标准化fields参数为列表
        if isinstance(fields, str):
            fields = [fields]
            
        # 读取输入数据
        if isinstance(input_df, str):
            df = pd.read_csv(input_df)
        else:
            df = input_df.copy()
            
        # 检查经纬度列是否存在
        if lon_col not in df.columns or lat_col not in df.columns:
            raise ValueError(f"未找到经度列 {lon_col} 或纬度列 {lat_col}")
            
        # 读取点shp文件
        points_gdf = gpd.read_file(point_shp)
        
        # 检查所需字段是否存在
        missing_fields = [f for f in fields if f not in points_gdf.columns]
        if missing_fields:
            raise ValueError(f"在点数据中未找到以下字段: {missing_fields}")
        
        # 过滤掉不需要的值的点
        if exclude_values:
            mask = np.ones(len(points_gdf), dtype=bool)
            for field, values in exclude_values.items():
                if field in points_gdf.columns:
                    # 创建一个临时掩码来处理所有类型的空值和特殊字符
                    field_mask = ~points_gdf[field].isna()  # 过滤 None 和 NaN
                    field_mask &= points_gdf[field].astype(str).str.strip() != ''  # 过滤空字符串和只包含空格的字符串
                    field_mask &= ~points_gdf[field].isin(values)  # 过滤指定的值
                    mask &= field_mask
            
            points_gdf = points_gdf[mask].reset_index(drop=True)
            
            if len(points_gdf) == 0:
                raise ValueError("过滤后没有剩余的有效点")
        
        # 提取坐标数组
        input_coords = np.column_stack([df[lon_col].values, df[lat_col].values])
        point_coords = np.column_stack([
            points_gdf.geometry.x.values,
            points_gdf.geometry.y.values
        ])
        
        # 构建KDTree
        tree = cKDTree(point_coords)
        
        # 查找最近点
        if search_radius is not None:
            distances, indices = tree.query(
                input_coords, 
                k=1, 
                distance_upper_bound=search_radius
            )
        else:
            distances, indices = tree.query(input_coords, k=1)
        
        # 处理超出搜索半径的点
        mask = np.isinf(distances)
        indices[mask] = -1
        
        # 添加属性值到DataFrame
        for field in fields:
            field_values = points_gdf[field].values
            df[field] = np.where(
                indices != -1,
                field_values[indices],
                None
            )
        
        # 添加距离字段
        df['nearest_distance'] = np.where(mask, None, distances)
        
        # 保存结果
        if output_csv:
            df.to_csv(output_csv, index=False, encoding='utf-8')
            return output_csv
        else:
            return df
        
    except Exception as e:
        print(f"发生错误: {str(e)}")
        raise

# 使用示例
if __name__ == "__main__":
    # 示例参数
    input_csv = r"G:\soil_property_result\qzs\grade_evaluation\table\grade_evaluation_sample.csv"
    point_shp = r"G:\soil_property_result\qzs\shp\qz_result_point4544.shp"
    output_csv = r"G:\soil_property_result\qzs\grade_evaluation\table\grade_evaluation_sample_near.csv"
    fields = ['TRZD']
    
    # 指定需要排除的值
    exclude_values = {
        'TRZD': ['/', None,'',' ']  # 排除这些值的点
    }
    
    # 调用函数
    output = get_nearest_point_attributes(
        input_df=input_csv,
        lon_col='Centroid_X',
        lat_col='Centroid_Y',
        point_shp=point_shp,
        fields=fields,
        exclude_values=exclude_values,  # 新增参数
        search_radius=5000,
        output_csv=output_csv
    )
    print(f"处理完成，输出文件：{output}") 