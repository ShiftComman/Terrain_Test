import geopandas as gpd
import pandas as pd
from typing import Dict, Any

def load_shapefile(file_path: str) -> gpd.GeoDataFrame:
    """
    加载 shapefile 文件
    """
    try:
        gdf = gpd.read_file(file_path)
        return gdf
    except Exception as e:
        print(f"读取文件时出错: {str(e)}")
        return None

def get_user_dtype_choice(column_name: str, current_dtype: str) -> str:
    """
    获取用户对每列数据类型的选择
    """
    print(f"\n列名: {column_name}")
    print(f"当前数据类型: {current_dtype}")
    print("\n可选的数据类型:")
    print("1. int - 整数")
    print("2. float - 浮点数")
    print("3. str - 字符串")
    print("4. bool - 布尔值")
    print("5. datetime - 日期时间")
    print("6. 保持原样")
    
    choice = input("请选择数据类型 (1-6): ").strip()
    
    dtype_map = {
        "1": "int",
        "2": "float",
        "3": "str",
        "4": "bool",
        "5": "datetime",
        "6": str(current_dtype)
    }
    
    return dtype_map.get(choice, str(current_dtype))

def convert_column_dtype(gdf: gpd.GeoDataFrame, column: str, dtype: str) -> gpd.GeoDataFrame:
    """
    转换列的数据类型
    """
    try:
        if dtype == "int":
            gdf[column] = pd.to_numeric(gdf[column], errors='coerce').astype('Int64')
        elif dtype == "float":
            gdf[column] = pd.to_numeric(gdf[column], errors='coerce')
        elif dtype == "bool":
            gdf[column] = gdf[column].astype(bool)
        elif dtype == "datetime":
            gdf[column] = pd.to_datetime(gdf[column], errors='coerce')
        elif dtype == "str":
            gdf[column] = gdf[column].astype(str)
    except Exception as e:
        print(f"转换列 {column} 到类型 {dtype} 时出错: {str(e)}")
    
    return gdf

def main(shp_path:str,output_path:str):
    # 获取输入文件路径
    file_path = shp_path
    
    # 加载shapefile
    gdf = load_shapefile(file_path)
    if gdf is None:
        return
    
    # 显示当前列信息
    print("\n当前文件包含以下列:")
    for column in gdf.columns:
        if column != 'geometry':
            print(f"{column}: {gdf[column].dtype}")
    
    # 获取用户对每列的数据类型选择
    dtype_changes = {}
    for column in gdf.columns:
        if column != 'geometry':
            new_dtype = get_user_dtype_choice(column, gdf[column].dtype)
            if str(new_dtype) != str(gdf[column].dtype):
                dtype_changes[column] = new_dtype
    
    # 应用数据类型转换
    if dtype_changes:
        print("\n正在转换数据类型...")
        for column, dtype in dtype_changes.items():
            gdf = convert_column_dtype(gdf, column, dtype)
        
        # 保存修改后的文件
        gdf.to_file(output_path)
        print(f"\n修改后的文件已保存至: {output_path}")
        
        # 显示更新后的数据类型
        print("\n更新后的数据类型:")
        for column in gdf.columns:
            if column != 'geometry':
                print(f"{column}: {gdf[column].dtype}")
    else:
        print("\n没有进行任何数据类型修改")

if __name__ == "__main__":
    shp_path=r"F:\collection_spb_info\XJSH\ALL_DATA\ALL_RESULT\ALL_SHP\11_北京市_result_20250107.shp"
    output_path=r"F:\collection_spb_info\XJSH\ALL_DATA\ALL_RESULT\ALL_SHP\11_北京市_result_20250107_modified.shp"
    main(shp_path,output_path)