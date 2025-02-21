import os
import datetime
import geopandas as gpd
import pandas as pd

# 获取当前日期
def get_dtime():
    return datetime.datetime.now().strftime('%Y%m%d')

# 文件路径
result_table_path = r"F:\cache_data\zone_ana\qz\prediction_result\prediction_class_RandomForestEntr_20250219.csv"
shp_path = r"F:\cache_data\shp_file\qz\ele_qz\qz_merge_data_single_result.shp"

# 读取数据
gdf = gpd.read_file(shp_path)
df_results = pd.read_csv(result_table_path)

# 合并数据
model_name = 'RandomForestEntr'
merged_gdf = gdf.merge(df_results, on='FID', how='left')

# 保存结果
out_path = os.path.join(os.path.dirname(shp_path), f"qz_soiltype_{model_name}_{get_dtime()}.shp")
merged_gdf.to_file(out_path)