import arcpy
from arcpy import env
import arcpy
import traceback
import multiprocessing
import os
import numpy as np
from tqdm import tqdm



def merge_small_parcels_optimized(input_fc, output_fc, land_type_field, dz_field, thresholds, output_gdb=None, batch_size=1000):
    try:
        if output_gdb is None:
            output_gdb = arcpy.env.workspace

        arcpy.CreateFeatureclass_management(output_gdb, output_fc, "POLYGON", input_fc, "DISABLED", "DISABLED", input_fc)
        output_fc_path = os.path.join(output_gdb, output_fc)

        total_features = int(arcpy.GetCount_management(input_fc)[0])

        arcpy.AddSpatialIndex_management(input_fc)

        min_threshold = min(thresholds.values())

        with arcpy.da.SearchCursor(input_fc, ["OID@", land_type_field, dz_field, "SHAPE@"]) as search_cursor, \
             arcpy.da.InsertCursor(output_fc_path, ["OID@", land_type_field, dz_field, "SHAPE@"]) as insert_cursor:
            
            features = []
            for row in tqdm(search_cursor, total=total_features, desc="Processing features"):
                features.append(row)
                
                if len(features) >= batch_size:
                    process_batch(features, land_type_field, dz_field, thresholds, min_threshold, insert_cursor)
                    features = []

            if features:
                process_batch(features, land_type_field, dz_field, thresholds, min_threshold, insert_cursor)

        print(f"操作完成。结果保存在要素类: {output_fc_path}")

    except Exception as e:
        print(f"发生错误: {str(e)}")
        print(traceback.format_exc())

def process_batch(features, land_type_field, dz_field, thresholds, min_threshold, insert_cursor):
    if not features:
        return  # 如果特征列表为空，直接返回

    feature_array = np.array([(f[0], f[1], f[2], f[3].area, f[3]) for f in features], 
                             dtype=[('OID', int), ('land_type', 'U50'), ('dz', 'U50'), ('area', float), ('shape', object)])

    for land_type in np.unique(feature_array['land_type']):
        land_type_features = feature_array[feature_array['land_type'] == land_type]
        
        if len(land_type_features) == 0:
            continue  # 如果该土地类型没有特征，跳过处理

        # 使用 vectorize 函数来高效地应用阈值
        get_threshold = np.vectorize(lambda dz: thresholds.get(dz, min_threshold))
        thresholds_array = get_threshold(land_type_features['dz'])
        
        small_parcels = land_type_features[land_type_features['area'] < thresholds_array]
        
        while len(small_parcels) > 0:
            current_parcel = small_parcels[0]
            neighbors = find_neighbors(current_parcel['shape'], land_type_features)
            
            if len(neighbors) > 0:
                largest_neighbor = max(neighbors, key=lambda x: x['area'])
                merged_shape = current_parcel['shape'].union(largest_neighbor['shape'])
                land_type_features = land_type_features[
                    (land_type_features['OID'] != current_parcel['OID']) & 
                    (land_type_features['OID'] != largest_neighbor['OID'])
                ]
                new_feature = (largest_neighbor['OID'], land_type, largest_neighbor['dz'], merged_shape.area, merged_shape)
                land_type_features = np.append(land_type_features, np.array([new_feature], dtype=land_type_features.dtype))
            else:
                insert_cursor.insertRow((current_parcel['OID'], current_parcel['land_type'], current_parcel['dz'], current_parcel['shape']))
                land_type_features = land_type_features[land_type_features['OID'] != current_parcel['OID']]
            
            if len(land_type_features) == 0:
                break  # 如果所有特征都被处理，退出循环
            
            thresholds_array = get_threshold(land_type_features['dz'])
            small_parcels = land_type_features[land_type_features['area'] < thresholds_array]
        
        for feature in land_type_features:
            insert_cursor.insertRow((feature['OID'], feature['land_type'], feature['dz'], feature['shape']))

def find_neighbors(shape, features):
    return [f for f in features if f['shape'].touches(shape)]

# 使用示例
if __name__ == "__main__":
    env.workspace = r'C:\Users\Runker\Desktop\DEM_test\multi.gdb'
    env.overwriteOutput = True
    input_fc = "DY_single_end"
    output_fc = "DY_single_end_result"
    land_type_field = "DLMC"
    dz_field = "DZ"
    thresholds = {
        "01": 50,
        "03": 1000,
        "04": 1000,
        # 可以继续添加其他DZ值的阈值
    }

    merge_small_parcels_optimized(input_fc, output_fc, land_type_field, dz_field, thresholds,batch_size=1000)