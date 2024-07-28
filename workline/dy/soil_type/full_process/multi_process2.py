import arcpy
from arcpy import env
import traceback
import numpy as np
from tqdm import tqdm
import os

def merge_small_parcels_optimized(input_fc, output_fc, land_type_field, dz_field, thresholds, output_gdb=None, batch_size=1000):
    try:
        if output_gdb is None:
            output_gdb = arcpy.env.workspace

        arcpy.CreateFeatureclass_management(output_gdb, output_fc, "POLYGON", input_fc, "DISABLED", "DISABLED", input_fc)
        output_fc_path = os.path.join(output_gdb, output_fc)

        total_features = int(arcpy.GetCount_management(input_fc)[0])
        arcpy.AddSpatialIndex_management(input_fc)

        fields = [f.name for f in arcpy.ListFields(input_fc) if f.type not in ['Geometry']]
        fields.append("SHAPE@")

        min_threshold = min(thresholds.values())

        # 读取所有特征
        all_features = []
        with arcpy.da.SearchCursor(input_fc, fields) as search_cursor:
            for row in tqdm(search_cursor, total=total_features, desc="Reading features"):
                all_features.append(row)

        # 进行多次迭代，直到没有更多的合并
        iteration = 1
        while True:
            print(f"开始第 {iteration} 次迭代")
            merged_features, merge_count = process_all_features(all_features, land_type_field, dz_field, thresholds, min_threshold, fields)
            print(f"第 {iteration} 次迭代完成，合并了 {merge_count} 个图斑")
            
            if merge_count == 0:
                break
            
            all_features = merged_features
            iteration += 1

        # 将最终结果写入输出要素类
        with arcpy.da.InsertCursor(output_fc_path, fields) as insert_cursor:
            for feature in tqdm(all_features, desc="Writing final features"):
                insert_cursor.insertRow(feature)

        print(f"操作完成。结果保存在要素类: {output_fc_path}")
        print(f"最终要素数量: {len(all_features)}")

    except Exception as e:
        print(f"发生错误: {str(e)}")
        print(traceback.format_exc())

def process_all_features(features, land_type_field, dz_field, thresholds, min_threshold, fields):
    dtype = [(f, 'O') for f in fields]
    feature_array = np.array(features, dtype=dtype)

    land_type_index = fields.index(land_type_field)
    dz_index = fields.index(dz_field)
    shape_index = fields.index("SHAPE@")

    merge_count = 0

    for land_type in np.unique(feature_array[land_type_field]):
        land_type_features = feature_array[feature_array[land_type_field] == land_type]
        
        if len(land_type_features) == 0:
            continue

        get_threshold = np.vectorize(lambda dz: thresholds.get(dz, min_threshold))
        thresholds_array = get_threshold(land_type_features[dz_field])
        
        small_parcels = land_type_features[np.array([f[shape_index].area for f in land_type_features]) < thresholds_array]
        
        while len(small_parcels) > 0:
            current_parcel = small_parcels[0]
            neighbors = find_neighbors(current_parcel[shape_index], land_type_features, land_type_field, land_type)
            
            if len(neighbors) > 0:
                largest_neighbor = max(neighbors, key=lambda x: x[shape_index].area)
                merged_shape = current_parcel[shape_index].union(largest_neighbor[shape_index])
                
                merged_feature = list(largest_neighbor)
                merged_feature[shape_index] = merged_shape
                
                land_type_features = land_type_features[
                    (land_type_features[fields[0]] != current_parcel[fields[0]]) & 
                    (land_type_features[fields[0]] != largest_neighbor[fields[0]])
                ]
                
                land_type_features = np.append(land_type_features, np.array([tuple(merged_feature)], dtype=land_type_features.dtype))
                merge_count += 1
            else:
                land_type_features = land_type_features[land_type_features[fields[0]] != current_parcel[fields[0]]]
            
            if len(land_type_features) == 0:
                break
            
            thresholds_array = get_threshold(land_type_features[dz_field])
            small_parcels = land_type_features[np.array([f[shape_index].area for f in land_type_features]) < thresholds_array]
        
        feature_array = np.append(feature_array[feature_array[land_type_field] != land_type], land_type_features)

    return feature_array, merge_count

def find_neighbors(shape, features, land_type_field, land_type):
    shape_index = -1
    return [f for f in features if f[shape_index].touches(shape) and f[land_type_field] == land_type]

if __name__ == "__main__":
    env.workspace = r'D:\ArcGISProjects\workspace\shbyq\DZ.gdb'
    env.overwriteOutput = True
    input_fc = "DY_SD_MZ_SLOPEPOSITION_INTERSECT_SINGLE_COPY"
    output_fc = "DY_SD_MZ_SLOPEPOSITION_INTERSECT_SINGLE_COPY_endd"
    land_type_field = "DLMC"
    dz_field = "DZ"
    thresholds = {
        "01": 51,
        "03": 2000,
        "04": 2000,
    }

    merge_small_parcels_optimized(input_fc, output_fc, land_type_field, dz_field, thresholds, batch_size=1000)
