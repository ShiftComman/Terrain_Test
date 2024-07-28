import arcpy
import json
import sys
import traceback
import os
from tqdm import tqdm

def process_partition(input_fc, output_fc, land_type_field, dz_field, thresholds, process_num):
    try:
        print(f"进程 {process_num} 开始处理")
        
        arcpy.env.overwriteOutput = True

        if not arcpy.Exists(input_fc):
            raise ValueError(f"输入要素类不存在: {input_fc}")

        total_features = int(arcpy.GetCount_management(input_fc)[0])
        print(f"进程 {process_num} 要处理的要素数: {total_features}")

        min_threshold = min(thresholds.values())
        merged_count = 0

        with arcpy.da.Editor(os.path.dirname(output_fc)) as edit:
            update_fields = ["OBJECTID", land_type_field, dz_field, "SHAPE@", "SHAPE@AREA"]
            with arcpy.da.UpdateCursor(input_fc, update_fields) as cursor:
                for row in tqdm(cursor, total=total_features, desc=f"进程 {process_num} 处理要素"):
                    oid, land_type, dz, shape, area = row
                    threshold = thresholds.get(dz, min_threshold)
                    
                    if area < threshold:
                        neighbors = find_neighbors(shape, input_fc, land_type_field, land_type)
                        if neighbors:
                            largest_neighbor = max(neighbors, key=lambda x: x[1])
                            merged_shape = shape.union(largest_neighbor[0])
                            cursor.deleteRow()
                            try:
                                with arcpy.da.UpdateCursor(input_fc, ["OBJECTID", "SHAPE@"], f"OBJECTID = {largest_neighbor[2]}") as update_cursor:
                                    for update_row in update_cursor:
                                        update_cursor.updateRow([update_row[0], merged_shape])
                                        merged_count += 1
                                        break
                            except RuntimeError as e:
                                print(f"进程 {process_num} 更新游标错误: {str(e)}")
                                print(f"OBJECTID: {largest_neighbor[2]}, 字段: OBJECTID, SHAPE@")
                                raise

        print(f"进程 {process_num} 合并了 {merged_count} 个要素")
        print(f"进程 {process_num} 完成处理，最终要素数量: {arcpy.GetCount_management(input_fc)[0]}")

    except Exception as e:
        print(f"进程 {process_num} 发生错误: {str(e)}")
        print(traceback.format_exc())

def find_neighbors(shape, fc, land_type_field, land_type):
    neighbors = []
    fields = ["SHAPE@", "SHAPE@AREA", "OBJECTID", land_type_field]
    with arcpy.da.SearchCursor(fc, fields, f"{land_type_field} = '{land_type}'") as cursor:
        for row in cursor:
            if shape.touches(row[0]):
                neighbors.append((row[0], row[1], row[2]))
    return neighbors

if __name__ == "__main__":
    param_file = sys.argv[1]
    
    with open(param_file, 'r') as f:
        params = json.load(f)
    
    process_partition(**params)