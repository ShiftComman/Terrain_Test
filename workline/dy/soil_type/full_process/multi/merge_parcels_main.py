import arcpy
import os
import subprocess
import json
import uuid
from tqdm import tqdm
import sys
from collections import defaultdict

def merge_small_parcels_multiprocess(input_fc, output_fc, land_type_field, dz_field, thresholds, output_gdb=None, num_processes=4, temp_folder=None):
    try:
        if output_gdb is None:
            output_gdb = arcpy.env.workspace

        if temp_folder is None:
            temp_folder = arcpy.env.scratchFolder
        else:
            # 确保指定的临时文件夹存在
            if not os.path.exists(temp_folder):
                os.makedirs(temp_folder)

        if not os.path.isabs(input_fc):
            input_fc = os.path.join(arcpy.env.workspace, input_fc)

        if not arcpy.Exists(input_fc):
            raise ValueError(f"输入要素类不存在: {input_fc}")

        print(f"输入要素类: {input_fc}")
        print(f"临时文件夹: {temp_folder}")
        total_features = int(arcpy.GetCount_management(input_fc)[0])
        print(f"总要素数: {total_features}")

        # 统计每个 DLMC 类型的总数和需要处理的数量
        dlmc_stats = defaultdict(lambda: {"total": 0, "to_process": 0})
        min_threshold = min(thresholds.values())

        with arcpy.da.SearchCursor(input_fc, [land_type_field, dz_field, "SHAPE@AREA"]) as cursor:
            for row in tqdm(cursor, total=total_features, desc="分析 DLMC 类别"):
                land_type, dz, area = row
                dlmc_stats[land_type]["total"] += 1
                threshold = thresholds.get(dz, min_threshold)
                if area < threshold:
                    dlmc_stats[land_type]["to_process"] += 1

        print(f"DLMC类别数量: {len(dlmc_stats)}")

        # 按需要处理的数量排序 DLMC 类型
        sorted_dlmc = sorted(dlmc_stats.items(), key=lambda x: x[1]["to_process"], reverse=True)

        # 分配 DLMC 类型给进程
        process_allocation = [[] for _ in range(num_processes)]
        process_workloads = [0] * num_processes

        for dlmc, stats in sorted_dlmc:
            # 找到当前工作量最少的进程
            min_workload_process = min(range(num_processes), key=lambda i: process_workloads[i])
            process_allocation[min_workload_process].append(dlmc)
            process_workloads[min_workload_process] += stats["to_process"]

        # 创建临时地理数据库并分配数据
        temp_gdbs = []
        for i in range(num_processes):
            temp_gdb = arcpy.management.CreateFileGDB(temp_folder, f"temp_{i}_{str(uuid.uuid4())[:8]}")
            temp_gdb_path = temp_gdb.getOutput(0)
            temp_gdbs.append(temp_gdb_path)
            
            output_partition = os.path.join(temp_gdb_path, f"partition_{i}")
            dlmc_query = " OR ".join([f"{land_type_field} = '{dlmc}'" for dlmc in process_allocation[i]])
            print(f"为进程 {i} 选择数据")
            print(f"分配的 DLMC 类型: {process_allocation[i]}")
            arcpy.analysis.Select(input_fc, output_partition, dlmc_query)
            total_count = int(arcpy.GetCount_management(output_partition)[0])
            to_process_count = sum(dlmc_stats[dlmc]["to_process"] for dlmc in process_allocation[i])
            print(f"进程 {i} 的数据总数: {total_count}")
            print(f"进程 {i} 需要处理的数据数: {to_process_count}")

        current_dir = os.path.dirname(os.path.abspath(__file__))
        subprocess_script = os.path.join(current_dir, "merge_parcels_subprocess.py")

        processes = []
        for i, temp_gdb in enumerate(temp_gdbs):
            output_partition = os.path.join(temp_gdb, f"partition_{i}")
            params = {
                "input_fc": output_partition,
                "output_fc": output_partition,
                "land_type_field": land_type_field,
                "dz_field": dz_field,
                "thresholds": thresholds,
                "process_num": i
            }
            param_file = os.path.join(temp_folder, f"params_{i}.json")
            with open(param_file, 'w') as f:
                json.dump(params, f)
            
            # 启动子进程
            process = subprocess.Popen([sys.executable, subprocess_script, param_file], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            processes.append(process)

        # 等待所有子进程完成
        for process in tqdm(processes, desc="处理 DLMC 类别"):
            stdout, stderr = process.communicate()
            print(f"子进程输出: {stdout.decode('utf-8', errors='ignore')}")
            if process.returncode != 0:
                print(f"子进程返回错误代码: {process.returncode}")
                print(f"标准错误: {stderr.decode('utf-8', errors='ignore')}")

        # 合并所有分区的结果
        if arcpy.Exists(os.path.join(output_gdb, output_fc)):
            arcpy.Delete_management(os.path.join(output_gdb, output_fc))
        arcpy.CreateFeatureclass_management(output_gdb, output_fc, "POLYGON", input_fc, "DISABLED", "DISABLED", input_fc)
        output_fc_path = os.path.join(output_gdb, output_fc)
        
        total_features = 0
        for i, temp_gdb in enumerate(temp_gdbs):
            partition_fc = os.path.join(temp_gdb, f"partition_{i}")
            if arcpy.Exists(partition_fc):
                feature_count = int(arcpy.GetCount_management(partition_fc)[0])
                total_features += feature_count
                print(f"分区 {i} 的最终要素数: {feature_count}")
                arcpy.Append_management(partition_fc, output_fc_path, "NO_TEST")
            else:
                print(f"警告: 分区 {i} 的输出不存在")

        print(f"合并后的总要素数: {total_features}")
        print(f"最终输出要素数: {arcpy.GetCount_management(output_fc_path)[0]}")

        # 清理临时文件
        for temp_gdb in temp_gdbs:
            arcpy.Delete_management(temp_gdb)
        for i in range(num_processes):
            os.remove(os.path.join(temp_folder, f"params_{i}.json"))

        print(f"操作完成。结果保存在要素类: {output_fc_path}")
        print(f"临时文件保存在: {temp_folder}")

    except Exception as e:
        print(f"发生错误: {str(e)}")
        import traceback
        print(traceback.format_exc())

if __name__ == "__main__":
    arcpy.env.workspace = r'C:\Users\Runker\Desktop\DEM_test\multi.gdb'
    arcpy.env.overwriteOutput = True
    input_fc = r"C:\Users\Runker\Desktop\DEM_test\multi.gdb\DY_single"
    output_fc = "DY_single_end_4"
    land_type_field = "DLMC"
    dz_field = "DZ"
    thresholds = {
        "01": 50,
        "03": 1000,
        "04": 1000,
    }
    # 指定自定义的临时文件夹路径
    custom_temp_folder = r"C:\Users\Runker\Desktop\DEM_test"
    merge_small_parcels_multiprocess(input_fc, output_fc, land_type_field, dz_field, thresholds, num_processes=4, temp_folder=custom_temp_folder)