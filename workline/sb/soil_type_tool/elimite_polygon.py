import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.ops import unary_union
from shapely.validation import make_valid
from tqdm import tqdm
import time
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def safe_union(geom1, geom2):
    try:
        union = unary_union([geom1, geom2])
        if not union.is_valid:
            union = make_valid(union)
        return union
    except Exception as e:
        logging.error(f"合并几何形状时出错: {str(e)}")
        return None

def find_neighbors(parcel, gdf):
    try:
        possible_matches_index = list(gdf.sindex.intersection(parcel.geometry.bounds))
        possible_matches = gdf.iloc[possible_matches_index]
        return possible_matches[possible_matches.geometry.touches(parcel.geometry)]
    except Exception as e:
        logging.error(f"在查找邻居时出错: {str(e)}")
        return gpd.GeoDataFrame()

def mark_small_parcels_for_merging(gdf, small_parcels, large_parcels, dlmc_field):
    """标记需要合并的小图斑（用于第一阶段）"""
    merge_operations = []
    processed = set()
    
    for index, small_parcel in small_parcels.iterrows():
        if index in processed:
            continue
            
        neighbors = find_neighbors(small_parcel, large_parcels)
        if neighbors.empty:
            continue
        
        same_dlmc_neighbors = neighbors[neighbors[dlmc_field] == small_parcel[dlmc_field]]
        if not same_dlmc_neighbors.empty:
            target = same_dlmc_neighbors.loc[same_dlmc_neighbors['area'].idxmax()]
            merge_operations.append((index, target.name))
            processed.add(index)
    
    return merge_operations

def mark_small_parcels_for_merging_phase2(gdf, small_parcels, dlmc_field):
    """优化后的第二阶段合并标记函数"""
    merge_operations = []
    processed = set()
    
    # 按面积从小到大排序小图斑
    small_parcels_sorted = small_parcels.sort_values('area')
    
    for idx, current_parcel in small_parcels_sorted.iterrows():
        if idx in processed:
            continue
            
        # 查找所有邻居（不仅限于small_parcels中的）
        neighbors = find_neighbors(current_parcel, gdf)
        if neighbors.empty:
            continue
        
        # 筛选具有相同DLMC的邻居
        same_dlmc_neighbors = neighbors[neighbors[dlmc_field] == current_parcel[dlmc_field]]
        if same_dlmc_neighbors.empty:
            continue
        
        # 按面积排序邻居
        same_dlmc_neighbors = same_dlmc_neighbors.sort_values('area', ascending=False)
        
        # 尝试找到最佳的合并目标
        best_target = None
        for neighbor_idx, neighbor in same_dlmc_neighbors.iterrows():
            if neighbor_idx not in processed:
                # 如果邻居也是小图斑，检查合并后是否超过阈值
                if neighbor_idx in small_parcels.index:
                    combined_area = current_parcel['area'] + neighbor['area']
                    if combined_area > current_parcel['threshold']:
                        best_target = neighbor_idx
                        break
                else:
                    # 如果邻居是大图斑，直接选择
                    best_target = neighbor_idx
                    break
        
        if best_target is not None:
            merge_operations.append((idx, best_target))
            processed.add(idx)
            # 只有当目标也是小图斑时才标记为已处理
            if best_target in small_parcels.index:
                processed.add(best_target)
    
    return merge_operations

def execute_merges(gdf, merge_operations):
    """优化后的合并执行函数"""
    merged_count = 0
    skipped_count = 0
    
    gdf['to_delete'] = False
    
    # 按照目标图斑的索引分组，以便批量处理
    merge_groups = {}
    for small_idx, large_idx in merge_operations:
        if large_idx in merge_groups:
            merge_groups[large_idx].append(small_idx)
        else:
            merge_groups[large_idx] = [small_idx]
    
    # 处理每个合并组
    for large_idx, small_indices in merge_groups.items():
        if large_idx not in gdf.index:
            skipped_count += len(small_indices)
            continue
            
        if gdf.loc[large_idx, 'to_delete']:
            skipped_count += len(small_indices)
            continue
        
        # 收集所有要合并的几何图形
        geometries = [gdf.loc[large_idx, 'geometry']]
        valid_small_indices = []
        
        for small_idx in small_indices:
            if small_idx in gdf.index and not gdf.loc[small_idx, 'to_delete']:
                geometries.append(gdf.loc[small_idx, 'geometry'])
                valid_small_indices.append(small_idx)
        
        if len(geometries) > 1:
            # 执行合并
            new_geometry = safe_union(geometries[0], unary_union(geometries[1:]))
            
            if new_geometry is not None and new_geometry.is_valid:
                new_area = new_geometry.area
                original_area = sum(gdf.loc[[large_idx] + valid_small_indices, 'area'])
                
                # 检查面积变化
                area_difference = abs(new_area - original_area) / original_area
                if area_difference <= 0.01:  # 1%的容差
                    gdf.loc[large_idx, 'geometry'] = new_geometry
                    gdf.loc[large_idx, 'area'] = new_area
                    gdf.loc[valid_small_indices, 'to_delete'] = True
                    merged_count += len(valid_small_indices)
                else:
                    skipped_count += len(valid_small_indices)
            else:
                skipped_count += len(valid_small_indices)
    
    # 删除已合并的图斑并更新面积
    gdf = gdf[~gdf['to_delete']]
    gdf = gdf.drop(columns=['to_delete'])
    gdf['area'] = gdf.geometry.area
    
    logging.info(f"合并了 {merged_count} 个图斑，跳过了 {skipped_count} 个图斑")
    
    return gdf, merged_count

def merge_small_parcels_two_phase(input_shp, output_base, dldm_field, dlmc_field, base_thresholds, 
                                default_threshold=50, phase1_max_iterations=5, 
                                phase2_steps=5, phase2_max_iterations=3):
    """优化后的两阶段处理主函数，加强面积计算的准确性"""
    start_time = time.time()
    logging.info(f"开始处理。输入Shapefile: {input_shp}")
    
    try:
        gdf = gpd.read_file(input_shp)
    except Exception as e:
        logging.error(f"读取shapefile时出错: {str(e)}")
        return
    
    original_crs = gdf.crs
    original_count = len(gdf)
    
    logging.info("正在将多部件要素转换为单部件...")
    gdf = gdf.explode(index_parts=True).reset_index(drop=True)
    
    # 初始面积计算
    gdf['area'] = gdf.geometry.area
    original_area = gdf['area'].sum()
    logging.info(f"读取了 {original_count} 个图斑，总面积: {original_area}")
    
    total_merged = 0
    
    # 第一阶段：使用完整阈值
    logging.info("\n=== 第一阶段：使用完整阈值处理 ===")
    iteration = 0
    while iteration < phase1_max_iterations:
        iteration += 1
        logging.info(f"\n开始第一阶段第 {iteration} 轮处理")
        
        # 重新计算面积和应用阈值
        gdf['area'] = gdf.geometry.area
        gdf['threshold'] = gdf[dldm_field].map(lambda x: base_thresholds.get(x, default_threshold))
        
        # 基于最新面积划分大小图斑
        small_parcels = gdf[gdf['area'] < gdf['threshold']]
        large_parcels = gdf[gdf['area'] >= gdf['threshold']]
        
        if len(small_parcels) == 0:
            logging.info("没有找到小面积图斑，处理完成。")
            break
        
        current_area = gdf['area'].sum()
        logging.info(f"当前总面积: {current_area}")
        logging.info(f"小面积图斑数量: {len(small_parcels)}")
        logging.info(f"小面积图斑DLDM分布: {small_parcels[dldm_field].value_counts().to_dict()}")
        
        merge_operations = mark_small_parcels_for_merging(gdf, small_parcels, large_parcels, dlmc_field)
        
        if not merge_operations:
            logging.info("第一阶段没有可以合并的图斑，进入第二阶段。")
            break
        
        gdf, merged_count = execute_merges(gdf, merge_operations)
        total_merged += merged_count
        
        if merged_count == 0:
            break
        
        # 验证总面积变化
        new_total_area = gdf['area'].sum()
        area_change = abs(new_total_area - current_area) / current_area
        logging.info(f"本轮面积变化: {area_change*100:.2f}%")
    
    # 第二阶段：使用递减阈值处理剩余的小图斑
    logging.info("\n=== 第二阶段：处理剩余的小图斑 ===")
    threshold_percentages = np.linspace(100, 20, phase2_steps)
    
    for threshold_percent in threshold_percentages:
        logging.info(f"\n使用 {threshold_percent:.0f}% 阈值进行处理")
        current_thresholds = {k: v * (threshold_percent / 100) for k, v in base_thresholds.items()}
        
        # 重置处理状态
        any_merged = True
        phase2_iteration = 0
        
        while any_merged and phase2_iteration < phase2_max_iterations:
            phase2_iteration += 1
            logging.info(f"第二阶段 {threshold_percent:.0f}% 阈值的第 {phase2_iteration} 轮处理")
            
            # 重新计算面积和应用当前阈值
            gdf['area'] = gdf.geometry.area
            gdf['threshold'] = gdf[dldm_field].map(lambda x: current_thresholds.get(x, default_threshold))
            current_area = gdf['area'].sum()
            
            small_parcels = gdf[gdf['area'] < gdf['threshold']]
            
            if len(small_parcels) == 0:
                logging.info(f"当前阈值 {threshold_percent:.0f}% 下没有小面积图斑，进入下一个阈值")
                break
            
            logging.info(f"当前阈值下的小面积图斑数量: {len(small_parcels)}")
            merge_operations = mark_small_parcels_for_merging_phase2(gdf, small_parcels, dlmc_field)
            
            if not merge_operations:
                logging.info(f"当前阈值 {threshold_percent:.0f}% 下没有可合并的图斑")
                any_merged = False
                continue
            
            gdf, merged_count = execute_merges(gdf, merge_operations)
            total_merged += merged_count
            
            any_merged = merged_count > 0
            
            if merged_count > 0:
                logging.info(f"本轮合并了 {merged_count} 个图斑")
            
            # 验证总面积变化
            new_total_area = gdf['area'].sum()
            area_change = abs(new_total_area - current_area) / current_area
            logging.info(f"本轮面积变化: {area_change*100:.2f}%")
    
    # 最终检查和保存
    gdf = gdf.set_crs(original_crs, allow_override=True)
    final_area = gdf.geometry.area.sum()
    area_change = abs(final_area - original_area) / original_area
    
    logging.info("\n=== 处理完成 ===")
    logging.info(f"总处理时间: {(time.time() - start_time) / 60:.2f} 分钟")
    logging.info(f"初始图斑数量: {original_count}")
    logging.info(f"最终图斑数量: {len(gdf)}")
    logging.info(f"减少的图斑数量: {original_count - len(gdf)}")
    logging.info(f"总面积变化: {area_change*100:.2f}%")
    logging.info(f"总共合并了 {total_merged} 个图斑")
    
    # 保存结果
    output_shp = f"{output_base}"
    result_truncated = gdf.rename(columns={col: col[:10] for col in gdf.columns if len(col) > 10})
    result_truncated.to_file(output_shp, encoding='utf-8')
    # 单部件
    result_single_part = result_truncated.explode(index_parts=True).reset_index(drop=True)
    result_single_part.to_file(output_shp, encoding='utf-8')
    logging.info(f"结果已保存至: {output_shp}")

# 使用示例
if __name__ == "__main__":
    input_shp = r"C:\Users\Runker\Desktop\ele_sb\gl_merge_data_singles.shp"
    output_base = r"C:\Users\Runker\Desktop\ele_sb\gl_merge_data_singles_ele.shp"
    dldm_field = "DLBM"
    dlmc_field = "DLMC"
    base_thresholds = {
        "01": 50,
        "02": 50,
        "03": 500,
        "04": 500
    }
    
    merge_small_parcels_two_phase(
        input_shp,
        output_base,
        dldm_field,
        dlmc_field,
        base_thresholds,
        default_threshold=50,
        phase1_max_iterations=10,     # 第一阶段最大迭代次数
        phase2_steps=20,              # 第二阶段的阈值递减步数
        phase2_max_iterations=5      # 每个阈值级别的最大迭代次数
    )