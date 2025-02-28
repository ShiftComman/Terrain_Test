import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.ops import unary_union
from shapely.geometry import Polygon, MultiPolygon
from shapely.validation import make_valid
from shapely import oriented_envelope
from tqdm import tqdm
import time
import logging
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ----------------- 工具函数 -----------------
def safe_union(geom1, geom2):
    """安全合并两个几何体，自动修复有效性"""
    try:
        union = unary_union([geom1, geom2])
        if not union.is_valid:
            union = make_valid(union)
        return union.buffer(0)  # 确保几何有效性
    except Exception as e:
        logging.error(f"合并失败: {e}\n几何1: {geom1.wkt[:80]}\n几何2: {geom2.wkt[:80]}")
        return None

def calculate_min_width(geometry):
    """计算多边形最小宽度，带几何修复"""
    try:
        geom = geometry.buffer(0)
        if isinstance(geom, MultiPolygon):
            geom = max(geom.geoms, key=lambda x: x.area)
        rect = oriented_envelope(geom)
        coords = np.array(rect.exterior.coords)[:-1]
        edges = np.diff(coords, axis=0, append=[coords[0]])
        lengths = np.sqrt(np.sum(edges**2, axis=1))
        return min(lengths)
    except Exception as e:
        logging.warning(f"最小宽度计算失败: {str(e)}")
        return 0  # 返回0强制合并

# ----------------- 核心逻辑 -----------------
class ParcelMerger:
    def __init__(self, input_shp, output_shp, dldm_field, dlmc_field, 
                 thresholds, min_width, default_thresh=50, max_iters=20):
        self.input_shp = input_shp
        self.output_shp = output_shp
        self.dldm_field = dldm_field
        self.dlmc_field = dlmc_field
        self.thresholds = thresholds
        self.min_width = min_width
        self.default_thresh = default_thresh
        self.max_iters = max_iters
        
        # 加载数据
        self.gdf = self.load_data()
        self.original_count = len(self.gdf)
        self.original_area = self.gdf.geometry.area.sum()
        
    def load_data(self):
        """加载并预处理数据"""
        assert os.path.exists(self.input_shp), "输入文件不存在"
        gdf = gpd.read_file(self.input_shp)
        assert gdf.crs is not None, "必须包含有效CRS"
        assert self.dldm_field in gdf.columns, f"字段{self.dldm_field}不存在"
        assert self.dlmc_field in gdf.columns, f"字段{self.dlmc_field}不存在"
        
        # 分解为单部件
        gdf = gdf.explode(index_parts=False).reset_index(drop=True)
        gdf['geometry'] = gdf.geometry.apply(lambda x: make_valid(x.buffer(0)))
        return gdf
    
    def find_candidates(self):
        """识别需要合并的小图斑"""
        # 动态计算阈值
        self.gdf['threshold'] = self.gdf[self.dldm_field].map(
            lambda x: self.thresholds.get(x, self.default_thresh))
        
        # 计算几何属性
        self.gdf['area'] = self.gdf.geometry.area
        self.gdf['min_width'] = self.gdf.geometry.apply(calculate_min_width)
        
        # 合并条件：面积或宽度不达标
        is_small_area = self.gdf['area'] < self.gdf['threshold']
        is_narrow = self.gdf['min_width'] < self.min_width
        return self.gdf[is_small_area | is_narrow].copy()
    
    def find_best_neighbor(self, parcel, candidates):
        """寻找最优合并邻居"""
        # 空间查询
        possible = list(self.gdf.sindex.query(parcel.geometry, predicate='touches'))
        if not possible:
            return None
        
        # 筛选条件
        neighbors = self.gdf.iloc[possible]
        valid = neighbors[
            (neighbors[self.dlmc_field] == parcel[self.dlmc_field]) &
            (~neighbors.index.isin(candidates.index))  # 排除其他待合并图斑
        ]
        if valid.empty:
            return None
        
        # 优先选择面积最大的邻居
        return valid.loc[valid['area'].idxmax()]
    
    def merge_operation(self):
        """执行一轮合并操作"""
        candidates = self.find_candidates()
        merge_ops = []
        
        # 构建候选集索引
        candidate_indices = set(candidates.index)
        
        for idx in candidates.index:
            if idx not in candidate_indices:
                continue  # 已被合并
            
            parcel = candidates.loc[idx]
            neighbor = self.find_best_neighbor(parcel, candidates)
            if neighbor is None:
                continue
                
            # 记录合并对
            merge_ops.append((idx, neighbor.name))
            # 移除已匹配的小图斑
            candidate_indices.discard(idx)
        
        return merge_ops
    
    def execute_merges(self, merge_ops):
        """执行批量合并"""
        merged = set()
        delete_list = []
        
        for small_idx, large_idx in tqdm(merge_ops, desc="合并进度"):
            if small_idx in merged or large_idx in merged:
                continue
                
            small = self.gdf.loc[small_idx]
            large = self.gdf.loc[large_idx]
            
            new_geom = safe_union(small.geometry, large.geometry)
            if not new_geom or new_geom.is_empty:
                continue
                
            # 更新大图斑
            self.gdf.at[large_idx, 'geometry'] = new_geom
            # 标记删除小图斑
            delete_list.append(small_idx)
            merged.update([small_idx, large_idx])
        
        # 清理数据
        self.gdf = self.gdf[~self.gdf.index.isin(delete_list)]
        # 分解多部件
        self.gdf = self.gdf.explode(index_parts=False).reset_index(drop=True)
        # 更新空间索引
        self.gdf.sindex
        return len(delete_list)
    
    def run(self):
        """主运行流程"""
        start_time = time.time()
        total_merged = 0
        
        for iter_num in range(1, self.max_iters+1):
            logging.info(f"=== 第 {iter_num} 轮合并 ===")
            
            merge_ops = self.merge_operation()
            if not merge_ops:
                logging.info("无更多可合并图斑")
                break
                
            merged_count = self.execute_merges(merge_ops)
            total_merged += merged_count
            logging.info(f"本轮合并 {merged_count} 个图斑，累计合并 {total_merged} 个")
            
            # 提前终止检查
            current_small = len(self.find_candidates())
            logging.info(f"当前待处理图斑数量: {current_small}")
            if current_small == 0:
                break
        
        # 保存结果
        self.save_results()
        logging.info(f"处理完成，总耗时 {(time.time()-start_time)/60:.1f} 分钟")
    
    def save_results(self):
        """处理字段名并保存"""
        # 字段名截断处理
        trunc_cols = {col: col[:10] for col in self.gdf.columns}
        # 处理重复字段名
        seen = {}
        for col in self.gdf.columns:
            trunc = trunc_cols[col]
            if trunc in seen:
                seen[trunc] += 1
                new_col = f"{trunc[:8]}_{seen[trunc]:02d}"
            else:
                seen[trunc] = 0
                new_col = trunc
            trunc_cols[col] = new_col
        
        result = self.gdf.rename(columns=trunc_cols)
        result.to_file(self.output_shp, encoding='utf-8')

# ----------------- 使用示例 -----------------
if __name__ == "__main__":
    merger = ParcelMerger(
        input_shp = r"F:\cache_data\shp_file\qz\ele_qz\merged_result_1.shp",
        output_shp = r"F:\cache_data\shp_file\qz\ele_qz\merged_result_2.shp",
        dldm_field = "DLBM",
        dlmc_field = "DLMC",
        thresholds = {"0101": 50, "0102": 50, "0103": 50},
        min_width = 6,
        default_thresh = 500,
        max_iters = 1
    )
    merger.run()