import numpy as np
import pandas as pd
import geopandas as gpd
from scipy import stats
from sklearn.preprocessing import StandardScaler
import rasterio
from rasterio.features import rasterize
from shapely.geometry import mapping
import logging
from tqdm import tqdm
import os
from scipy.spatial import cKDTree
from scipy.stats import gaussian_kde
import warnings
warnings.filterwarnings('ignore')

class GeologyProcessModel:
    """地质过程概率模型"""
    
    def __init__(self, slope_data, landuse_data, points_data, initial_map):
        """
        初始化模型
        
        Args:
            slope_data: 坡位数据（栅格）
            landuse_data: 土地利用数据（矢量）
            points_data: 地质点位数据（带岩性信息）
            initial_map: 初始地质图（矢量）
        """
        self.slope_data = slope_data
        self.landuse_data = landuse_data
        self.points_data = points_data
        self.initial_map = initial_map
        self.priors = {}
        self.transition_matrix = None
        
    def estimate_lithology_distribution(self):
        """估计岩性分布先验概率"""
        try:
            logging.info("开始估计岩性分布先验概率")
            
            # 获取研究区域内的岩性类型及其面积比例
            total_area = self.initial_map.area.sum()
            lithology_areas = self.initial_map.groupby('岩性')['geometry'].apply(
                lambda x: gpd.GeoSeries(x).area.sum() / total_area
            )
            
            # 考虑地质点位的信息
            point_counts = self.points_data['岩性'].value_counts(normalize=True)
            
            # 综合考虑面积比例和点位频率
            combined_prob = (lithology_areas + point_counts) / 2
            combined_prob = combined_prob / combined_prob.sum()
            
            self.priors['lithology'] = combined_prob
            logging.info(f"岩性先验概率估计完成，共{len(combined_prob)}种岩性")
            
            return combined_prob
            
        except Exception as e:
            logging.error(f"岩性分布估计失败: {str(e)}")
            raise
    
    def estimate_weathering_patterns(self):
        """估计风化模式"""
        try:
            logging.info("开始估计风化模式")
            
            # 创建坡位-岩性关联矩阵
            slope_lithology_matrix = pd.crosstab(
                self.points_data['坡位'],
                self.points_data['岩性'],
                normalize='index'
            )
            
            # 计算各坡位的风化程度指数
            weathering_index = {}
            for slope_class in range(1, 7):  # 6个坡位类别
                # 获取该坡位的点位
                slope_points = self.points_data[self.points_data['坡位'] == slope_class]
                
                if not slope_points.empty:
                    # 计算该坡位的平均风化程度
                    weathering_index[slope_class] = self._calculate_weathering_index(slope_points)
            
            self.priors['weathering'] = weathering_index
            logging.info("风化模式估计完成")
            
            return weathering_index
            
        except Exception as e:
            logging.error(f"风化模式估计失败: {str(e)}")
            raise
    
    def _calculate_weathering_index(self, points):
        """计算风化指数"""
        # 这里可以根据实际需求定制风化指数的计算方法
        # 示例：根据坡位、高程等因素计算
        try:
            elevation = points['高程'].mean()
            slope = points['坡度'].mean()
            
            # 简化的风化指数计算
            weathering_index = (elevation / 1000 * 0.3 + slope / 90 * 0.7)
            return np.clip(weathering_index, 0, 1)
            
        except Exception as e:
            logging.error(f"风化指数计算失败: {str(e)}")
            raise
    
    def estimate_erosion_rates(self):
        """估计侵蚀速率"""
        try:
            logging.info("开始估计侵蚀速率")
            
            # 基于坡位和土地利用类型估计侵蚀速率
            erosion_rates = {}
            
            # 计算每个坡位的平均坡度
            with rasterio.open(self.slope_data) as src:
                slope_array = src.read(1)
                
            # 计算每个坡位类别的侵蚀潜力
            for slope_class in range(1, 7):
                mask = slope_array == slope_class
                if mask.any():
                    mean_slope = np.mean(slope_array[mask])
                    # 简化的侵蚀速率计算
                    erosion_rates[slope_class] = self._calculate_erosion_rate(
                        mean_slope,
                        slope_class
                    )
            
            self.priors['erosion'] = erosion_rates
            logging.info("侵蚀速率估计完成")
            
            return erosion_rates
            
        except Exception as e:
            logging.error(f"侵蚀速率估计失败: {str(e)}")
            raise
    
    def _calculate_erosion_rate(self, slope, slope_class):
        """计算侵蚀速率"""
        # 简化的侵蚀速率计算模型
        # 可以根据实际需求调整参数
        base_rate = 0.1  # 基础侵蚀速率
        slope_factor = np.sin(np.radians(slope)) * 0.5
        position_factor = (7 - slope_class) / 6  # 坡位因子
        
        return base_rate * (1 + slope_factor) * position_factor
    
    def build_transition_matrix(self):
        """构建转换矩阵"""
        try:
            logging.info("开始构建转换矩阵")
            
            # 获取所有岩性类型
            lithology_types = list(self.priors['lithology'].index)
            n_types = len(lithology_types)
            
            # 初始化转换矩阵
            transition_matrix = np.zeros((n_types, 6, len(self.landuse_data['用地类型'].unique())))
            
            # 基于先验知识填充转换矩阵
            for i, lithology in enumerate(lithology_types):
                for slope_class in range(6):
                    for landuse_type in self.landuse_data['用地类型'].unique():
                        probability = self._calculate_transition_probability(
                            lithology,
                            slope_class + 1,
                            landuse_type
                        )
                        transition_matrix[i, slope_class, landuse_type] = probability
            
            # 归一化
            transition_matrix = transition_matrix / transition_matrix.sum(axis=0, keepdims=True)
            
            self.transition_matrix = transition_matrix
            logging.info("转换矩阵构建完成")
            
            return transition_matrix
            
        except Exception as e:
            logging.error(f"转换矩阵构建失败: {str(e)}")
            raise
    
    def _calculate_transition_probability(self, lithology, slope_class, landuse_type):
        """计算转换概率"""
        try:
            # 获取基础概率
            base_prob = self.priors['lithology'][lithology]
            
            # 考虑风化影响
            weathering_factor = self.priors['weathering'].get(slope_class, 0.5)
            
            # 考虑侵蚀影响
            erosion_factor = self.priors['erosion'].get(slope_class, 0.5)
            
            # 考虑土地利用影响
            landuse_factor = self._get_landuse_factor(landuse_type)
            
            # 组合概率
            combined_prob = base_prob * weathering_factor * erosion_factor * landuse_factor
            
            return combined_prob
            
        except Exception as e:
            logging.error(f"转换概率计算失败: {str(e)}")
            raise
    
    def _get_landuse_factor(self, landuse_type):
        """获取土地利用影响因子"""
        # 可以根据实际需求自定义不同土地利用类型的影响因子
        landuse_factors = {
            '耕地': 0.8,
            '林地': 0.9,
            '草地': 0.7,
            '水域': 0.5,
            '建设用地': 0.3,
            '未利用地': 1.0
        }
        return landuse_factors.get(landuse_type, 0.5)
    
    def update_geology_map(self, confidence_threshold=0.85):
        """更新地质图"""
        try:
            logging.info("开始更新地质图")
            
            # 读取栅格数据
            with rasterio.open(self.slope_data) as src:
                slope_array = src.read(1)
                transform = src.transform
                
            # 栅格化土地利用数据
            landuse_array = rasterize(
                [(mapping(geom), value) for geom, value in zip(
                    self.landuse_data.geometry,
                    self.landuse_data['用地类型']
                )],
                out_shape=slope_array.shape,
                transform=transform
            )
            
            # 初始化结果数组
            result = np.zeros_like(slope_array, dtype=np.int32)
            confidence = np.zeros_like(slope_array, dtype=np.float32)
            
            # 逐像元更新
            for i in tqdm(range(slope_array.shape[0])):
                for j in range(slope_array.shape[1]):
                    slope_class = slope_array[i, j]
                    landuse_type = landuse_array[i, j]
                    
                    if slope_class > 0:  # 有效值
                        probs = self.transition_matrix[:, slope_class-1, landuse_type]
                        result[i, j] = np.argmax(probs)
                        confidence[i, j] = np.max(probs)
            
            # 转换为矢量
            updated_map = self._raster_to_vector(
                result,
                confidence,
                confidence_threshold,
                transform
            )
            
            logging.info("地质图更新完成")
            return updated_map
            
        except Exception as e:
            logging.error(f"地质图更新失败: {str(e)}")
            raise
    
    def _raster_to_vector(self, result_array, confidence_array, threshold, transform):
        """将栅格结果转换为矢量"""
        try:
            # 创建掩膜
            mask = confidence_array >= threshold
            
            # 提取符合阈值的区域
            valid_results = np.where(mask, result_array, -1)
            
            # 转换为矢量
            shapes = rasterio.features.shapes(valid_results, transform=transform)
            
            # 创建GeoDataFrame
            geometries = []
            values = []
            confidences = []
            
            for shape, value in shapes:
                if value != -1:  # 忽略无效值
                    geometries.append(shape)
                    values.append(value)
                    # 计算该区域的平均置信度
                    region_mask = rasterio.features.geometry_mask(
                        [shape],
                        out_shape=confidence_array.shape,
                        transform=transform,
                        invert=True
                    )
                    mean_confidence = np.mean(confidence_array[region_mask])
                    confidences.append(mean_confidence)
            
            # 创建GeoDataFrame
            gdf = gpd.GeoDataFrame({
                'geometry': geometries,
                '岩性': values,
                '置信度': confidences
            })
            
            # 设置投影
            gdf.crs = self.initial_map.crs
            
            return gdf
            
        except Exception as e:
            logging.error(f"栅格转矢量失败: {str(e)}")
            raise

def main():
    """主函数"""
    try:
        # 设置日志
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        
        # 读取输入数据
        slope_data = "path/to/slope.tif"
        landuse_data = gpd.read_file("path/to/landuse.shp")
        points_data = gpd.read_file("path/to/geology_points.shp")
        initial_map = gpd.read_file("path/to/initial_geology.shp")
        
        # 初始化模型
        model = GeologyProcessModel(slope_data, landuse_data, points_data, initial_map)
        
        # 估计先验概率
        model.estimate_lithology_distribution()
        model.estimate_weathering_patterns()
        model.estimate_erosion_rates()
        
        # 构建转换矩阵
        model.build_transition_matrix()
        
        # 更新地质图
        updated_map = model.update_geology_map(confidence_threshold=0.85)
        
        # 保存结果
        updated_map.to_file("path/to/output/updated_geology.shp")
        
        logging.info("处理完成")
        
    except Exception as e:
        logging.error(f"处理失败: {str(e)}")
        raise

if __name__ == "__main__":
    main()