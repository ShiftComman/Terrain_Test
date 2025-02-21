import os
import time
import numpy as np
from osgeo import gdal, ogr
from scipy.ndimage import gaussian_filter, distance_transform_edt
import warnings
warnings.filterwarnings('ignore')

gdal.AllRegister()

"""
排水能力评价模块
用于评估区域排水条件，考虑地形、土壤、地质、降雨等因素
"""

class DrainageCapabilityAssessment:
    """
    排水能力评价类
    主要功能：
    1. 评估地形条件对排水的影响
    2. 评估土壤和地质条件对排水的影响
    3. 评估排水网络的分布特征
    4. 评估降雨对排水的压力
    5. 综合评价区域排水能力
    """
    
    def __init__(self, workspace, params=None):
        """初始化工作环境和参数"""
        self.workspace = workspace
        gdal.AllRegister()
        
        # 设置GDAL配置
        gdal.SetConfigOption("GTIFF_SRS_SOURCE", "EPSG")
        
        # 默认参数
        default_params = {
            # 排水沟渠影响参数
            'distance_threshold': 1000,     # 缩小影响距离（米）
            'density_radius': 800,          # 缩小密度计算半径（米）
            
            # 坡度参数
            'slope_threshold': {
                'min': 1,                   # 最小坡度（度）
                'opt': 8,                   # 最适宜坡度（度）
                'max': 25                   # 最大坡度（度）
            },
            
            # 坡位权重（更符合实际排水情况）
            'slope_position_weights': {
                1: 0.3,    # 平地（排水差）
                2: 0.2,    # 坡底（最差，易积水）
                3: 0.5,    # 坡中平地
                4: 0.7,    # 坡中上部
                5: 0.9,    # 坡上部
                6: 1.0     # 坡顶部（最好）
            },
            
            # 评价权重（调整各因子重要性）
            'weights': {
                'slope': 0.30,              # 坡度（最重要）
                'slope_position': 0.25,     # 坡位
                'drainage_net': 0.20,       # 排水网络
                'twi': 0.15,                # 地形湿度
                'rainfall': 0.10            # 降雨压力
            },
            
            # 分级阈值
            'level_thresholds': {
                'poor': 0.35,               # 提高阈值
                'medium': 0.50,
                'good': 0.65
            }
        }
        
        # 更新参数
        self.params = default_params
        if params:
            self.params.update(params)
            
        # 验证权重和
        weights = self.params['weights']
        weight_sum = sum(weights.values())
        if abs(weight_sum - 1.0) > 0.001:
            print(f"警告：权重总和 ({weight_sum:.3f}) 不等于1，已自动归一化")
            for key in weights:
                weights[key] /= weight_sum
    
    def assess_slope_suitability(self, slope_data):
        """评估坡度适宜性（排水视角）"""
        slope_params = self.params['slope_threshold']
        
        # 坡度适宜性评价（分段函数）
        slope_suit = np.zeros_like(slope_data)
        
        # 小于最小坡度（排水差）
        mask_low = slope_data < slope_params['min']
        slope_suit[mask_low] = 0.3 * slope_data[mask_low] / slope_params['min']
        
        # 最小到最适坡度（线性增加）
        mask_opt = (slope_data >= slope_params['min']) & (slope_data <= slope_params['opt'])
        slope_suit[mask_opt] = 0.3 + 0.7 * (slope_data[mask_opt] - slope_params['min']) / (slope_params['opt'] - slope_params['min'])
        
        # 最适到最大坡度（缓慢下降）
        mask_high = (slope_data > slope_params['opt']) & (slope_data <= slope_params['max'])
        slope_suit[mask_high] = 1.0 - 0.3 * (slope_data[mask_high] - slope_params['opt']) / (slope_params['max'] - slope_params['opt'])
        
        # 超过最大坡度（显著下降）
        mask_over = slope_data > slope_params['max']
        slope_suit[mask_over] = 0.7 * np.exp(-(slope_data[mask_over] - slope_params['max']) / 10)
        
        # 输出坡度统计
        print("\n坡度分布统计:")
        print(f"- 平缓区域 (<{slope_params['min']}°): {np.sum(mask_low) / slope_data.size * 100:.1f}%")
        print(f"- 适宜区域 ({slope_params['min']}-{slope_params['opt']}°): {np.sum(mask_opt) / slope_data.size * 100:.1f}%")
        print(f"- 较陡区域 ({slope_params['opt']}-{slope_params['max']}°): {np.sum(mask_high) / slope_data.size * 100:.1f}%")
        print(f"- 过陡区域 (>{slope_params['max']}°): {np.sum(mask_over) / slope_data.size * 100:.1f}%")
        
        return slope_suit
    
    def vectorize_drainage_network(self, vector_path, shape, geotransform):
        """评估排水网络影响（面状水系）"""
        try:
            # 创建空白栅格
            raster = np.zeros(shape, dtype=np.float32)
            
            # 读取水系矢量
            vector_ds = ogr.Open(vector_path)
            if vector_ds is None:
                raise ValueError(f"无法打开水系文件: {vector_path}")
            
            layer = vector_ds.GetLayer()
            
            # 创建临时栅格
            mem_driver = gdal.GetDriverByName('MEM')
            target_ds = mem_driver.Create('', shape[1], shape[0], 1, gdal.GDT_Float32)
            target_ds.SetGeoTransform(geotransform)
            
            # 计算水系特征
            print("分析排水网络...")
            features_info = []
            total_area = 0
            
            for feature in layer:
                geom = feature.GetGeometryRef()
                if geom:
                    area = geom.GetArea()  # 面积（平方米）
                    perimeter = geom.Boundary().Length()  # 周长
                    compactness = 4 * np.pi * area / (perimeter * perimeter)  # 紧凑度
                    
                    features_info.append({
                        'area': area,
                        'perimeter': perimeter,
                        'compactness': compactness
                    })
                    total_area += area
            
            if features_info:
                # 计算统计信息
                avg_area = total_area / len(features_info)
                
                print(f"排水网络统计:")
                print(f"- 水体数量: {len(features_info)}")
                print(f"- 总面积: {total_area/1e6:.2f}平方公里")
                print(f"- 平均面积: {avg_area/1e4:.2f}公顷")
                
                # 重置图层读取位置
                layer.ResetReading()
                
                # 创建临时内存图层
                mem_driver = ogr.GetDriverByName('Memory')
                mem_ds = mem_driver.CreateDataSource('memory')
                mem_layer = mem_ds.CreateLayer('temp', layer.GetSpatialRef(), layer.GetGeomType())
                
                # 添加权重字段
                weight_field = ogr.FieldDefn('WEIGHT', ogr.OFTReal)
                mem_layer.CreateField(weight_field)
                
                # 计算最大面积（用于归一化）
                max_area = max(f['area'] for f in features_info)
                
                # 为每个水体计算综合权重
                for feature, info in zip(layer, features_info):
                    geom = feature.GetGeometryRef()
                    if geom:
                        out_feature = ogr.Feature(mem_layer.GetLayerDefn())
                        out_feature.SetGeometry(geom.Clone())
                        
                        # 计算综合权重（考虑面积和形状）
                        area_weight = np.log1p(info['area']) / np.log1p(max_area)
                        shape_weight = info['compactness']  # 0-1之间
                        weight = (area_weight * 0.7 + shape_weight * 0.3)
                        
                        out_feature.SetField('WEIGHT', weight)
                        mem_layer.CreateFeature(out_feature)
                
                # 栅格化带权重的图层
                gdal.RasterizeLayer(target_ds, [1], mem_layer, 
                                  options=['ATTRIBUTE=WEIGHT', 'ALL_TOUCHED=TRUE'])
                
                base_raster = target_ds.ReadAsArray()
                
                # 获取DEM数据（需要从类中传入）
                dem_data = self.current_dem  # 在assess_drainage_capability中设置
                
                # 计算水体区域的平均高程
                water_mask = base_raster > 0
                if np.any(water_mask):
                    water_elevation = np.mean(dem_data[water_mask])
                    
                    # 计算相对高差
                    elevation_diff = dem_data - water_elevation
                    
                    # 计算高差影响（考虑防洪和排水效率）
                    elevation_effect = np.where(
                        elevation_diff >= 0,  # 高于水体
                        np.where(
                            elevation_diff <= 2,  # 0-2米（高风险区）
                            0.2,  # 极低评分（防洪风险极高）
                            np.where(
                                elevation_diff <= 5,  # 2-5米（中风险区）
                                0.4,  # 较低评分（防洪风险较高）
                                np.where(
                                    elevation_diff <= 15,  # 5-15米（适宜区）
                                    0.7 + 0.3 * (elevation_diff - 5) / 10,  # 线性增加
                                    0.8 * np.exp(-(elevation_diff - 15) / 30)  # 距离过远效率降低
                                )
                            )
                        ),
                        0.1  # 低于水体（极不适宜）
                    )
                    
                    # 计算到水系的距离
                    distance = distance_transform_edt(base_raster == 0) * abs(geotransform[1])
                    distance_effect = np.exp(-distance / self.params['distance_threshold'])
                    
                    # 计算水系密度
                    density = gaussian_filter(base_raster > 0, 
                                           sigma=self.params['density_radius'] / abs(geotransform[1]))
                    density = density / np.max(density) if np.max(density) > 0 else density
                    
                    # 综合排水网络影响
                    influence = (
                        elevation_effect * 0.45 +     # 高差影响（最重要）
                        distance_effect * 0.35 +      # 距离影响（次重要）
                        density * 0.2                 # 密度影响（辅助）
                    )
                    
                    # 对结果进行修正
                    influence = np.where(
                        elevation_diff < 2,  # 高风险区域
                        influence * 0.5,     # 降低评分
                        influence
                    )
                    
                    # 输出统计信息
                    print(f"排水网络评价:")
                    print(f"- 水系覆盖率: {np.sum(base_raster > 0) / base_raster.size * 100:.1f}%")
                    print(f"- 有效影响区域: {np.sum(influence > 0.1) / influence.size * 100:.1f}%")
                    print(f"- 平均影响强度: {np.mean(influence[influence > 0]):.3f}")
                    
                    # 输出高程分布统计
                    elev_stats = {
                        '低于水体': np.sum(elevation_diff < 0) / elevation_diff.size * 100,
                        '0-2m': np.sum((elevation_diff >= 0) & (elevation_diff <= 2)) / elevation_diff.size * 100,
                        '2-5m': np.sum((elevation_diff > 2) & (elevation_diff <= 5)) / elevation_diff.size * 100,
                        '5-15m': np.sum((elevation_diff > 5) & (elevation_diff <= 15)) / elevation_diff.size * 100,
                        '>15m': np.sum(elevation_diff > 15) / elevation_diff.size * 100
                    }
                    print("\n高程分布统计:")
                    for key, value in elev_stats.items():
                        print(f"- {key}: {value:.1f}%")
                    
                    return influence
                    
                else:
                    print("警告：未找到有效水体区域")
                    return np.zeros(shape, dtype=np.float32)
            
        except Exception as e:
            print(f"排水网络评估失败: {str(e)}")
            return np.zeros(shape, dtype=np.float32)
    
    def assess_drainage_capability(self, **input_data):
        """综合评估排水能力"""
        try:
            # 读取数据
            dem_data, geotransform, projection = self.read_raster(input_data['dem'])
            self.current_dem = dem_data  # 保存DEM数据供其他方法使用
            rainfall_data, _, _ = self.read_raster(input_data['rainfall'])
            twi_data, _, _ = self.read_raster(input_data['twi'])
            slope_position_data, _, _ = self.read_raster(input_data['slope_position'])
            
            print("评估地形条件...")
            # 计算坡度
            dx, dy = np.gradient(dem_data)
            slope = np.degrees(np.arctan(np.sqrt(dx*dx + dy*dy)))
            slope_suit = self.assess_slope_suitability(slope)
            
            # 评价坡位适宜性
            print("评估坡位条件...")
            slope_position_suit = np.zeros_like(slope_position_data, dtype=np.float32)
            for pos, weight in self.params['slope_position_weights'].items():
                slope_position_suit[slope_position_data == pos] = weight
            
            # TWI适宜性评价
            print("评估地形湿度...")
            twi_suit = 1 - self.normalize_array(twi_data)
            
            # 评估排水网络影响
            print("评估排水网络...")
            drainage_influence = self.vectorize_drainage_network(
                input_data['drainage_network'],
                dem_data.shape,
                geotransform
            )
            
            # 降雨压力评价
            rainfall_stress = 1 - self.normalize_array(rainfall_data)
            
            # 使用参数化的权重
            weights = self.params['weights']
            capability = (
                slope_suit * weights['slope'] +
                slope_position_suit * weights['slope_position'] +
                twi_suit * weights['twi'] +
                drainage_influence * weights['drainage_net'] +
                rainfall_stress * weights['rainfall']
            )
            
            # 确保结果有效
            capability = np.clip(capability, 0, 1)
            
            # 分级
            thresholds = self.params['level_thresholds']
            levels = np.ones_like(capability)
            levels[capability > thresholds['poor']] = 2    # 较差
            levels[capability > thresholds['medium']] = 3  # 中等
            levels[capability > thresholds['good']] = 4    # 较好
            
            # 输出分级统计
            for i in range(1, 5):
                percent = np.sum(levels == i) / levels.size * 100
                print(f"等级 {i} 占比: {percent:.1f}%")
            
            # 保存结果
            timestamp = time.strftime("%Y%m%d%H")
            cap_path = os.path.join(self.workspace, f"drainage_capability_{timestamp}.tif")
            lvl_path = os.path.join(self.workspace, f"drainage_level_{timestamp}.tif")
            
            self.write_raster(capability, geotransform, projection, cap_path)
            self.write_raster(levels, geotransform, projection, lvl_path)
            
            print(f"评估完成！")
            print(f"- 结果已保存至: {cap_path}")
            print(f"- 分级结果已保存至: {lvl_path}")
            
            return capability, levels
            
        except Exception as e:
            print(f"排水能力评估失败: {str(e)}")
            raise

    def __del__(self):
        """清理资源"""
        try:
            gdal.SetConfigOption("GTIFF_SRS_SOURCE", None)
            gdal.SetConfigOption("GDAL_DATA", None)
        except:
            pass

    def read_raster(self, raster_path):
        """读取栅格数据"""
        try:
            ds = gdal.Open(raster_path)
            if ds is None:
                raise ValueError(f"无法打开栅格文件: {raster_path}")
            
            band = ds.GetRasterBand(1)
            data = band.ReadAsArray().astype(np.float32)
            geotransform = ds.GetGeoTransform()
            projection = ds.GetProjection()
            
            return data, geotransform, projection
        except Exception as e:
            raise Exception(f"读取栅格失败: {str(e)}")

    def write_raster(self, data, geotransform, projection, output_path):
        """保存栅格数据"""
        try:
            driver = gdal.GetDriverByName('GTiff')
            rows, cols = data.shape
            
            ds = driver.Create(
                output_path, 
                cols, 
                rows, 
                1, 
                gdal.GDT_Float32
            )
            
            ds.SetGeoTransform(geotransform)
            ds.SetProjection(projection)
            ds.GetRasterBand(1).WriteArray(data)
            ds.FlushCache()
            
            return output_path
        except Exception as e:
            raise Exception(f"保存栅格失败: {str(e)}")

    def normalize_array(self, array):
        """数组归一化"""
        try:
            valid_data = array[~np.isnan(array) & ~np.isinf(array)]
            if len(valid_data) == 0:
                raise ValueError("数组中没有有效值")
            
            min_val = np.percentile(valid_data, 1)
            max_val = np.percentile(valid_data, 99)
            
            normalized = np.clip(array, min_val, max_val)
            normalized = (normalized - min_val) / (max_val - min_val)
            
            return normalized
        except Exception as e:
            print(f"归一化失败: {str(e)}")
            raise

if __name__ == "__main__":
    try:
        workspace = r"C:\Users\Runker\Desktop\genarate_feature"
        
        # 准备输入数据
        input_data = {
            'dem': os.path.join(workspace, "raster_file", "dem.tif"),
            'drainage_network': os.path.join(workspace, "shp_file", "河流.shp"),
            'rainfall': os.path.join(workspace, "raster_file", "pre2022mean.tif"),
            'twi': os.path.join(workspace, "raster_file", "twi.tif"),
            'slope_position': os.path.join(workspace, "raster_file", "slope_position.tif")
        }
        
        # 检查文件是否存在
        for key, path in input_data.items():
            if not os.path.exists(path):
                raise ValueError(f"文件不存在: {path}")
                
        # 自定义参数
        params = {
            'distance_threshold': 1000,
            'density_radius': 800,
            'slope_threshold': {
                'min': 1,
                'opt': 8,
                'max': 25
            },
            'slope_position_weights': {
                1: 0.3,    # 平地（排水差）
                2: 0.2,    # 坡底（最差，易积水）
                3: 0.5,    # 坡中平地
                4: 0.7,    # 坡中上部
                5: 0.9,    # 坡上部
                6: 1.0     # 坡顶部（最好）
            },
            'weights': {
                'slope': 0.3,
                'slope_position': 0.4,
                'drainage_net': 0.15,
                'twi': 0.1,
                'rainfall': 0.05
            },
            'level_thresholds': {
                'poor': 0.30,
                'medium': 0.40,
                'good': 0.55
            }
        }
        
        print("开始排水能力评估...")
        assessment = DrainageCapabilityAssessment(workspace, params)
        capability, levels = assessment.assess_drainage_capability(**input_data)
        print("评估完成！")
        
    except Exception as e:
        print(f"评估过程出错: {str(e)}")


