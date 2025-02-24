# encoding: utf-8
import os
import time
import numpy as np
from osgeo import gdal, ogr
from scipy.ndimage import gaussian_filter, distance_transform_edt
import warnings
warnings.filterwarnings('ignore')

# 设置环境编码
gdal.AllRegister()

# 设置工作空间
gdal.SetConfigOption("GDAL_DATA", os.path.dirname(os.path.abspath(__file__)))

"""
灌溉能力评价模块
用于评估区域灌溉适宜性，考虑地形、土壤、水资源等因素
"""

class IrrigationCapabilityAssessment:
    """
    灌溉能力评价类
    主要功能：
    1. 评估地形条件对灌溉的适宜性
    2. 评估土壤条件对灌溉的适应性
    3. 评估水资源可利用程度
    4. 综合评价区域灌溉能力
    """
    
    def __init__(self, workspace, params=None):
        """初始化工作环境和参数"""
        self.workspace = workspace
        gdal.AllRegister()
        
        # 默认参数
        default_params = {
            # 水源影响参数
            'distance_threshold': 2000,     # 水源影响距离阈值（米）
            'density_radius': 1500,         # 水源密度计算半径（米）
            # 高程影响参数
            'height_good': 20,              # 最佳抽水高度（米）
            'height_medium': 50,            # 可抽水最大高度（米）
            'height_decay_rate': 200,       # 高程影响衰减率（米）
            # 坡度参数
            'slope_threshold': 25,          # 坡度阈值（度）
            # 坡位权重（1-6）
            'slope_position_weights': {
                1: 0.9,    # 平地
                2: 0.9,    # 坡底
                3: 0.8,    # 坡中平地
                4: 0.5,    # 坡中上部
                5: 0.4,    # 坡上部
                6: 0.2     # 坡顶部
            },
            # 权重参数
            'weights': {
                'water_influence': 0.40,    # 水源可达性权重
                'elevation': 0.15,          # 高程影响权重
                'slope': 0.15,              # 坡度适宜性权重
                'slope_position': 0.12,     # 坡位权重
                'twi': 0.10,                # 地形湿度权重
                'rainfall': 0.08            # 降雨补给权重
            },
            # 分级阈值
            'level_thresholds': {
                'poor': 0.30,              # 较差等级阈值
                'medium': 0.45,            # 中等等级阈值
                'good': 0.55               # 较好等级阈值
            }
        }
        
        # 更新参数（如果用户提供）
        self.params = default_params
        if params:
            self.params.update(params)
            
        # 验证权重和是否为1
        weights = self.params['weights']
        weight_sum = sum(weights.values())
        if abs(weight_sum - 1.0) > 0.001:
            print(f"警告：权重总和 ({weight_sum:.3f}) 不等于1，已自动归一化")
            for key in weights:
                weights[key] /= weight_sum
        
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
        """数组归一化，处理异常值"""
        try:
            # 移除无效值
            valid_data = array[~np.isnan(array) & ~np.isinf(array)]
            if len(valid_data) == 0:
                raise ValueError("数组中没有有效值")
            
            # 使用百分位数来避免异常值影响
            min_val = np.percentile(valid_data, 1)  # 使用1%分位数
            max_val = np.percentile(valid_data, 99)  # 使用99%分位数
            
            # 归一化
            normalized = np.clip(array, min_val, max_val)
            normalized = (normalized - min_val) / (max_val - min_val)
            
            return normalized
            
        except Exception as e:
            print(f"归一化失败: {str(e)}")
            raise
        
    def vectorize_water_sources(self, vector_path, shape, geotransform):
        """评估水源影响"""
        try:
            # 创建空白栅格
            raster = np.zeros(shape, dtype=np.float32)
            
            # 读取水源矢量
            vector_ds = ogr.Open(vector_path)
            if vector_ds is None:
                raise ValueError(f"无法打开水源文件: {vector_path}")
            
            layer = vector_ds.GetLayer()
            
            # 创建临时栅格
            mem_driver = gdal.GetDriverByName('MEM')
            target_ds = mem_driver.Create('', shape[1], shape[0], 1, gdal.GDT_Float32)
            target_ds.SetGeoTransform(geotransform)
            
            # 计算水源特征
            print("分析水源特征...")
            features_info = []
            total_area = 0
            
            for feature in layer:
                geom = feature.GetGeometryRef()
                if geom:
                    area = geom.GetArea()  # 面积（平方米）
                    perimeter = geom.Boundary().Length()  # 周长
                    compactness = 4 * np.pi * area / (perimeter * perimeter)  # 紧凑度
                    
                    # 计算外接矩形
                    env = geom.GetEnvelope()
                    width = env[1] - env[0]
                    height = env[3] - env[2]
                    extent = width * height
                    
                    features_info.append({
                        'area': area,
                        'perimeter': perimeter,
                        'compactness': compactness,
                        'extent': extent
                    })
                    total_area += area
            
            if features_info:
                # 计算统计信息
                avg_area = total_area / len(features_info)
                max_area = max(f['area'] for f in features_info)
                
                print(f"水源统计:")
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
                
                # 为每个水体计算综合权重
                for feature, info in zip(layer, features_info):
                    geom = feature.GetGeometryRef()
                    if geom:
                        out_feature = ogr.Feature(mem_layer.GetLayerDefn())
                        out_feature.SetGeometry(geom.Clone())
                        
                        # 计算综合权重
                        area_weight = np.log1p(info['area']) / np.log1p(max_area)
                        compact_weight = info['compactness']  # 0-1之间
                        
                        # 综合权重（考虑面积和形状）
                        weight = (area_weight * 0.7 + compact_weight * 0.3)
                        out_feature.SetField('WEIGHT', weight)
                        mem_layer.CreateFeature(out_feature)
                
                # 栅格化带权重的图层
                gdal.RasterizeLayer(target_ds, [1], mem_layer, 
                                  options=['ATTRIBUTE=WEIGHT', 'ALL_TOUCHED=TRUE'])
                
                base_raster = target_ds.ReadAsArray()
                
                # 使用参数化的距离阈值
                distance = distance_transform_edt(base_raster == 0) * abs(geotransform[1])
                distance_effect = np.exp(-distance / self.params['distance_threshold'])
                
                # 使用参数化的密度半径
                sigma = self.params['density_radius'] / abs(geotransform[1])  # 转换为像素数
                density = gaussian_filter(base_raster > 0, sigma=sigma)
                density = density / np.max(density) if np.max(density) > 0 else density
                
                # 综合水源影响（考虑权重、距离和密度）
                influence = (
                    distance_effect * 0.45 +  # 距离影响
                    density * 0.35 +          # 密度影响（增加权重）
                    (base_raster / np.max(base_raster) if np.max(base_raster) > 0 else 0) * 0.20  # 权重影响
                )
                
                # 输出统计信息
                print(f"水源影响评价:")
                print(f"- 水源覆盖率: {np.sum(base_raster > 0) / base_raster.size * 100:.1f}%")
                print(f"- 有效影响区域: {np.sum(influence > 0.1) / influence.size * 100:.1f}%")
                print(f"- 平均影响强度: {np.mean(influence[influence > 0]):.3f}")
                
                return influence
                
            else:
                print("警告：未找到有效水源要素")
                return np.zeros(shape, dtype=np.float32)
            
        except Exception as e:
            print(f"水源影响评估失败: {str(e)}")
            return np.zeros(shape, dtype=np.float32)

    def assess_irrigation_capability(self, **input_data):
        """综合评估灌溉能力"""
        try:
            # 读取必需的数据
            dem_data, geotransform, projection = self.read_raster(input_data['dem'])
            rainfall_data, _, _ = self.read_raster(input_data['rainfall'])
            twi_data, _, _ = self.read_raster(input_data['twi'])
            slope_position_data, _, _ = self.read_raster(input_data['slope_position'])
            
            print("评估地形条件...")
            # 计算坡度
            dx, dy = np.gradient(dem_data)
            slope = np.degrees(np.arctan(np.sqrt(dx*dx + dy*dy)))
            slope_suit = np.where(
                slope <= self.params['slope_threshold'], 
                1 - (slope / self.params['slope_threshold']), 
                0
            )
            
            # 评价坡位适宜性
            print("评估坡位条件...")
            slope_position_suit = np.zeros_like(slope_position_data, dtype=np.float32)
            for pos, weight in self.params['slope_position_weights'].items():
                slope_position_suit[slope_position_data == pos] = weight
            
            # TWI适宜性评价（TWI越大表示汇水能力越强）
            print("评估地形湿度...")
            twi_suit = self.normalize_array(twi_data)
            
            print("评估水源条件...")
            water_influence = self.vectorize_water_sources(
                input_data['water_sources'],
                dem_data.shape,
                geotransform
            )
            
            # 计算与水源的相对高差影响
            print("计算高程影响...")
            water_mask = water_influence > 0
            if np.any(water_mask):
                # 获取水源区域的高程统计
                water_elevations = dem_data[water_mask]
                water_min_elev = np.percentile(water_elevations, 10)  # 使用10%分位数避免异常值
                
                # 计算相对高差
                elevation_diff = dem_data - water_min_elev
                
                # 高差影响评价（使用参数化的高程阈值）
                elevation_effect = np.where(
                    elevation_diff > 0,  # 高于水源
                    np.where(
                        elevation_diff <= self.params['height_good'],
                        0.9,  # 较好
                        np.where(
                            elevation_diff <= self.params['height_medium'],
                            0.7 * np.exp(-(elevation_diff - self.params['height_good']) / 
                                       self.params['height_medium']),
                            0.3 * np.exp(-(elevation_diff - self.params['height_medium']) / 
                                       self.params['height_decay_rate'])
                        )
                    ),
                    1.0  # 低于水源，最适宜
                )
            else:
                elevation_effect = np.ones_like(dem_data)
            
            # 评价降雨补给
            rainfall_suit = self.normalize_array(rainfall_data)
            
            # 使用参数化的权重
            weights = self.params['weights']
            capability = (
                water_influence * weights['water_influence'] +
                elevation_effect * weights['elevation'] +
                slope_suit * weights['slope'] +
                slope_position_suit * weights['slope_position'] +
                twi_suit * weights['twi'] +
                rainfall_suit * weights['rainfall']
            )
            
            # 确保结果有效
            capability = np.clip(capability, 0, 1)
            
            # 使用参数化的分级阈值
            thresholds = self.params['level_thresholds']
            levels = np.ones_like(capability)
            levels[capability > thresholds['poor']] = 2    # 较差
            levels[capability > thresholds['medium']] = 3  # 中等
            levels[capability > thresholds['good']] = 4    # 较好
            
            # 输出分级统计
            for i in range(1, 5):
                percent = np.sum(levels == i) / levels.size * 100
                print(f"等级 {i} 占比: {percent:.1f}%")
            
            # 输出高程统计
            if np.any(water_mask):
                elev_stats = {
                    '≤20m': np.sum((elevation_diff > 0) & (elevation_diff <= 20)) / elevation_diff.size * 100,
                    '20-50m': np.sum((elevation_diff > 20) & (elevation_diff <= 50)) / elevation_diff.size * 100,
                    '>50m': np.sum(elevation_diff > 50) / elevation_diff.size * 100,
                    '低于水源': np.sum(elevation_diff <= 0) / elevation_diff.size * 100
                }
                print("\n高程分布统计:")
                for key, value in elev_stats.items():
                    print(f"- {key}: {value:.1f}%")
            
            # 保存结果
            timestamp = time.strftime("%Y%m%d%H")
            cap_path = os.path.join(self.workspace, f"irrigation_capability_{timestamp}.tif")
            lvl_path = os.path.join(self.workspace, f"irrigation_level_{timestamp}.tif")
            
            self.write_raster(capability, geotransform, projection, cap_path)
            self.write_raster(levels, geotransform, projection, lvl_path)
            
            print(f"评估完成！")
            print(f"- 结果已保存至: {cap_path}")
            print(f"- 分级结果已保存至: {lvl_path}")
            
            return capability, levels
            
        except Exception as e:
            print(f"灌溉能力评估失败: {str(e)}")
            raise

# 筛选指定字段的指定值的shp并保存
def filter_shp_by_field_value(shp_path, field_name, value_list, output_path):
    """
    筛选指定字段的指定值的shp并保存，支持中文属性表
    """
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        gdal.SetConfigOption('SHAPE_ENCODING', 'UTF-8')
        
        # 读取输入shp
        ds = ogr.Open(shp_path)
        layer = ds.GetLayer()
        
        # 创建输出shp
        driver = ogr.GetDriverByName('ESRI Shapefile')
        if os.path.exists(output_path):
            driver.DeleteDataSource(output_path)
        out_ds = driver.CreateDataSource(output_path)
        out_layer = out_ds.CreateLayer(
            os.path.splitext(os.path.basename(output_path))[0],
            layer.GetSpatialRef(),
            layer.GetGeomType()
        )
        
        # 复制字段结构并调整字段宽度
        layer_defn = layer.GetLayerDefn()
        for i in range(layer_defn.GetFieldCount()):
            field_defn = layer_defn.GetFieldDefn(i)
            if field_defn.GetType() == ogr.OFTString:
                field_defn.SetWidth(254)  # 加大字符字段长度以支持中文
            elif field_defn.GetType() == ogr.OFTReal:
                field_defn.SetWidth(32)    # 加大数值字段宽度
                field_defn.SetPrecision(8) # 设置小数位数
            out_layer.CreateField(field_defn)
        
        # 筛选并复制要素
        for feature in layer:
            if feature.GetField(field_name) in value_list:
                out_feature = ogr.Feature(out_layer.GetLayerDefn())
                out_feature.SetGeometry(feature.GetGeometryRef().Clone())
                for i in range(layer_defn.GetFieldCount()):
                    out_feature.SetField(i, feature.GetField(i))
                out_layer.CreateFeature(out_feature)
                
        out_ds = None
        ds = None
        print(f"已成功创建筛选后的Shapefile: {output_path}")
        
    except Exception as e:
        print(f"筛选shp失败: {str(e)}")
        raise
    finally:
        gdal.SetConfigOption('SHAPE_ENCODING', '')
    
if __name__ == "__main__":
    try:
        # 设置工作空间
        workspace = r"G:\soil_property_result\qzs\irrigation_drainage_generate"
        # 设置栅格路径
        raster_path = r"G:\tif_features\county_feature\qz"
        # 变量名称
        dem_name = "dem"
        rainfall_name = "pre22_mean"
        twi_name = "topographicwetnessindex"
        slope_position_name = "slopepostion"
        # 水源路径
        water_sources_path = r"G:\soil_property_result\qzs\shp\water_sources.shp"
         # 土地利用类型shp路径
        land_use_path = r"F:\cache_data\shp_file\qz\qz_sd_polygon.shp"
        # 河流、湖泊、水库、沟渠代码
        field_name = 'DLBM'
        # use_land_code_list = ["1101", "1102", "1103", "1107"]
        use_land_code_list = ["11"]
        # 筛选指定字段的指定值的shp并保存
        filter_shp_by_field_value(land_use_path, field_name, use_land_code_list, water_sources_path)
        # 变量名称
        dem_name = "dem"
        rainfall_name = "pre22_mean"
        twi_name = "topographicwetnessindex"
        slope_position_name = "slopepostion"

    
        os.makedirs(workspace, exist_ok=True)
        print(f"工作空间: {workspace}")
        
        # 自定义参数示例
        custom_params = {
            'distance_threshold': 1500,     # 修改水源影响距离为1.5km
            'height_good': 15,              # 修改最佳抽水高度为15m
            'height_medium': 40,            # 修改可抽水最大高度为40m
            'weights': {
                'water_influence': 0.40,    # 水源可达性
                'elevation': 0.1,          # 高程影响
                'slope': 0.1,              # 坡度适宜性
                'slope_position': 0.2,     # 坡位影响
                'twi': 0.1,                # 地形湿度
                'rainfall': 0.1            # 降雨补给
            }
        }
        
        # 准备输入数据
        
        input_data = {
            'dem': os.path.join(raster_path, f"{dem_name}.tif"),
            'water_sources': water_sources_path,
            'rainfall': os.path.join(raster_path, f"{rainfall_name}.tif"),
            'twi': os.path.join(raster_path, f"{twi_name}.tif"),
            'slope_position': os.path.join(raster_path, f"{slope_position_name}.tif")
        }
        # 检查文件是否存在
        for key, path in input_data.items():
            if not os.path.exists(path):
                raise ValueError(f"文件不存在: {path}")
        # 使用自定义参数创建评估对象
        assessment = IrrigationCapabilityAssessment(workspace, custom_params)
        capability, levels = assessment.assess_irrigation_capability(**input_data)
        
        print("评价完成！结果已保存。")
        
    except Exception as e:
        print(f"评价过程出错：{str(e)}")
    finally:
        try:
            gdal.SetConfigOption("GDAL_DATA", os.path.dirname(os.path.abspath(__file__)))
        except:
            pass
    