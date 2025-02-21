import os
import numpy as np
import pandas as pd
from osgeo import gdal, ogr
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from rasterio.windows import Window
from rasterio.transform import Affine
import warnings
import rasterio
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from multiprocessing import freeze_support

warnings.filterwarnings('ignore')

class SoilTexturePredictor:
    """土壤质地预测类"""
    
    def __init__(self, workspace):
        self.workspace = workspace
        gdal.AllRegister()
        
        # 土壤质地分类编码映射
        self.texture_codes = {
            '壤土': 1,
            '粉(砂)质壤土': 2,
            '砂质壤土': 3,
            '粉(砂)质黏壤土': 4,
            '黏壤土': 5,
            '砂质黏壤土': 6,
            '壤质黏土': 7,
            '黏土': 8,
            '砂质黏土': 9,
            '重黏土': 10,
            '粉(砂)质黏土': 11,
            '砂土及壤质砂土': 12
        }
        
        # 初始化模型
        self.model = RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features='sqrt',
            random_state=42
        )
        
        self.scaler = StandardScaler()
    
    def read_sample_points(self, shp_path):
        """读取采样点数据"""
        try:
            ds = ogr.Open(shp_path)
            if ds is None:
                raise ValueError(f"无法打开矢量文件: {shp_path}")
            
            layer = ds.GetLayer()
            points_data = []
            
            for feature in layer:
                geom = feature.GetGeometryRef()
                if geom:
                    x = geom.GetX()
                    y = geom.GetY()
                    texture_code = feature.GetField("TRZD_CODE")
                    points_data.append([x, y, texture_code])
            
            return np.array(points_data)
            
        except Exception as e:
            raise Exception(f"读取采样点失败: {str(e)}")
    
    def extract_features_at_points(self, points, **raster_data):
        """在采样点位置提取特征值"""
        try:
            features = {}
            labels = points[:, 2]  # 质地编码
            
            for name, raster_path in raster_data.items():
                ds = gdal.Open(raster_path)
                if ds is None:
                    raise ValueError(f"无法打开栅格文件: {raster_path}")
                
                geotransform = ds.GetGeoTransform()
                data = ds.GetRasterBand(1).ReadAsArray()
                
                # 计算栅格索引
                px = ((points[:, 0] - geotransform[0]) / geotransform[1]).astype(int)
                py = ((points[:, 1] - geotransform[3]) / geotransform[5]).astype(int)
                
                # 提取值
                values = data[py, px]
                features[name] = values
            
            return features, labels
            
        except Exception as e:
            raise Exception(f"特征提取失败: {str(e)}")
    
    def train_model(self, features, labels):
        """训练模型"""
        try:
            # 准备训练数据
            X = np.column_stack([feat for feat in features.values()])
            y = labels
            
            # 数据标准化
            X = self.scaler.fit_transform(X)
            
            # 训练模型
            self.model.fit(X, y)
            
            # 输出特征重要性
            importance = dict(zip(features.keys(), self.model.feature_importances_))
            print("\n特征重要性:")
            for feat, imp in sorted(importance.items(), key=lambda x: x[1], reverse=True):
                print(f"- {feat}: {imp:.3f}")
            
            # 输出模型评分
            score = self.model.score(X, y)
            print(f"\n模型训练得分: {score:.3f}")
            
        except Exception as e:
            raise Exception(f"模型训练失败: {str(e)}")
    
    def predict_texture(self, **raster_data):
        """预测土壤质地分布"""
        try:
            # 读取参考栅格获取空间信息
            ref_ds = gdal.Open(list(raster_data.values())[0])
            if ref_ds is None:
                raise ValueError("无法打开参考栅格")
            
            # 获取栅格信息
            gdal_transform = ref_ds.GetGeoTransform()
            projection = ref_ds.GetProjection()
            shape = (ref_ds.RasterYSize, ref_ds.RasterXSize)
            
            # 创建 Affine transform
            transform = Affine(
                gdal_transform[1], gdal_transform[2], gdal_transform[0],
                gdal_transform[4], gdal_transform[5], gdal_transform[3]
            )
            
            # 创建rasterio profile
            profile = {
                'driver': 'GTiff',
                'height': shape[0],
                'width': shape[1],
                'count': 1,
                'dtype': 'float32',
                'crs': projection,
                'transform': transform,
                'nodata': np.nan,
                'compress': 'lzw'
            }
            
            # 设置分块处理参数
            chunk_size = 1000
            max_workers = min(os.cpu_count(), 8)
            
            # 获取特征名称列表
            feature_names = list(raster_data.keys())
            
            # 创建数据块
            chunks = [
                Window(col, row, 
                      min(chunk_size, shape[1] - col), 
                      min(chunk_size, shape[0] - row))
                for row in range(0, shape[0], chunk_size)
                for col in range(0, shape[1], chunk_size)
            ]
            
            # 准备输出文件
            pred_path = os.path.join(self.workspace, "soil_texture_prediction.tif")
            
            # 确保输出目录存在
            os.makedirs(self.workspace, exist_ok=True)
            
            with rasterio.open(pred_path, 'w', **profile) as dst:
                with ProcessPoolExecutor(max_workers=max_workers) as executor:
                    futures = [
                        executor.submit(
                            self.process_chunk,
                            chunk,
                            raster_data,
                            self.model,
                            self.scaler,
                            feature_names  # 传递特征名称列表
                        )
                        for chunk in chunks
                    ]
                    
                    for future in tqdm(as_completed(futures), 
                                     total=len(futures),
                                     desc="预测土壤质地",
                                     ncols=100):
                        try:
                            window, pred_chunk = future.result()
                            dst.write(pred_chunk, 1, window=window)
                        except Exception as e:
                            print(f"处理数据块失败: {str(e)}")
                            continue
            
            print(f"\n预测结果已保存: {pred_path}")
            return pred_path
            
        except Exception as e:
            raise Exception(f"预测失败: {str(e)}")

    @staticmethod
    def process_chunk(window, raster_data, model, scaler, feature_names):
        """处理单个数据块"""
        try:
            # 读取数据块
            chunk_data = {}
            for name, path in raster_data.items():
                with rasterio.open(path) as src:
                    chunk_data[name] = src.read(1, window=window)
            
            # 准备特征数据
            features = []
            for name in feature_names:  # 使用传入的特征名称列表
                if name in chunk_data:
                    features.append(chunk_data[name].ravel())
            
            X = np.column_stack(features)
            valid_mask = ~np.isnan(X).any(axis=1)
            X = X[valid_mask]
            
            if X.shape[0] > 0:
                # 标准化特征
                X = scaler.transform(X)
                
                # 预测
                pred = model.predict(X)
                
                # 重塑结果
                pred_chunk = np.full(chunk_data[list(chunk_data.keys())[0]].shape, np.nan)
                pred_chunk.ravel()[valid_mask] = pred
                
                return window, pred_chunk
            else:
                return window, np.full(chunk_data[list(chunk_data.keys())[0]].shape, np.nan)
            
        except Exception as e:
            raise Exception(f"处理数据块失败: {str(e)}")

    def read_raster(self, raster_path):
        """读取栅格数据"""
        try:
            ds = gdal.Open(raster_path)
            if ds is None:
                raise ValueError(f"无法打开栅格文件: {raster_path}")
            
            data = ds.GetRasterBand(1).ReadAsArray()
            return data.astype(np.float32)
        except Exception as e:
            raise Exception(f"读取栅格失败: {str(e)}")

    def write_raster(self, data, geotransform, projection, output_path):
        """写入栅格数据"""
        try:
            driver = gdal.GetDriverByName("GTiff")
            ds = driver.Create(output_path, data.shape[1], data.shape[0], 1, gdal.GDT_Float32)
            ds.SetGeoTransform(geotransform)
            ds.SetProjection(projection)
            ds.GetRasterBand(1).WriteArray(data)
            ds.FlushCache()
            ds = None
        except Exception as e:
            raise Exception(f"写入栅格失败: {str(e)}")

if __name__ == '__main__':
    # 添加这行
    freeze_support()
    
    # 初始化预测器
    workspace = r"C:\Users\Runker\Desktop\genarate_feature"
    predictor = SoilTexturePredictor(workspace)
    
    # 准备输入数据
    input_data = {
        'dem': os.path.join(workspace, "raster_file", "dem.tif"),
        'twi': os.path.join(workspace, "raster_file", "twi.tif"),
        'rainfall': os.path.join(workspace, "raster_file", "pre2022mean.tif"),
        'slope_position': os.path.join(workspace, "raster_file", "slope_position.tif")
    }
    
    try:
        # 读取采样点
        sample_points = predictor.read_sample_points(
            os.path.join(workspace, "shp_file", "土壤质地.shp")
        )
        
        # 提取训练特征
        features, labels = predictor.extract_features_at_points(sample_points, **input_data)
        
        # 训练模型
        predictor.train_model(features, labels)
        
        # 预测
        prediction = predictor.predict_texture(**input_data)
        
        print("处理完成！")
        
    except Exception as e:
        print(f"处理过程出错: {str(e)}") 