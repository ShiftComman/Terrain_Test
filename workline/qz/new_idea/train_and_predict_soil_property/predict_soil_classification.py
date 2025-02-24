import logging
import pickle
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.windows import Window
from rasterio.features import geometry_mask
from shapely.geometry import mapping
from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from concurrent.futures import ProcessPoolExecutor, as_completed
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
import psutil
import os
from contextlib import nullcontext
matplotlib.use('Agg')  # 使用非交互式后端

class SoilClassificationPredictor:
    def __init__(self, log_file, model_dir, feature_dir, output_dir,
                 shapefile_path=None, output_uncertainty=False, output_visualization=False):
        # 验证输入参数
        if not all(isinstance(path, (str, Path)) for path in [log_file, model_dir, feature_dir, 
                                                             output_dir]):
            raise ValueError("文件路径参数必须是字符串或Path对象")
            
        # 确保输出目录存在
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        self.logger = self._setup_logger(log_file)
        self.model_dir = Path(model_dir)
        self.feature_dir = Path(feature_dir)
        self.output_dir = Path(output_dir)
        self.shapefile_path = shapefile_path
        self.output_uncertainty = output_uncertainty
        self.output_visualization = output_visualization

    def _setup_logger(self, log_file):
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger = logging.getLogger(__name__)
        if not logger.handlers:  # 避免重复添加处理器
            handler = logging.FileHandler(log_file, encoding='utf-8')
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger

    def load_model(self, model_path):
        self.logger.info(f"正在加载模型: {model_path}")
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        if isinstance(model_data, dict) and 'model' in model_data and 'feature_names' in model_data:
            model = model_data['model']
            feature_names = [str(f) for f in model_data['feature_names']]
            n_features = model.n_features_in_
            self.logger.info(f"加载的模型期望 {n_features} 个特征")
            return model, feature_names[:n_features]
        else:
            raise ValueError("模型文件格式不正确，缺少模型或特征名称信息")

    @staticmethod
    def get_raster_info(raster_path):
        with rasterio.open(raster_path) as src:
            return src.profile, src.shape, src.transform

    @staticmethod
    def create_mask_from_shapefile(shapefile_path, raster_path):
        gdf = gpd.read_file(shapefile_path)
        with rasterio.open(raster_path) as src:
            geometries = [mapping(geom) for geom in gdf.geometry]
            mask = geometry_mask(geometries, out_shape=src.shape, transform=src.transform, invert=True)
        return mask

    @staticmethod
    def predict_chunk(model, feature_data, feature_names, need_uncertainty=False, max_chunk_size=100000):
        """
        根据需求选择是否计算不确定性的预测方法
        """
        if not isinstance(model, RandomForestClassifier):
            raise ValueError("此方法仅适用于随机森林分类器")
        
        feature_data = feature_data.astype(np.float32)
        predictions = np.zeros(feature_data.shape[0], dtype=np.int32)
        uncertainties = np.zeros(feature_data.shape[0], dtype=np.float32) if need_uncertainty else None
        
        for i in range(0, feature_data.shape[0], max_chunk_size):
            chunk = feature_data[i:i+max_chunk_size]
            chunk_df = pd.DataFrame(chunk, columns=feature_names)
            
            if need_uncertainty:
                # 如果需要不确定性，使用predict_proba
                probas = model.predict_proba(chunk_df)
                predictions[i:i+max_chunk_size] = np.argmax(probas, axis=1)
                uncertainties[i:i+max_chunk_size] = 1 - np.max(probas, axis=1)
            else:
                # 如果不需要不确定性，直接使用predict
                predictions[i:i+max_chunk_size] = model.predict(chunk_df)
        
        return (predictions, uncertainties) if need_uncertainty else predictions

    @staticmethod
    def process_raster_chunk(args):
        model, feature_files, window, feature_names, mask, need_uncertainty = args
        chunk_data = {}
        for file in feature_files:
            with rasterio.open(file) as src:
                chunk_data[Path(file).stem] = src.read(1, window=window)
        
        rows, cols = next(iter(chunk_data.values())).shape
        feature_array = np.stack([chunk_data[f] for f in feature_names], axis=-1)
        
        if mask is not None:
            chunk_mask = mask[window.row_off:window.row_off+window.height, 
                            window.col_off:window.col_off+window.width]
            feature_array[~chunk_mask] = np.nan
        
        valid_pixels = ~np.isnan(feature_array).any(axis=-1)
        feature_array = feature_array[valid_pixels]
        
        result = np.full((rows, cols), np.nan)
        uncertainty = np.full((rows, cols), np.nan) if need_uncertainty else None
        
        if np.sum(valid_pixels) > 0:
            if need_uncertainty:
                predictions, uncertainties = SoilClassificationPredictor.predict_chunk(
                    model, feature_array, feature_names, need_uncertainty=True)
                result[valid_pixels] = predictions
                uncertainty[valid_pixels] = uncertainties
            else:
                predictions = SoilClassificationPredictor.predict_chunk(
                    model, feature_array, feature_names, need_uncertainty=False)
                result[valid_pixels] = predictions
        
        return window, result, uncertainty

    def predict_soil_class(self, model, feature_names, class_name, chunk_size=1000):
        try:
            # 添加内存监控
            available_memory = psutil.virtual_memory().available
            estimated_memory_per_chunk = chunk_size * chunk_size * len(feature_names) * 4
            if estimated_memory_per_chunk > available_memory * 0.5:
                new_chunk_size = int(np.sqrt((available_memory * 0.5) / (len(feature_names) * 4)))
                self.logger.warning(f"由于内存限制，调整chunk_size从{chunk_size}到{new_chunk_size}")
                chunk_size = new_chunk_size

            self.logger.info(f"开始预测土壤分类: {class_name}")
            feature_files = list(self.feature_dir.glob('*.tif'))
            profile, (height, width), transform = self.get_raster_info(feature_files[0])

            mask = self.create_mask_from_shapefile(self.shapefile_path, feature_files[0]) if self.shapefile_path else None
            
            chunks = [
                (model, feature_files, Window(col, row, min(chunk_size, width - col), 
                                           min(chunk_size, height - row)),
                 feature_names, mask, self.output_uncertainty)
                for row in range(0, height, chunk_size) 
                for col in range(0, width, chunk_size)
            ]

            output_path = self.output_dir / f"{class_name}_prediction.tif"
            uncertainty_path = self.output_dir / f"{class_name}_uncertainty.tif" if self.output_uncertainty else None
            
            with rasterio.open(output_path, 'w', **profile) as dst, \
                 (rasterio.open(uncertainty_path, 'w', **profile) if uncertainty_path else nullcontext()) as uncertainty_dst:
                
                with ProcessPoolExecutor(max_workers=min(8, os.cpu_count())) as executor:
                    futures = [executor.submit(self.process_raster_chunk, chunk) for chunk in chunks]
                    
                    for future in tqdm(as_completed(futures), total=len(futures), desc="Processing chunks"):
                        try:
                            window, result, uncertainty = future.result(timeout=300)
                            dst.write(result.astype(profile['dtype']), 1, window=window)
                            if uncertainty_dst:
                                uncertainty_dst.write(uncertainty.astype(np.float32), 1, window=window)
                        except Exception as e:
                            self.logger.error(f"处理数据块时发生错误: {str(e)}")
                            continue

            self.logger.info(f"预测结果已保存到: {output_path}")
            if uncertainty_path:
                self.logger.info(f"不确定性结果已保存到: {uncertainty_path}")

        except Exception as e:
            self.logger.error(f"预测土壤分类过程中发生错误: {str(e)}")
            raise

    def predict_soil_classes(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)

        model_files = list(self.model_dir.glob('*_model.pkl'))
        for model_file in tqdm(model_files, desc="预测土壤分类"):
            try:
                class_name = model_file.stem.replace('_model', '')
                
                # 如果该分类已经预测过，则跳过
                if (self.output_dir / f"{class_name}_prediction.tif").exists():
                    self.logger.info(f"已预测过 {class_name}，跳过")
                    continue

                self.logger.info(f"正在预测 {class_name}")
                model, feature_names = self.load_model(model_file)
                self.predict_soil_class(model, feature_names, class_name)
                
            except KeyboardInterrupt:
                self.logger.warning("用户中断处理")
                break
            except Exception as e:
                self.logger.error(f"处理 {model_file.stem} 时发生错误: {str(e)}")
                continue

    def run(self):
        self.logger.info("开始预测土壤分类")
        try:
            self.predict_soil_classes()
            self.logger.info("土壤分类预测完成")
        except Exception as e:
            self.logger.error(f"预测土壤分类过程中发生错误: {str(e)}")
            raise

if __name__ == "__main__":
    # 配置参数
    config = {
        "log_file": r"G:\soil_property_result\qzs\logs\predict_soil_classification.log",
        "model_dir": r"G:\soil_property_result\qzs\models\soil_property_class\models",
        "feature_dir": r"G:\tif_features\county_feature\qz",
        "output_dir": r"G:\soil_property_result\qzs\soil_property_class_predict",
        "shapefile_path": r"F:\cache_data\shp_file\qz\qz_extent_p_500.shp",
        "output_uncertainty": False,  # 是否输出不确定性结果
        "output_visualization": False  # 是否输出可视化结果
    }
    
    try:
        predictor = SoilClassificationPredictor(**config)
        predictor.run()
    except Exception as e:
        print(f"程序执行出错: {str(e)}") 