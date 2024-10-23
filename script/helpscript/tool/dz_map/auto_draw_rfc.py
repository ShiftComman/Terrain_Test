# 导入必要的库
import os
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.features import rasterize, geometry_mask
from rasterio.mask import mask as rasterio_mask
from rasterio.transform import from_bounds
from rasterio.enums import Resampling
from shapely.geometry import mapping, shape, box
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb
from scipy.ndimage import zoom
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import gen_batches
import multiprocessing
from tqdm import tqdm
import logging
from concurrent.futures import ProcessPoolExecutor
import pandas as pd
import psutil  # 用于监控系统资源
import gc     # 用于垃圾回收
import time   # 用于计时
from functools import partial
import warnings
warnings.filterwarnings('ignore')  # 忽略警告信息
from rasterio.windows import Window
import unidecode
import json

# 常量定义
GEOLOGY_TYPE_FIELD_POLYGON = '成土母'
GEOLOGY_TYPE_FIELD_POINT = 'MZ'
LAND_USE_FIELD = 'DLMC'
NIR_BAND_INDEX = 8
RED_BAND_INDEX = 4

class Timer:
    """用于计时的上下文管理器类"""
    def __init__(self, description):
        self.description = description

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.duration = self.end - self.start
        logging.info(f"{self.description}: {self.duration:.2f} 秒")

def setup_logging(log_dir='logs'):
    """设置日志记录
    
    Args:
        log_dir: 日志文件存储目录
    """
    # 创建日志目录
    os.makedirs(log_dir, exist_ok=True)
    
    # 生成日志文件名，包含时间戳
    log_file = os.path.join(log_dir, f'geology_processing_{time.strftime("%Y%m%d_%H%M%S")}.log')
    
    # 配置日志记录器
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    
    logging.info("日志系统初始化完成")
    return log_file

def get_system_info():
    """获取系统信息，用于优化性能参数"""
    memory = psutil.virtual_memory()
    cpu_count = multiprocessing.cpu_count()
    
    logging.info(f"""
    系统信息:
    - CPU核心数: {cpu_count}
    - 总内存: {memory.total / (1024**3):.2f} GB
    - 可用内存: {memory.available / (1024**3):.2f} GB
    - 内存使用率: {memory.percent}%
    """)
    
    return {
        'cpu_count': cpu_count,
        'total_memory': memory.total,
        'available_memory': memory.available,
        'memory_percent': memory.percent
    }

def get_optimal_chunk_size(data_shape, dtype_size=4):
    """计算最优的数据处理块大小
    
    Args:
        data_shape: 数据形状
        dtype_size: 数据类型大小（字节）
    
    Returns:
        chunk_size: 最优块大小
    """
    # 获取可用内存
    available_memory = psutil.virtual_memory().available
    
    # 使用25%的可用内存作为安全阈值
    safe_memory = available_memory * 0.25
    
    # 计算单个数据点占用的内存
    point_memory = dtype_size * np.prod(data_shape[:-2])  # 考虑多个波段/特征
    
    # 计算可以同时处理的数据点数量
    chunk_size = int(safe_memory / point_memory)
    
    # 确保chunk_size不小于1000
    chunk_size = max(1000, chunk_size)
    
    logging.info(f"计算得到的最优块大小: {chunk_size}")
    return chunk_size

def check_input_files(file_paths):
    """检查输入文件是否存在并可访问
    
    Args:
        file_paths: 文件路径列表
    
    Raises:
        FileNotFoundError: 件不存在
        PermissionError: 如果文件无法访问
    """
    for file_path in file_paths:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"找不到文件: {file_path}")
        if not os.access(file_path, os.R_OK):
            raise PermissionError(f"无法读取文件: {file_path}")
        logging.info(f"文件检查通过: {file_path}")
def preprocess_data(dem_path, satellite_image_path, geology_points_path, geology_polygons_path, 
                    slope_position_path, land_use_path, vector_boundary=None):
    logging.info("开始数据预处理...")
    
    try:
        with Timer("DEM数据读取"):
            dem = rasterio.open(dem_path)
            dem_array = dem.read(1, masked=True)
            logging.info(f"DEM数据形状: {dem_array.shape}")
        
        with Timer("坡位数据读取"):
            slope_position = rasterio.open(slope_position_path)
            slope_position_array = slope_position.read(1, masked=True)
            logging.info(f"坡位数据形状: {slope_position_array.shape}")
        
        with Timer("卫星影像读取"):
            satellite = rasterio.open(satellite_image_path)
            satellite_array = satellite.read()
            logging.info(f"卫星影像形状: {satellite_array.shape}")
        
        with Timer("矢量数据读取"):
            geology_points = gpd.read_file(geology_points_path, encoding='utf-8')
            geology_polygons = gpd.read_file(geology_polygons_path, encoding='utf-8')
            land_use = gpd.read_file(land_use_path, encoding='utf-8')
            logging.info(f"地质点数量: {len(geology_points)}")
            logging.info(f"地质面数量: {len(geology_polygons)}")
            logging.info(f"土地利用数量: {len(land_use)}")
        
        if vector_boundary is not None:
            logging.info("矢量边界将用于预测和更新范围")
        
        return dem, dem_array, satellite, satellite_array, geology_points, geology_polygons, \
               slope_position_array, land_use, vector_boundary
    
    except Exception as e:
        logging.error(f"数据预处理失败: {str(e)}")
        raise

def clip_raster_to_vector(raster, vector):
    try:
        if isinstance(raster, str):
            # 如果 raster 是文件路径
            with rasterio.open(raster) as src:
                out_image, out_transform = rasterio_mask(src, vector.geometry, crop=True, all_touched=True)
                out_meta = src.meta.copy()
        else:
            # 如果 raster 已经是一个打开的数据集
            out_image, out_transform = rasterio_mask(raster, vector.geometry, crop=True, all_touched=True)
            out_meta = raster.meta.copy()

        out_meta.update({
            "driver": "GTiff",
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "transform": out_transform
        })

        # 创建一个临时的内存文件来存储裁剪后的栅格
        with rasterio.io.MemoryFile() as memfile:
            with memfile.open(**out_meta) as dataset:
                dataset.write(out_image)
            
            # 重新打开内存文件以返回一个新的数据集
            return memfile.open(), out_image[0]

    except ValueError as e:
        if "Input shapes do not overlap raster" in str(e):
            logging.error("矢量边界与栅格数据不重叠,请检查数据范围和坐标系统")
        raise
    except Exception as e:
        logging.error(f"裁剪栅格到矢量时发生错误: {str(e)}")
        raise

def get_common_bounds(datasets):
    bounds = []
    for dataset in datasets:
        if isinstance(dataset, gpd.GeoDataFrame):
            bounds.append(dataset.total_bounds)
        elif isinstance(dataset, rasterio.io.DatasetReader):
            bounds.append(dataset.bounds)
    return (
        max(b[0] for b in bounds),  # minx
        max(b[1] for b in bounds),  # miny
        min(b[2] for b in bounds),  # maxx
        min(b[3] for b in bounds)   # maxy
    )

def clip_raster_to_bounds(raster, bounds):
    with rasterio.open(raster) as src:
        window = rasterio.windows.from_bounds(*bounds, transform=src.transform)
        out_image = src.read(window=window)
        out_transform = rasterio.windows.transform(window, src.transform)
        out_meta = src.meta.copy()
        out_meta.update({
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "transform": out_transform
        })
    return rasterio.io.MemoryFile().open(**out_meta), out_image[0]

def clip_vector_to_bounds(vector, bounds):
    bbox = box(*bounds)
    return vector.clip(bbox)
def align_and_clip_raster(raster_data, vector_data, all_touched=False):
    """栅格数据对齐和裁剪函数
    
    Args:
        raster_data: 栅格数据
        vector_data: 矢量数据
        all_touched: 是否包含所有接触的像素
    
    Returns:
        tuple: 裁剪后的数和变换参数
    """
    try:
        bounds = vector_data.total_bounds
        
        if isinstance(raster_data, np.ndarray):
            height, width = raster_data.shape
            transform = from_bounds(*bounds, width, height)
            
            # 创建掩膜
            with Timer("创建栅格掩膜"):
                mask = geometry_mask(
                    vector_data.geometry,
                    out_shape=raster_data.shape,
                    transform=transform,
                    invert=True,
                    all_touched=all_touched
                )
            
            return raster_data * mask, transform
        else:
            # 使用rasterio的掩膜功能
            with Timer("格剪"):
                out_image, out_transform = rasterio_mask(
                    raster_data,
                    [{'type': 'Polygon', 'coordinates': [[(bounds[0], bounds[1]),
                                                        (bounds[2], bounds[1]),
                                                        (bounds[2], bounds[3]),
                                                        (bounds[0], bounds[3])]]}],
                    crop=True
                )
                
                mask = geometry_mask(
                    vector_data.geometry,
                    out_shape=out_image.shape[1:],
                    transform=out_transform,
                    invert=True,
                    all_touched=all_touched
                )
            
            return out_image[0] * mask, out_transform
    
    except Exception as e:
        logging.error(f"栅格对齐和裁剪失败: {str(e)}")
        raise

def rasterize_vector_data(vector_data, attribute, reference_raster):
    """矢量数据栅格化函数
    
    Args:
        vector_data: 矢数据
        attribute: 属性字段
        reference_raster: 参考栅格
    
    Returns:
        tuple: 栅格化���果和值映射字典
    """
    try:
        with Timer("矢量数据栅格化"):
            # 创建唯一值映射
            unique_values = vector_data[attribute].unique()
            value_map = {value: i for i, value in enumerate(unique_values)}
            
            # 栅格化处理
            shapes = [(geom, value_map[value]) 
                     for geom, value in zip(vector_data.geometry, vector_data[attribute])]
            
            rasterized = rasterize(
                shapes,
                out_shape=reference_raster.shape,
                transform=reference_raster.transform,
                dtype='float32',
                all_touched=True
            )
            
            logging.info(f"栅格化完成，唯一值数量: {len(unique_values)}")
            return rasterized, value_map
    
    except Exception as e:
        logging.error(f"矢量数据栅格化失败: {str(e)}")
        raise

def extract_features(dem, satellite, slope_position_array, land_use):
    """特征提取函数
    
    Args:
        dem: DEM数据
        satellite: 卫星影像数据
        slope_position_array: 坡位数据
        land_use: 土地利用数据
    
    Returns:
        tuple: 提取的特征数据和相关参数
    """
    try:
        with Timer("特征提取"):
            # 获取参考信息
            ref_profile = dem.profile
            ref_transform = dem.transform
            ref_shape = dem.shape
            
            logging.info("开始采样处理...")
            
            def resample_raster(src, ref_profile):
                """内部重采样函数"""
                if isinstance(src, np.ndarray):
                    zoom_factors = (ref_profile['height'] / src.shape[0],
                                  ref_profile['width'] / src.shape[1])
                    return zoom(src, zoom_factors, order=1)
                else:
                    return src.read(
                        out_shape=(src.count, ref_profile['height'], ref_profile['width']),
                        resampling=Resampling.bilinear
                    )
            
            # 读取和重采数据
            dem_array = dem.read(1)
            satellite_array = resample_raster(satellite, ref_profile)
            slope_position_array = resample_raster(slope_position_array, ref_profile)
            
            # 栅格化土地利用数据
            land_use_raster, land_use_map = rasterize_vector_data(
                land_use, LAND_USE_FIELD, dem)
            
            # 计算NDVI
            logging.info("计算NDVI...")
            nir = satellite_array[NIR_BAND_INDEX].astype(np.float32)
            red = satellite_array[RED_BAND_INDEX].astype(np.float32)
            ndvi = np.divide(nir - red, nir + red + 1e-8, 
                           out=np.zeros_like(nir), where=(nir + red) != 0)
            
            # 组合特征
            features = np.vstack((
                dem_array[np.newaxis, ...],
                ndvi[np.newaxis, ...],
                satellite_array,
                slope_position_array[np.newaxis, ...],
                land_use_raster[np.newaxis, ...]
            ))
            
            logging.info(f"特征提取完成，特征形状: {features.shape}")
            return features, land_use_map, ref_transform, dem_array
    
    except Exception as e:
        logging.error(f"特征提取失败: {str(e)}")
        raise

def prepare_training_data(features, geology_points, dem):
    """准备训练数据
    
    Args:
        features: 特征数据
        geology_points: 地质点数据
        dem: DEM数据
    
    Returns:
        tuple: 训练数Xy
    """
    try:
        with Timer("准备训练数据"):
            label_encoder = LabelEncoder()
            geology_pixels = []
            skipped_points = 0
            
            # 收集地质点像素位置
            for idx, point in geology_points.iterrows():
                x, y = point.geometry.x, point.geometry.y
                rock_type = point[GEOLOGY_TYPE_FIELD_POINT]
                try:
                    row, col = dem.index(x, y)
                    if 0 <= row < features.shape[1] and 0 <= col < features.shape[2]:
                        geology_pixels.append((row, col, rock_type))
                    else:
                        skipped_points += 1
                except IndexError:
                    skipped_points += 1
            
            # 提取特征和标签
            X = []
            y = []
            for row, col, rock_type in geology_pixels:
                X.append(features[:, row, col])
                y.append(rock_type)
            
            X = np.array(X)
            y = np.array(y)
            
            # 使用 LabelEncoder 对岩石类型编码
            label_encoder.fit(y)
            y_encoded = label_encoder.transform(y)
            
            # 为未知类别添加一个标签
            unknown_label = len(label_encoder.classes_)
            label_encoder.classes_ = np.append(label_encoder.classes_, 'unknown')
            
            logging.info(f"""
            训练数据准备完成:
            - 有效样本数: {len(X)}
            - 跳数: {skipped_points}
            - 特征维度: {X.shape}
            - 唯一岩石类型数: {len(label_encoder.classes_)}
            - 岩石类型映射: {dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))}
            - 未知类别标签: {unknown_label}
            """)
            
            return X, y_encoded, label_encoder, unknown_label
    
    except Exception as e:
        logging.error(f"训练数据准备失败: {str(e)}")
        raise
    
def train_random_forest(X, y):
    """训练随机森林模型
    
    Args:
        X: 特征数据
        y: 标签数据
    
    Returns:
        RandomForestClassifier: 训练好的模型
    """
    try:
        with Timer("随机森林模型训练"):
            # 数据分割
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            logging.info(f"训练集大小: {X_train.shape}, 测试集大小: {X_test.shape}")
            
            # 初始化并训练模型
            rf = RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                n_jobs=-1,  # 使用所有CPU核心
                verbose=1    # 显示训练进度
            )
            
            rf.fit(X_train, y_train)
            
            # 模型评估
            y_pred = rf.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            logging.info(f"""
            随森林模型评估结果:
            - 准确率: {accuracy:.4f}
            - 详细分类报告:
            {classification_report(y_test, y_pred)}
            """)
            
            return rf
            
    except Exception as e:
        logging.error(f"随机森林模型训练失败: {str(e)}")
        raise

def train_xgboost(X, y):
    try:
        with Timer("XGBoost模型训练"):
            # 使用分层抽样来分割数据
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # 标签编码
            le = LabelEncoder()
            y_train_encoded = le.fit_transform(y_train)
            
            # 初始化并训练模型
            xgb_model = xgb.XGBClassifier(
                eval_metric='mlogloss',
                n_jobs=-1,
                verbosity=1,
                use_label_encoder=False
            )
            
            # 训练模型
            xgb_model.fit(
                X_train, 
                y_train_encoded,
                eval_set=[(X_test, le.transform(y_test))],
                early_stopping_rounds=10
            )
            
            # 模型评估
            y_pred = xgb_model.predict(X_test)
            y_pred = le.inverse_transform(y_pred)
            
            logging.info(f"""
            XGBoost模型评估结果:
            - 准确率: {accuracy_score(y_test, y_pred):.4f}
            - 详细分类报告:
            {classification_report(y_test, y_pred)}
            """)
            
            return xgb_model, le
            
    except Exception as e:
        logging.error(f"XGBoost模型训练失败: {str(e)}")
        raise

def process_batch(batch_data):
    """处理数据批次
    
    Args:
        batch_data: 包含特征和模型的元组
    
    Returns:
        tuple: (随机林预测结果, XGBoost测结果)
    """
    try:
        features, rf_model, xgb_model, label_encoder, unknown_label = batch_data
        
        # 进行预测
        rf_pred = rf_model.predict(features)
        xgb_pred = label_encoder.inverse_transform(xgb_model.predict(features))
        
        return rf_pred, xgb_pred
        
    except Exception as e:
        logging.error(f"批处理失败: {str(e)}")
        raise

def predict_with_memory_optimization(features, model, batch_size):
    """使用内存优化的预测函
    
    Args:
        features: 特征数据
        model: 训练好的模型
        batch_size: 批次大小
    
    Returns:
        array: 预测结果
    """
    try:
        n_samples = features.shape[0]
        predictions = np.empty(n_samples, dtype=object)
        
        for i in range(0, n_samples, batch_size):
            end = min(i + batch_size, n_samples)
            batch_predictions = model.predict(features[i:end])
            predictions[i:end] = batch_predictions
            
            if i % (batch_size * 10) == 0:
                gc.collect()  # 定期垃圾回收
                
        return predictions
        
    except Exception as e:
        logging.error(f"预失败: {str(e)}")
        raise

def predict_and_update_boundaries(rf_model, xgb_model, label_encoder, features, dem, geology_polygons, intermediate_output_path, unknown_label, vector_boundary=None):
    try:
        with Timer("预测和边界更新"):
            logging.info("开始预测和更新边界")
            
            # 获取特征数据的形状
            n_features, height, width = features.shape
            
            # 创建预测掩膜
            if vector_boundary is not None:
                mask = rasterio.features.geometry_mask(
                    [vector_boundary.geometry.iloc[0]],
                    out_shape=(height, width),
                    transform=dem.transform,
                    invert=True
                )
            else:
                mask = np.ones((height, width), dtype=bool)
            
            # 创建内存映射文件来存储预测结果
            rf_predictions_path = os.path.join(intermediate_output_path, 'rf_predictions.npy')
            xgb_predictions_path = os.path.join(intermediate_output_path, 'xgb_predictions.npy')
            
            rf_predictions = np.memmap(rf_predictions_path, dtype='int16', mode='w+', shape=(height, width))
            xgb_predictions = np.memmap(xgb_predictions_path, dtype='int16', mode='w+', shape=(height, width))
            
            # 初始化为未知类别
            rf_predictions[:] = unknown_label
            xgb_predictions[:] = unknown_label
            
            # 定义每个块的大小
            block_size = 1000  # 可以根据需要调整
            
            # 计算块的数量
            n_blocks_y = (height + block_size - 1) // block_size
            n_blocks_x = (width + block_size - 1) // block_size
            
            total_blocks = n_blocks_y * n_blocks_x
            logging.info(f"总块数: {total_blocks}")
            
            # 使用进程池进行并行处理
            with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
                futures = []
                
                # 提交块处理任务
                for i in range(n_blocks_y):
                    for j in range(n_blocks_x):
                        y_start = i * block_size
                        y_end = min((i + 1) * block_size, height)
                        x_start = j * block_size
                        x_end = min((j + 1) * block_size, width)
                        
                        block_features = features[:, y_start:y_end, x_start:x_end].copy()
                        block_mask = mask[y_start:y_end, x_start:x_end]
                        
                        if np.any(block_mask):
                            future = executor.submit(
                                process_block,
                                (block_features, rf_model, xgb_model, label_encoder, unknown_label, block_mask)
                            )
                            futures.append((future, y_start, y_end, x_start, x_end))
                
                # 处理结果
                with tqdm(total=len(futures), desc="处理块") as pbar:
                    for future, y_start, y_end, x_start, x_end in futures:
                        try:
                            rf_pred, xgb_pred = future.result()
                            rf_predictions[y_start:y_end, x_start:x_end] = rf_pred
                            xgb_predictions[y_start:y_end, x_start:x_end] = xgb_pred
                        except Exception as e:
                            logging.error(f"处理块 ({y_start}:{y_end}, {x_start}:{x_end}) 时发生错误: {str(e)}")
                        pbar.update(1)
                        
                        if pbar.n % 100 == 0:
                            gc.collect()  # 定期垃圾回收
            
            # 合并预测结果
            try:
                final_predictions = np.memmap(os.path.join(intermediate_output_path, 'final_predictions.npy'), 
                                              dtype='int16', mode='w+', shape=(height, width))
                
                total_blocks = ((height + block_size - 1) // block_size) * ((width + block_size - 1) // block_size)
                with tqdm(total=total_blocks, desc="合并预测结果") as pbar:
                    for i in range(0, height, block_size):
                        for j in range(0, width, block_size):
                            y_end = min(i + block_size, height)
                            x_end = min(j + block_size, width)
                            
                            rf_block = rf_predictions[i:y_end, j:x_end]
                            final_predictions[i:y_end, j:x_end] = rf_block
                            pbar.update(1)
                
                final_predictions.flush()
            except Exception as e:
                logging.error(f"合并预测结果时发生错误: {str(e)}")
                raise

            # 关闭内存射文件
            del rf_predictions
            del xgb_predictions

            # 更新边界
            updated_polygons = update_geology_boundaries(
                final_predictions,
                dem,
                geology_polygons,
                vector_boundary
            )

            # 清理文件
            def safe_delete(file_path):
                try:
                    os.unlink(file_path)
                except (PermissionError, FileNotFoundError) as e:
                    logging.warning(f"无法删除文件 {file_path}: {str(e)}")

            safe_delete(rf_predictions_path)
            safe_delete(xgb_predictions_path)
            
            # 注意：我们不删除 final_predictions，因为我们需要在后续步骤中使用它
            # safe_delete(os.path.join(intermediate_output_path, 'final_predictions.npy'))

            return updated_polygons, final_predictions  # 返回 final_predictions 而是 None
            
    except Exception as e:
        logging.error(f"预测和边界更新失败: {str(e)}")
        raise

def process_block(data):
    features, rf_model, xgb_model, label_encoder, unknown_label, mask = data
    n_features, block_height, block_width = features.shape
    features_reshaped = features.reshape(n_features, -1).T
    
    rf_pred = np.full((block_height, block_width), unknown_label, dtype='int16')
    xgb_pred = np.full((block_height, block_width), unknown_label, dtype='int16')
    
    mask_flat = mask.flatten()
    features_masked = features_reshaped[mask_flat]
    
    if features_masked.shape[0] > 0:
        try:
            rf_pred_masked = rf_model.predict(features_masked)
            xgb_pred_masked = xgb_model.predict(features_masked)
            
            rf_pred_flat = rf_pred.flatten()
            xgb_pred_flat = xgb_pred.flatten()
            
            rf_pred_flat[mask_flat] = rf_pred_masked
            xgb_pred_flat[mask_flat] = xgb_pred_masked
            
            rf_pred = rf_pred_flat.reshape((block_height, block_width))
            xgb_pred = xgb_pred_flat.reshape((block_height, block_width))
        except Exception as e:
            logging.error(f"处理块时发生错误: {str(e)}")
            # 如果发生错误，保持默认的 unknown_label
    
    return rf_pred, xgb_pred

def save_prediction_results(rf_predictions, xgb_predictions, dem, output_path):
    """保存预测结果
    
    Args:
        rf_predictions: 随机森林预测结果
        xgb_predictions: XGBoost预测结果
        dem: DEM数据
        output_path: 输出路径
    """
    try:
        # 保存随机森林预测结果
        rf_output_path = os.path.join(output_path, 'rf_predictions.tif')
        with rasterio.open(rf_output_path, 'w', **dem.profile) as dst:
            dst.write(rf_predictions.astype(rasterio.float32), 1)
        logging.info(f"随机森林预测结果已保存: {rf_output_path}")
        
        # 保存XGBoost预测结果
        xgb_output_path = os.path.join(output_path, 'xgb_predictions.tif')
        with rasterio.open(xgb_output_path, 'w', **dem.profile) as dst:
            dst.write(xgb_predictions.astype(rasterio.float32), 1)
        logging.info(f"XGBoost预测结果已保存: {xgb_output_path}")
        
    except Exception as e:
        logging.error(f"保存预测结果失败: {str(e)}")
        raise
def update_geology_boundaries(predicted_mask, dem, geology_polygons, vector_boundary=None):
    try:
        with Timer("地质边界更新"):
            logging.info("开始将预测结果转换为矢量")
            
            # 获取唯一值并创建映射
            unique_values = np.unique(predicted_mask)
            value_map = {value: i for i, value in enumerate(unique_values)}
            predicted_mask_numeric = np.vectorize(value_map.get)(predicted_mask)
            
            # 将预测结果转换为矢量
            shapes = list(rasterio.features.shapes(
                predicted_mask_numeric,
                transform=dem.transform
            ))
            
            # 创建几何对象和岩石类型列表
            geometries = [shape(geom) for geom, _ in shapes]
            rock_types = [value for _, value in shapes]
            
            logging.info(f"创建GeoDataFrame，共{len(geometries)}个多边形")
            
            # 创建GeoDataFrame
            predicted_polygons = gpd.GeoDataFrame(
                {'geometry': geometries, 'rock_type': rock_types},
                crs=geology_polygons.crs
            )
            
            # 将数值映射回岩石类型
            predicted_polygons['rock_type'] = predicted_polygons['rock_type'].map(
                {v: k for k, v in value_map.items()}
            )
            
            # 如果提供了矢量边界，则裁剪预测结果
            if vector_boundary is not None:
                predicted_polygons = gpd.clip(predicted_polygons, vector_boundary)
            
            return predicted_polygons
            
    except Exception as e:
        logging.error(f"地质边界更新失败: {str(e)}")
        raise

def process_polygon_chunk(chunk_bounds, predicted_polygons, geology_polygons):
    """处理多边形分块
    
    Args:
        chunk_bounds: 分块边界
        predicted_polygons: 预测的多边形
        geology_polygons: 原始地质多边形
    
    Returns:
        GeoDataFrame: 处理后的多边形
    """
    try:
        chunk_box = box(*chunk_bounds)
        chunk_predicted = predicted_polygons[predicted_polygons.intersects(chunk_box)]
        chunk_geology = geology_polygons[geology_polygons.intersects(chunk_box)]
        return gpd.overlay(chunk_geology, chunk_predicted, how='union')
        
    except Exception as e:
        logging.error(f"多边形分块处理失败: {str(e)}")
        raise

def determine_geology_type(adjusted_polygons, original_polygons, geology_points, predicted_mask, transform):
    try:
        with Timer("地质类型确定"):
            logging.info("开始确定地质类型")
            
            if predicted_mask is None:
                logging.error("预测掩码为 None，无法进行地质类型确定")
                return adjusted_polygons

            # 初始化新的列
            adjusted_polygons[GEOLOGY_TYPE_FIELD_POLYGON] = None
            adjusted_polygons['confidence'] = 0
            
            # 1. 使用原始多边形的类型
            logging.info("步骤1: 使用原始多边形的类型")
            with tqdm(total=len(adjusted_polygons), desc="处理原始多边形") as pbar:
                for idx, adjusted_poly in adjusted_polygons.iterrows():
                    overlaps = gpd.overlay(
                        original_polygons,
                        gpd.GeoDataFrame({'geometry': [adjusted_poly.geometry]}, crs=original_polygons.crs),
                        how='intersection'
                    )
                    if not overlaps.empty:
                        # 计算重叠区域的面积
                        overlaps['overlap_area'] = overlaps.geometry.area
                        max_overlap = overlaps['overlap_area'].idxmax()
                        adjusted_polygons.at[idx, GEOLOGY_TYPE_FIELD_POLYGON] = \
                            overlaps.loc[max_overlap, GEOLOGY_TYPE_FIELD_POLYGON]
                        adjusted_polygons.at[idx, 'confidence'] = \
                            overlaps.loc[max_overlap, 'overlap_area'] / adjusted_poly.geometry.area
                    pbar.update(1)
            
            # 2. 考虑野外采集的地质点数据
            logging.info("步骤2: 考虑野外采集的地质点数据")
            with tqdm(total=len(adjusted_polygons), desc="处理地质点数据") as pbar:
                for idx, poly in adjusted_polygons.iterrows():
                    points_in_poly = geology_points[geology_points.within(poly.geometry)]
                    if not points_in_poly.empty:
                        point_types = points_in_poly[GEOLOGY_TYPE_FIELD_POINT].value_counts()
                        most_common_type = point_types.index[0]
                        type_confidence = point_types[most_common_type] / len(points_in_poly)
                        
                        if type_confidence > adjusted_polygons.at[idx, 'confidence']:
                            adjusted_polygons.at[idx, GEOLOGY_TYPE_FIELD_POLYGON] = most_common_type
                            adjusted_polygons.at[idx, 'confidence'] = type_confidence
                    pbar.update(1)
            
            # 3. 使用预测结果
            logging.info("步骤3: 使用预测结果")
            chunk_size = 1000  # 可以根据可用内存调整这个值
            with tqdm(total=len(adjusted_polygons), desc="处理预测结果") as pbar:
                for i in range(0, len(adjusted_polygons), chunk_size):
                    chunk = adjusted_polygons.iloc[i:i+chunk_size]
                    for idx, poly in chunk.iterrows():
                        if adjusted_polygons.at[idx, 'confidence'] < 0.5:
                            mask = rasterio.features.rasterize(
                                [poly.geometry],
                                out_shape=predicted_mask.shape,
                                transform=transform,
                                dtype='uint8'
                            )
                            predicted_types = predicted_mask[mask.astype(bool)]
                            if len(predicted_types) > 0:
                                type_counts = np.bincount(predicted_types)
                                most_common_type = type_counts.argmax()
                                type_confidence = type_counts[most_common_type] / len(predicted_types)
                                
                                if type_confidence > adjusted_polygons.at[idx, 'confidence']:
                                    adjusted_polygons.at[idx, GEOLOGY_TYPE_FIELD_POLYGON] = most_common_type
                                    adjusted_polygons.at[idx, 'confidence'] = type_confidence
                        pbar.update(1)
            
            # 4. 处理未分类的多边形
            logging.info("步骤4: 处理未分类的多边形")
            unclassified = adjusted_polygons[adjusted_polygons[GEOLOGY_TYPE_FIELD_POLYGON].isnull()]
            if not unclassified.empty:
                logging.warning(f"有 {len(unclassified)} 个多边形未被分类")
                # 这里可以添加额外的处理逻辑，例如分配给最近的已分类多边形
            
            return adjusted_polygons
            
    except Exception as e:
        logging.error(f"地质类型确定失败: {str(e)}")
        raise

def sanitize_column_name(name):
    """将中文字段名转换为 ASCII 兼容的名称"""
    ascii_name = unidecode.unidecode(name).replace(' ', '_')
    return ascii_name[:10]  # Shapefile 限制字段名长度为 10 个字符

def main(vector_boundary_path=None):
    """主函数"""
    try:
        # 设置日志
        log_file = setup_logging()
        logging.info("开始处理...")
        
        # 获取系统信息
        system_info = get_system_info()
        
        # 设置输入文件路径
        input_paths = {
            'dem_path': r"F:\rock_type_test\raster\DEM.tif",
            'satellite_image_path': r"F:\rock_type_test\raster\multi_bands.tif",
            'geology_points_path': r"F:\rock_type_test\shp\rock_points.shp",
            'geology_polygons_path': r"F:\rock_type_test\shp\rock_type.shp",
            'slope_position_path': r"F:\rock_type_test\raster\slopeclass.tif",
            'land_use_path': r"F:\rock_type_test\shp\land_use.shp"
        }
        
        # 设置输出路径
        intermediate_output_path = r'F:\rock_type_test\intermediate_results'
        final_output_path = r'F:\rock_type_test\result'
        
        # 创建输出目录
        os.makedirs(intermediate_output_path, exist_ok=True)
        os.makedirs(final_output_path, exist_ok=True)
        
        # 检查输入文件
        check_input_files(input_paths.values())
        
        with Timer("总处理时间"):
            # 如果提供了矢量边界，则读取它
            vector_boundary = None
            if vector_boundary_path:
                vector_boundary = gpd.read_file(vector_boundary_path)
                logging.info(f"使用提供的矢量边界进行处理")
            
            # 数据预处理
            dem, dem_array, satellite, satellite_array, geology_points, \
            geology_polygons, slope_position_array, land_use, vector_boundary = preprocess_data(
                **input_paths, vector_boundary=vector_boundary
            )
            
            # 如果需要，在这里进行坐标系统的检查和转换
            if vector_boundary is not None and vector_boundary.crs != geology_polygons.crs:
                vector_boundary = vector_boundary.to_crs(geology_polygons.crs)
                logging.info("矢量边界已转换为与地质多边形相同的坐标系统")
            
            # 特征提取
            features, land_use_map, dem_transform, dem_array = extract_features(
                dem, satellite, slope_position_array, land_use
            )
            
            # 准备训练数据
            X, y, label_encoder, unknown_label = prepare_training_data(features, geology_points, dem)
            
            # 训练模型
            rf_model = train_random_forest(X, y)
            xgb_model, _ = train_xgboost(X, y)  # 不需要单独的 label_encoder
            
            # 预测和更新边界
            updated_polygons, predicted_mask = predict_and_update_boundaries(
                rf_model, xgb_model, label_encoder, features, dem,
                geology_polygons, intermediate_output_path, unknown_label, vector_boundary
            )
            
            # 确定地质类型
            final_polygons = determine_geology_type(
                updated_polygons, geology_polygons, geology_points,
                predicted_mask, dem.transform
            )
            
            # 创建字段名映射
            field_mapping = {col: sanitize_column_name(col) for col in final_polygons.columns}
            
            # 重命名列
            final_polygons = final_polygons.rename(columns=field_mapping)
            
            # 保存结果
            output_file = os.path.join(final_output_path, 'final_geology_map.shp')
            final_polygons.to_file(output_file)
            logging.info(f"处理完成，结果已保存到: {output_file}")
            
            # 保存字段名映射为元数据
            metadata_file = os.path.join(final_output_path, 'field_mapping.json')
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(field_mapping, f, ensure_ascii=False, indent=2)
            logging.info(f"字段名映射已保存到: {metadata_file}")
            
            # 清理内存
            gc.collect()
        
    except Exception as e:
        logging.error(f"处理失败: {str(e)}", exc_info=True)
        raise
    finally:
        # 清理最终的预测结果文件
        final_predictions_path = os.path.join(intermediate_output_path, 'final_predictions.npy')
        if os.path.exists(final_predictions_path):
            try:
                os.unlink(final_predictions_path)
            except (PermissionError, FileNotFoundError) as e:
                logging.warning(f"无法删除文件 {final_predictions_path}: {str(e)}")
        
        logging.info("处理结束")

if __name__ == "__main__":
    vector_boundary_path = r"F:\rock_type_test\shp\extent.shp"
    main(vector_boundary_path)

