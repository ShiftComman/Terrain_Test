import os
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.features import rasterize, geometry_mask  # 添加 geometry_mask
from rasterio.mask import mask as rasterio_mask
from rasterio.transform import from_bounds  # 添加 from_bounds
from rasterio.enums import Resampling  # 添加 Resampling
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
import json

# 保留原有的常量定义
GEOLOGY_TYPE_FIELD_POLYGON = '成土母'
GEOLOGY_TYPE_FIELD_POINT = 'MZ'
LAND_USE_FIELD = 'DLMC'
NIR_BAND_INDEX = 8
RED_BAND_INDEX = 4

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def preprocess_data(dem_path, satellite_image_path, geology_points_path, geology_polygons_path, 
                    slope_position_path, land_use_path):
    # 读取DEM
    dem = rasterio.open(dem_path)
    dem_array = dem.read(1)
    
    # 读取坡位数据
    slope_position = rasterio.open(slope_position_path)
    slope_position_array = slope_position.read(1)
    
    # 读取卫星影像
    satellite = rasterio.open(satellite_image_path)
    satellite_array = satellite.read()
    
    # 读取地质点数据
    geology_points = gpd.read_file(geology_points_path)
    
    # 读取地质面数据
    geology_polygons = gpd.read_file(geology_polygons_path)
    
    # 读取土地利用类型面数据
    land_use = gpd.read_file(land_use_path)
    
    return dem, dem_array, satellite, satellite_array, geology_points, geology_polygons, slope_position_array, land_use

def align_and_clip_raster(raster_data, vector_data, all_touched=False):
    """将栅格数据裁剪到矢量数据的范围，并保持栅格的投影和分辨率"""
    # 获取矢量数据的边界
    bounds = vector_data.total_bounds
    
    if isinstance(raster_data, np.ndarray):
        # 如果输入是NumPy数组，我们需要创建一个临时的仿射变换
        height, width = raster_data.shape
        transform = from_bounds(*bounds, width, height)
        
        # 创建一个掩膜，只保留矢数据覆盖的区域
        mask = geometry_mask(vector_data.geometry, out_shape=raster_data.shape, transform=transform, invert=True, all_touched=all_touched)
        
        # 应用掩膜
        out_image = raster_data * mask
        
        return out_image, transform
    else:
        # 如果输入是rasterio数据集，使用原来的方法
        out_image, out_transform = rasterio_mask(raster_data, [{'type': 'Polygon', 'coordinates': [[(bounds[0], bounds[1]), (bounds[2], bounds[1]), (bounds[2], bounds[3]), (bounds[0], bounds[3])]]}], crop=True)
        
        # 创建一个掩膜，只保留矢量数据覆盖的区域
        mask = geometry_mask(vector_data.geometry, out_shape=out_image.shape[1:], transform=out_transform, invert=True, all_touched=all_touched)
        
        # 应用掩膜
        out_image = out_image * mask
        
        return out_image[0], out_transform

def rasterize_vector_data(vector_data, attribute, reference_raster):
    """将矢量数据栅格化，使用参考栅格的投影和分辨率"""
    # 创建一个字典，唯一的属性值映射到整数
    unique_values = vector_data[attribute].unique()
    value_map = {value: i for i, value in enumerate(unique_values)}
    
    # 使用映射后的数值进行栅格化
    rasterized = rasterize(
        [(geom, value_map[value]) for geom, value in zip(vector_data.geometry, vector_data[attribute])],
        out_shape=reference_raster.shape,
        transform=reference_raster.transform,
        dtype='float32',
        all_touched=True
    )
    return rasterized, value_map

def extract_features(dem, satellite, slope_position_array, land_use):
    # 使用DEM作为参考栅格
    ref_profile = dem.profile
    ref_transform = dem.transform
    ref_shape = dem.shape

    # 重采样函数
    def resample_raster(src, ref_profile):
        if isinstance(src, np.ndarray):
            # 如果输入是NumPy数组，使用scipy.ndimage.zoom进行重采样
            zoom_factors = (ref_profile['height'] / src.shape[0], ref_profile['width'] / src.shape[1])
            return zoom(src, zoom_factors, order=1)
        else:
            # 如果输入是rasterio数据集，使用rasterio的重采样方法
            data = src.read(
                out_shape=(src.count, ref_profile['height'], ref_profile['width']),
                resampling=Resampling.bilinear
            )
            return data

    # 重采样所有栅格数据
    dem_array = dem.read(1)
    satellite_array = resample_raster(satellite, ref_profile)
    slope_position_array = resample_raster(slope_position_array, ref_profile)

    # 栅格化土地利用数据
    land_use_raster, land_use_map = rasterize_vector_data(land_use, LAND_USE_FIELD, dem)

    # 计算NDVI
    nir = satellite_array[NIR_BAND_INDEX].astype(np.float64)
    red = satellite_array[RED_BAND_INDEX].astype(np.float64)
    ndvi = (nir - red) / (nir + red + 1e-8)

    # 组合所有特征
    features = np.vstack((dem_array[np.newaxis, ...], 
                          ndvi[np.newaxis, ...], 
                          satellite_array, 
                          slope_position_array[np.newaxis, ...], 
                          land_use_raster[np.newaxis, ...]))

    return features, land_use_map, ref_transform, dem_array

def prepare_training_data(features, geology_points, dem):
    geology_pixels = []
    for idx, point in geology_points.iterrows():
        x, y = point.geometry.x, point.geometry.y
        rock_type = point[GEOLOGY_TYPE_FIELD_POINT]
        try:
            row, col = dem.index(x, y)
            geology_pixels.append((row, col, rock_type))
        except IndexError:
            print(f"点 {idx} 坐标 ({x}, {y}) 超出范围。")
    
    # 创建标签编码器
    le = LabelEncoder()
    rock_types = le.fit_transform([point[2] for point in geology_pixels])
    
    X = []
    y = []
    for (row, col, _), rock_type in zip(geology_pixels, rock_types):
        if 0 <= row < features.shape[1] and 0 <= col < features.shape[2]:
            X.append(features[:, row, col])
            y.append(rock_type)
        else:
            print(f"跳过超出边界的像素 ({row}, {col})。")
    
    print(f"准备的训练数据数量：{len(X)}")
    print(f"训练数据中的岩石类型：{set(y)}")
    
    return np.array(X), np.array(y), le

def train_random_forest(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=1)
    rf.fit(X_train, y_train)
    
    y_pred = rf.predict(X_test)
    print("Random Forest Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred, zero_division=0))
    
    return rf

def train_xgboost(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 将标签转换为整数编码
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)
    y_test_encoded = le.transform(y_test)
    
    xgb_model = xgb.XGBClassifier(eval_metric='mlogloss', n_jobs=1)
    xgb_model.fit(X_train, y_train_encoded)
    
    y_pred_encoded = xgb_model.predict(X_test)
    y_pred = le.inverse_transform(y_pred_encoded)
    
    print("XGBoost Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred, zero_division=0))
    
    return xgb_model, le

def process_batch(batch_data):
    features, rf_model, xgb_model, label_encoder = batch_data
    rf_pred = rf_model.predict(features)
    xgb_pred = label_encoder.inverse_transform(xgb_model.predict(features))
    return rf_pred, xgb_pred

def process_chunk(chunk_bounds, predicted_polygons, geology_polygons):
    chunk_box = box(*chunk_bounds)
    chunk_predicted = predicted_polygons[predicted_polygons.intersects(chunk_box)]
    chunk_geology = geology_polygons[geology_polygons.intersects(chunk_box)]
    return gpd.overlay(chunk_geology, chunk_predicted, how='union')

def predict_and_update_boundaries(rf_model, xgb_model, label_encoder, features, dem, geology_polygons, intermediate_output_path):
    logging.info("开始预测和更新边界")
    
    features = features.astype(np.float32)
    n_samples = features.shape[1] * features.shape[2]
    n_features = features.shape[0]
    
    rf_predictions = np.empty((features.shape[1], features.shape[2]), dtype=object)
    xgb_predictions = np.empty((features.shape[1], features.shape[2]), dtype=object)
    
    batch_size = 1000000
    n_batches = (n_samples + batch_size - 1) // batch_size
    
    logging.info(f"总像元: {n_samples}, 批次大小: {batch_size}, 总批次数: {n_batches}")
    
    with multiprocessing.Pool() as pool:
        batch_data = ((features.reshape(n_features, -1)[:, batch].T, rf_model, xgb_model, label_encoder) 
                      for batch in gen_batches(n_samples, batch_size))
        results = list(tqdm(pool.imap(process_batch, batch_data), total=n_batches, desc="预测进度"))
    
    for i, (rf_pred, xgb_pred) in enumerate(results):
        start = i * batch_size
        end = min((i + 1) * batch_size, n_samples)
        rf_predictions.ravel()[start:end] = rf_pred
        xgb_predictions.ravel()[start:end] = xgb_pred
    
    logging.info("预测完成，开始保存中间结果")
    
    # 创建一个新的 profile，并更新 dtype 和 nodata 值
    output_profile = dem.profile.copy()
    output_profile.update(dtype=rasterio.uint8, nodata=255)  # 使用 255 作为 nodata 值
    
    print(f"DEM profile: {dem.profile}")
    print(f"Output profile: {output_profile}")
    print(f"RF predictions shape: {rf_predictions.shape}")
    print(f"RF predictions dtype: {rf_predictions.dtype}")
    print(f"XGB predictions shape: {xgb_predictions.shape}")
    print(f"XGB predictions dtype: {xgb_predictions.dtype}")
    
    # 保存随机森林预测结果
    rf_output_path = os.path.join(intermediate_output_path, 'rf_predictions.tif')
    with rasterio.open(rf_output_path, 'w', **output_profile) as dst:
        rf_pred_encoded = label_encoder.transform(rf_predictions.ravel()).reshape(rf_predictions.shape)
        dst.write(rf_pred_encoded.astype(rasterio.uint8), 1)
    logging.info(f"随机森林预测结果已保存到 {rf_output_path}")
    
    # 保存XGBoost预测结果
    xgb_output_path = os.path.join(intermediate_output_path, 'xgb_predictions.tif')
    with rasterio.open(xgb_output_path, 'w', **output_profile) as dst:
        xgb_pred_encoded = label_encoder.transform(xgb_predictions.ravel()).reshape(xgb_predictions.shape)
        dst.write(xgb_pred_encoded.astype(rasterio.uint8), 1)
    logging.info(f"XGBoost预测结果已保存到 {xgb_output_path}")
    
    logging.info("开始合并结果")
    
    # 合并预测结果
    final_predictions = np.where(rf_predictions == xgb_predictions, rf_predictions, rf_predictions)
    
    # 创建predicted_mask
    predicted_mask = final_predictions
    
    # 保存最终预测结果
    final_output_path = os.path.join(intermediate_output_path, 'final_predictions.tif')
    with rasterio.open(final_output_path, 'w', **dem.profile) as dst:
        dst.write(final_predictions.astype(rasterio.float32), 1)
    logging.info(f"最终预测结果已保存到 {final_output_path}")
    
    logging.info("开始将预测结果转换为矢量")
    
    # 将预测结果转换为数值
    unique_values = np.unique(predicted_mask)
    value_map = {value: i for i, value in enumerate(unique_values)}
    predicted_mask_numeric = np.vectorize(value_map.get)(predicted_mask)
    
    # 将预测结果转换为矢量
    shapes = list(rasterio.features.shapes(predicted_mask_numeric, transform=dem.transform))
    geometries = [shape(geom) for geom, _ in shapes]
    rock_types = [value for _, value in shapes]
    
    logging.info("创建 GeoDataFrame")
    
    # 创建 GeoDataFrame
    predicted_polygons = gpd.GeoDataFrame(
        {'geometry': geometries, 'rock_type': rock_types},
        crs=geology_polygons.crs
    )
    
    # 将数值映射回岩石类型
    predicted_polygons['rock_type'] = predicted_polygons['rock_type'].map({v: k for k, v in value_map.items()})
    
    logging.info("更新地质边界")
    
    # 获取整个区域的边界
    total_bounds = geology_polygons.total_bounds
    
    # 将区域分成 4x4 的网格
    x_splits = np.linspace(total_bounds[0], total_bounds[2], 5)
    y_splits = np.linspace(total_bounds[1], total_bounds[3], 5)
    
    chunks = []
    for i in range(4):
        for j in range(4):
            chunk_bounds = (x_splits[i], y_splits[j], x_splits[i+1], y_splits[j+1])
            chunks.append(chunk_bounds)
    
    # 并行处理每个块
    with ProcessPoolExecutor() as executor:
        results = list(tqdm(executor.map(process_chunk, chunks, [predicted_polygons]*16, [geology_polygons]*16), 
                            total=16, desc="处理区块"))
    
    # 合并结果
    updated_polygons = pd.concat(results)
    
    logging.info("边界更新完成")
    
    logging.info("保存标签编码映射")
    label_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
    label_mapping_path = os.path.join(intermediate_output_path, 'label_mapping.json')
    with open(label_mapping_path, 'w', encoding='utf-8') as f:
        json.dump(label_mapping, f, ensure_ascii=False, indent=4)
    logging.info(f"标签编码映射已保存到 {label_mapping_path}")
    
    return updated_polygons, predicted_mask

def determine_geology_type(adjusted_polygons, original_polygons, geology_points, predicted_mask, transform):
    logging.info("开始确定地质类")
    
    # 为调整后的多边形分配地质类型
    adjusted_polygons[GEOLOGY_TYPE_FIELD_POLYGON] = None
    adjusted_polygons['confidence'] = 0
    
    # 1. 使用原始多边形的类型
    logging.info("步骤1: 使用原始多边形的类型")
    for idx, adjusted_poly in tqdm(adjusted_polygons.iterrows(), total=len(adjusted_polygons), desc="处理原始多边形"):
        overlaps = original_polygons.overlay(gpd.GeoDataFrame({'geometry': [adjusted_poly.geometry]}, crs=original_polygons.crs), how='intersection')
        if not overlaps.empty:
            max_overlap = overlaps.area.idxmax()
            adjusted_polygons.at[idx, GEOLOGY_TYPE_FIELD_POLYGON] = original_polygons.loc[max_overlap, GEOLOGY_TYPE_FIELD_POLYGON]
            adjusted_polygons.at[idx, 'confidence'] = overlaps.loc[max_overlap, 'area'] / adjusted_poly.area
    
    # 2. 考虑野外采集的地质点数据
    logging.info("步骤2: 考虑野外采集的地质点数据")
    for idx, poly in tqdm(adjusted_polygons.iterrows(), total=len(adjusted_polygons), desc="处理地质点数据"):
        points_in_poly = geology_points[geology_points.within(poly.geometry)]
        if not points_in_poly.empty:
            point_types = points_in_poly[GEOLOGY_TYPE_FIELD_POINT].value_counts()
            most_common_type = point_types.index[0]
            type_confidence = point_types[most_common_type] / len(points_in_poly)
            
            if type_confidence > adjusted_polygons.at[idx, 'confidence']:
                adjusted_polygons.at[idx, GEOLOGY_TYPE_FIELD_POLYGON] = most_common_type
                adjusted_polygons.at[idx, 'confidence'] = type_confidence
    
    # 3. 使用预测结果
    logging.info("步骤3: 使用预测结")
    for idx, poly in tqdm(adjusted_polygons.iterrows(), total=len(adjusted_polygons), desc="处理预测结果"):
        if adjusted_polygons.at[idx, 'confidence'] < 0.5:
            mask = rasterio.features.rasterize([poly.geometry], out_shape=predicted_mask.shape, transform=transform)
            predicted_types = predicted_mask[mask]
            if len(predicted_types) > 0:
                type_counts = np.bincount(predicted_types)
                most_common_type = type_counts.argmax()
                type_confidence = type_counts[most_common_type] / len(predicted_types)
                
                if type_confidence > adjusted_polygons.at[idx, 'confidence']:
                    adjusted_polygons.at[idx, GEOLOGY_TYPE_FIELD_POLYGON] = most_common_type
                    adjusted_polygons.at[idx, 'confidence'] = type_confidence
    
    # 4. 处理仍然没有类型的多边形（如果有的话）
    logging.info("步骤4: 处理未分类的多边形")
    unclassified = adjusted_polygons[adjusted_polygons[GEOLOGY_TYPE_FIELD_POLYGON].isnull()]
    for idx, poly in tqdm(unclassified.iterrows(), total=len(unclassified), desc="处理未分类多边形"):
        neighbors = adjusted_polygons[adjusted_polygons.geometry.touches(poly.geometry)]
        if not neighbors.empty:
            neighbor_types = neighbors[GEOLOGY_TYPE_FIELD_POLYGON].value_counts()
            most_common_type = neighbor_types.index[0]
            adjusted_polygons.at[idx, GEOLOGY_TYPE_FIELD_POLYGON] = most_common_type
            adjusted_polygons.at[idx, 'confidence'] = neighbor_types[most_common_type] / len(neighbors)
    
    logging.info("地质类型确定完成")
    return adjusted_polygons

def check_points_in_raster(points, raster):
    in_bounds = 0
    out_bounds = 0
    for idx, point in points.iterrows():
        x, y = point.geometry.x, point.geometry.y
        row, col = raster.index(x, y)
        if 0 <= row < raster.height and 0 <= col < raster.width:
            in_bounds += 1
        else:
            out_bounds += 1
    print(f"在DEM范围内的点：{in_bounds}")
    print(f"在DEM范围外的点：{out_bounds}")

def main():
    # 设置输入文件路径
    dem_path = r"F:\rock_type_test\raster\DEM.tif"
    satellite_image_path = r"F:\rock_type_test\raster\multi_bands.tif"
    geology_points_path = r"F:\rock_type_test\shp\rock_points.shp"
    geology_polygons_path = r"F:\rock_type_test\shp\rock_type.shp"
    slope_position_path = r"F:\rock_type_test\raster\slopeclass.tif"
    land_use_path = r"F:\rock_type_test\shp\land_use.shp"
    # 添加中间结果输出路径
    intermediate_output_path = r'F:\rock_type_test\intermediate_results'
    os.makedirs(intermediate_output_path, exist_ok=True)
    
    logging.info("开始预处理数据")
    dem, dem_array, satellite, satellite_array, geology_points, geology_polygons, \
    slope_position_array, land_use = preprocess_data(
        dem_path, satellite_image_path, geology_points_path, geology_polygons_path,
        slope_position_path, land_use_path
    )
    
    logging.info("开始提取特征")
    features, land_use_map, dem_transform, dem_array = extract_features(dem, satellite, slope_position_array, land_use)
    
    logging.info("准备训练数据")
    X, y, label_encoder = prepare_training_data(features, geology_points, dem)
    
    logging.info("开始训练随机森林模型")
    rf_model = train_random_forest(X, y)
    
    logging.info("开始训练XGBoost模型")
    xgb_model, label_encoder = train_xgboost(X, y)
    
    
    logging.info("开始使用模型预测并更新边界")
    updated_polygons, predicted_mask = predict_and_update_boundaries(rf_model, xgb_model, label_encoder, features, dem, geology_polygons, intermediate_output_path)
    
    logging.info("开始确定最终地质类型")
    final_polygons = determine_geology_type(updated_polygons, geology_polygons, geology_points, predicted_mask, dem.transform)
    
    logging.info("开始保存最终结果")
    output_path = r'F:\rock_type_test\result\final_geology_map.shp'
    final_polygons.to_file(output_path)
    logging.info(f"地质图更新完成，结果已保存到 {output_path}")

if __name__ == "__main__":
    main()

