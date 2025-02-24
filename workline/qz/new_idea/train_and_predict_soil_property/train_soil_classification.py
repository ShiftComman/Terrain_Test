import os
import logging
import pickle
import json
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.feature_selection import RFECV
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path

def balance_classes(X, y, min_samples, logger):
    """平衡类别，确保每个类别至少有指定数量的样本"""
    unique_classes = np.unique(y)
    X_resampled = X.copy()
    y_resampled = y.copy()
    
    for cls in unique_classes:
        cls_idx = np.where(y == cls)[0]
        n_samples = len(cls_idx)
        
        if n_samples < min_samples:
            # 计算需要复制的次数
            n_copies = int(np.ceil(min_samples / n_samples))
            logger.info(f"类别 {cls} 样本数量不足 {min_samples}，将通过复制增加到 {n_samples * n_copies} 个样本")
            
            # 复制样本
            X_cls = X.iloc[cls_idx]
            y_cls = y[cls_idx]
            
            for _ in range(n_copies - 1):
                X_resampled = pd.concat([X_resampled, X_cls])
                y_resampled = np.concatenate([y_resampled, y_cls])
    
    return X_resampled, y_resampled

def load_and_clean_data(file_path, label_col, logger):
    """加载数据并进行清洗"""
    logger.info(f"正在从 {file_path} 加载数据")
    data = pd.read_csv(file_path)
    
    # 记录原始数据大小
    original_size = len(data)
    
    # 删除包含'/'的行
    data = data[~data[label_col].astype(str).str.contains('/', na=False)]
    
    # 删除包含'*'的行
    data = data[~data[label_col].astype(str).str.contains('\*', na=False)]
    
    # 删除空值行
    data = data.dropna(subset=[label_col])
    
    # 记录清洗后的数据大小
    cleaned_size = len(data)
    
    # 添加类别分布统计
    class_dist = data[label_col].value_counts()
    logger.info(f"\n类别分布统计：\n{class_dist}")
    
    # 可以选择过滤掉样本数过少的类别
    min_samples_per_class = 10  # 设置每个类别的最小样本数
    valid_classes = class_dist[class_dist >= min_samples_per_class].index
    data = data[data[label_col].isin(valid_classes)]
    
    # 记录过滤后的数据大小
    filtered_size = len(data)
    
    logger.info(f"数据清洗完成。原始数据量：{original_size}，清洗后数据量：{cleaned_size}，"
                f"过滤小类别后数据量：{filtered_size}")
    logger.info(f"删除的行数：清洗阶段 {original_size - cleaned_size}，"
                f"过滤小类别阶段 {cleaned_size - filtered_size}")
    
    return data

def encode_labels(data, label_col, save_dir, logger):
    """对标签进行编码并保存编码映射"""
    logger.info(f"开始对 {label_col} 进行标签编码")
    
    # 创建标签编码器
    le = LabelEncoder()
    
    # 获取唯一的标签值并排序
    unique_labels = sorted(data[label_col].unique())
    
    # 拟合编码器
    le.fit(unique_labels)
    
    # 创建编码映射字典，将 numpy 类型转换为 Python 原生类型
    encoding_map = {str(k): int(v) for k, v in zip(le.classes_, le.transform(le.classes_))}
    
    # 保存编码映射
    save_dir = Path(save_dir)
    mapping_dir = save_dir / 'label_mappings'
    mapping_dir.mkdir(parents=True, exist_ok=True)
    
    with open(mapping_dir / f"{label_col}_mapping.json", 'w', encoding='utf-8') as f:
        json.dump(encoding_map, f, ensure_ascii=False, indent=4)
    
    logger.info(f"标签编码映射已保存至 {mapping_dir / f'{label_col}_mapping.json'}")
    
    # 转换数据
    encoded_labels = le.transform(data[label_col])
    
    return encoded_labels, le

def preprocess_data(df, feature_cols, label_col, logger):
    """预处理数据"""
    # 检查并转换特征列的数据类型
    for col in feature_cols:
        if df[col].dtype == 'object':
            logger.warning(f"列 {col} 包含非数值数据。尝试进行转换。")
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            except ValueError:
                logger.error(f"无法将列 {col} 转换为数值。考虑删除此列或对其进行编码。")
                return None
    
    # 仅保留需要的列
    df = df.loc[:, [label_col] + feature_cols]
    
    # 处理缺失值
    df = df.dropna()
    
    # 检查是否有无限值，并将其替换为 NaN
    df = df.replace([np.inf, -np.inf], np.nan)
    
    # 再次删除包含 NaN 的行
    df = df.dropna()
    
    logger.info(f"数据预处理完成。新形状：{df.shape}")
    return df

def feature_optimization(X, y, estimator, feature_cols, logger):
    """特征优化"""
    logger.info("开始特征优化")
    # 根据最小类别样本数调整交叉验证折数
    n_splits = min(5, min(np.bincount(y)))
    logger.info(f"根据最小类别样本数，调整交叉验证折数为：{n_splits}")
    
    selector = RFECV(estimator, step=1, cv=n_splits, scoring='f1_weighted', n_jobs=-1)
    selector = selector.fit(X[feature_cols], y)
    selected_features = list(np.array(feature_cols)[selector.support_])
    logger.info(f"选择的特征：{selected_features}")
    return selected_features

def hyperparameter_tuning(X, y, estimator, param_grid, cv, logger):
    """超参数调优"""
    logger.info("开始超参数调优")
    # 根据最小类别样本数调整交叉验证折数
    n_splits = min(cv, min(np.bincount(y)))
    logger.info(f"根据最小类别样本数，调整交叉验证折数为：{n_splits}")
    
    n_iter_search = 50
    random_search = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=param_grid,
        n_iter=n_iter_search,
        cv=n_splits,
        scoring='f1_weighted',
        random_state=42,
        n_jobs=-1
    )
    random_search.fit(X, y)
    best_params = random_search.best_params_
    logger.info(f"随机搜索得到的最佳参数：{best_params}")
    
    # 在最佳参数周围进行网格搜索
    param_grid_fine = {
        'n_estimators': [max(10, best_params['n_estimators'] - 50),
                        best_params['n_estimators'],
                        min(1000, best_params['n_estimators'] + 50)],
        'min_samples_split': [max(2, best_params['min_samples_split'] - 2),
                             best_params['min_samples_split'],
                             best_params['min_samples_split'] + 2],
        'min_samples_leaf': [max(1, best_params['min_samples_leaf'] - 1),
                            best_params['min_samples_leaf'],
                            best_params['min_samples_leaf'] + 1]
    }
    
    if best_params['max_depth'] is None:
        param_grid_fine['max_depth'] = [None, 10, 20]
    else:
        param_grid_fine['max_depth'] = [max(1, best_params['max_depth'] - 5),
                                       best_params['max_depth'],
                                       best_params['max_depth'] + 5]
    
    grid_search = GridSearchCV(estimator=estimator,
                             param_grid=param_grid_fine,
                             cv=cv,
                             scoring='f1_weighted',
                             n_jobs=-1)
    grid_search.fit(X, y)
    best_params = grid_search.best_params_
    logger.info(f"网格搜索得到的最佳参数：{best_params}")
    return grid_search.best_estimator_

def evaluate_classification(y_true, y_pred):
    """评估分类模型"""
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    conf_matrix = confusion_matrix(y_true, y_pred)
    
    # 添加每个类别的评估指标
    class_report = pd.DataFrame(
        classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    ).T
    
    return {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "Confusion_Matrix": conf_matrix,
        "Class_Report": class_report
    }

def save_model(model, feature_names, label_encoder, save_dir, label_col, logger):
    """保存模型"""
    model_dir = save_dir / 'models'
    model_dir.mkdir(parents=True, exist_ok=True)
    
    model_data = {
        'model': model,
        'feature_names': feature_names,
        'label_encoder': label_encoder,
        'n_features': len(feature_names)
    }
    
    with open(model_dir / f"{label_col}_model.pkl", 'wb') as f:
        pickle.dump(model_data, f)
    
    logger.info(f"模型已保存，特征数量：{len(feature_names)}，特征列表：{feature_names}")

def create_classification_report(results, save_dir, logger):
    """生成分类报告"""
    report_dir = save_dir / 'reports'
    report_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建Excel报告
    with pd.ExcelWriter(report_dir / "classification_summary.xlsx") as writer:
        # 特征选择sheet
        feature_data = []
        for label, data in results.items():
            if data:
                for feature in data['selected_features']:
                    feature_data.append({
                        'Label': label,
                        'Feature': feature
                    })
        if feature_data:
            pd.DataFrame(feature_data).to_excel(writer, sheet_name='Selected Features', index=False)
        
        # 模型性能sheet
        performance_data = []
        for label, data in results.items():
            if data:
                class_report = data['metrics']['Class_Report']
                # 使用整体评估指标（在classification_report中是'weighted avg'行）
                weighted_avg = class_report.loc['weighted avg']
                performance_data.append({
                    'Label': label,
                    'Accuracy': class_report.loc['accuracy']['precision'],  # accuracy在class_report中的位置
                    'Precision': weighted_avg['precision'],
                    'Recall': weighted_avg['recall'],
                    'F1': weighted_avg['f1-score'],
                    'Support': weighted_avg['support']
                })
        if performance_data:
            pd.DataFrame(performance_data).to_excel(writer, sheet_name='Model Performance', index=False)
        
        # 超参数sheet
        hyperparams_data = []
        for label, data in results.items():
            if data:
                params = data['best_params']
                hyperparams_data.append({
                    'Label': label,
                    'n_estimators': params['n_estimators'],
                    'max_depth': params['max_depth'],
                    'min_samples_split': params['min_samples_split'],
                    'min_samples_leaf': params['min_samples_leaf']
                })
        if hyperparams_data:
            pd.DataFrame(hyperparams_data).to_excel(writer, sheet_name='Hyperparameters', index=False)
        
        # 添加每个类别的详细性能sheet
        for label, data in results.items():
            if data:
                class_report = data['metrics']['Class_Report']
                # 只保留具体类别的行，删除汇总行
                detailed_report = class_report.drop(['accuracy', 'macro avg', 'weighted avg'])
                detailed_report.to_excel(writer, sheet_name=f'{label}_Class_Details')
    
    logger.info(f"分类报告已保存至 {report_dir}")

def train_classification_model(df, label_col, feature_cols, param_grid, save_dir, use_feature_optimization, cv, logger):
    """训练分类模型"""
    logger.info(f"正在训练 {label_col} 的分类模型")
    
    # 数据预处理
    df_processed = preprocess_data(df, feature_cols, label_col, logger)
    if df_processed is None:
        logger.error(f"{label_col} 的数据预处理失败")
        return None
    
    # 标签编码
    y, label_encoder = encode_labels(df_processed, label_col, save_dir, logger)
    X = df_processed[feature_cols]
    
    # 确保每个类别至少有 2*cv 个样本
    min_samples_required = 2 * cv
    X_balanced, y_balanced = balance_classes(X, y, min_samples_required, logger)
    
    # 特征优化
    if use_feature_optimization:
        estimator = RandomForestClassifier(random_state=42)
        selected_features = feature_optimization(X_balanced, y_balanced, estimator, feature_cols, logger)
        X_selected = X_balanced[selected_features]
    else:
        selected_features = feature_cols
        X_selected = X_balanced
    
    # 使用分层采样进行训练集和测试集划分
    X_train, X_test, y_train, y_test = train_test_split(
        X_selected, y_balanced, 
        test_size=0.2, 
        random_state=42,
        stratify=y_balanced
    )
    
    # 训练模型
    estimator = RandomForestClassifier(random_state=42)
    best_model = hyperparameter_tuning(X_train, y_train, estimator, param_grid, cv, logger)
    
    # 评估模型
    y_pred = best_model.predict(X_test)
    model_metrics = evaluate_classification(y_test, y_pred)
    
    # 记录详细的评估结果
    logger.info(f"\n分类报告：\n{model_metrics['Class_Report']}")
    logger.info(f"\n混淆矩阵：\n{model_metrics['Confusion_Matrix']}")
    
    # 如果某些类别的预测效果特别差，记录警告
    class_report = model_metrics['Class_Report']
    for class_name, class_metrics in class_report.iterrows():
        if class_name not in ['accuracy', 'macro avg', 'weighted avg']:
            if float(class_metrics['f1-score']) < 0.5:  # 可以调整这个阈值
                logger.warning(f"类别 {class_name} 的F1分数较低: {class_metrics['f1-score']:.3f}")
                logger.warning(f"支持度: {class_metrics['support']}, 精确率: {class_metrics['precision']:.3f}, 召回率: {class_metrics['recall']:.3f}")
    
    # 保存模型
    save_model(best_model, selected_features, label_encoder, save_dir, label_col, logger)
    
    return {
        "selected_features": selected_features,
        "metrics": model_metrics,
        "best_params": best_model.get_params(),
        "best_model": best_model
    }

def main(file_path, label_cols, feature_cols, param_grid, save_dir, log_file, use_feature_optimization, cv):
    """主函数"""
    # 设置日志
    if log_file:
        log_dir = Path(log_file).parent
        log_dir.mkdir(parents=True, exist_ok=True)
        logging.basicConfig(
            filename=log_file,
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            encoding='utf-8'
        )
    else:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            encoding='utf-8'
        )
    
    logger = logging.getLogger(__name__)
    logger.info("开始训练分类模型")
    
    try:
        results = {}
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # 确保所有特征名称都是字符串
        feature_cols = [str(col) for col in feature_cols]
        
        for label_col in tqdm(label_cols, desc="处理标签"):
            # 加载和清洗数据
            df = load_and_clean_data(file_path, label_col, logger)
            
            # 训练模型
            result = train_classification_model(
                df, label_col, feature_cols, param_grid,
                save_dir, use_feature_optimization, cv, logger
            )
            
            results[label_col] = result
            if result is None:
                logger.warning(f"模型训练失败：{label_col}")
        
        # 生成分类报告
        create_classification_report(results, save_dir, logger)
        
        logger.info("所有分类模型已训练和评估完毕。结果已保存。")
        
    except Exception as e:
        logger.error(f"主函数中发生错误：{str(e)}")
        raise
    finally:
        logger.info("分类模型训练完成")

if __name__ == "__main__":
    # 测试配置
    file_path = r"G:\soil_property_result\qzs\table\soil_property_point.csv"
    label_cols = ["TRZD"]  # 修改为列表
    feature_cols = ['PH','CEC','OM','TN','TP','TK','TSE','AP','SK','AK','HG','AS2','PB','CD','CR','TRRZ','GZCHD','YXTCHD',]  # 修改为列表
    feature_cols = ['aspect', 'carbonate', 'channelnetworkbaselevel', 'channelnetworkdistance', 'clay_minerals', 'contrast', 
                'convergenceindex', 'correlation', 'dem', 'dissimilarity', 'dl', 'dz', 'entropy', 'etp22_3', 'etp22_mean', 
                'evi', 'ferrous_minerals', 'hillshade', 'homogeneity', 'lsfactor', 'lswi', 'mean', 'mndwi', 
                'mrrtf', 'mrvbf', 'ndmi', 'ndvi', 'ndwi', 'night22_', 'pc1', 'pc2', 'plancurvature', 'pre22_3', 'pre22_mean', 
                'profilecurvature', 'relativeslopeposition', 'rock_outcrop', 'savi', 'secondmoment', 'slope', 'slopepostion', 
                'terrainruggednessindex', 'tmp22_3', 'tmp22_mean', 'topographicwetnessindex', 'totalcatchmentarea', 'valleydepth',
                'vari', 'variance','lon','lat']
    param_grid = {
        'n_estimators': np.arange(10, 1000, 10),
        'max_depth': [None] + list(np.arange(10, 100, 10)),
        'min_samples_split': np.arange(2, 10, 1),
        'min_samples_leaf': np.arange(1, 10, 1)
    }
    use_feature_optimization = True
    save_dir = r"G:\soil_property_result\qzs\models\soil_property_class"
    log_file = r"G:\soil_property_result\qzs\logs\train_soil_property_class.log"
    cv = 10
    main(file_path, label_cols, feature_cols,param_grid, save_dir, log_file, use_feature_optimization, cv)