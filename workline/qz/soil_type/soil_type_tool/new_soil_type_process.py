import numpy as np
import pandas as pd
import geopandas as gpd
from sklearn.experimental import enable_iterative_imputer
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import RobustScaler, FunctionTransformer
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import cKDTree
from sklearn.model_selection import train_test_split
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from shapely.geometry import Point
from sklearn.mixture import GaussianMixture
from scipy.spatial.distance import cdist
from sklearn.preprocessing import PolynomialFeatures
from sklearn.mixture import BayesianGaussianMixture
# 图可以显示中文
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

class SoilSampleOptimizer:
    def __init__(self, 
                 target_samples=385,
                 min_samples=30,      # 添加最小样本数参数
                 spatial_resolution=1000,  # 修改为米单位
                 min_cluster_size=5,
                 pca_variance=0.95,
                 imbalance_threshold=0.3,
                 x_col='lon',            # X坐标列名
                 y_col='lat',            # Y坐标列名
                 class_col='NEW_TZ',     # 类别列名
                 feature_cols=None,     # 特征列名列表
                 env_weight=0.7,
                 min_features=3):  # 新增最小特征数参数
        """
        土壤样本优化器
        
        参数：
        target_samples: 样本数的参考值，用于判断是否过多
        min_samples: 每个类别的最小样本数（满足10折交叉验证）
        spatial_resolution: 空间聚类分辨率（米）
        min_cluster_size: 最小聚类样本数
        pca_variance: PCA保留的方差比例
        imbalance_threshold: 类别不平衡阈值
        x_col: X坐标列名
        y_col: Y坐标列名
        class_col: 类别列名
        feature_cols: 用于特征处理的列名列表，如果为None则只使用空间特征
        env_weight: 环境因素在样本生成中的权重(0-1)，越大表示越注重环境相似性
        min_features: 最小特征数
        """
        self.target_samples = target_samples
        self.min_samples = min_samples
        self.spatial_resolution = spatial_resolution
        self.min_cluster_size = min_cluster_size
        self.pca_variance = pca_variance
        self.imbalance_threshold = imbalance_threshold
        self.x_col = x_col
        self.y_col = y_col
        self.class_col = class_col
        self.feature_cols = feature_cols or []
        self.crs = "EPSG:4545"  # CGCS2000投影坐标系
        self.env_weight = env_weight
        self.min_features = min_features

    def _enhance_spatial_features(self, gdf):
        """增强空间特征（适配投影坐标）"""
        try:
            # 生成网格编码（米单位）
            gdf['grid_x'] = np.floor(gdf.geometry.x / self.spatial_resolution)
            gdf['grid_y'] = np.floor(gdf.geometry.y / self.spatial_resolution)
            
            # 计算局部密度（基于投影坐标的欧氏距离）
            coords = np.array([gdf.geometry.x, gdf.geometry.y]).T
            n_neighbors = min(5, len(coords)-1)  # 防止样本数小于5
            if n_neighbors < 2:
                gdf['local_density'] = 1.0  # 样本太少时使用默认值
                return gdf
                
            nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(coords)
            distances, _ = nbrs.kneighbors(coords)
            gdf['local_density'] = 1 / (distances[:, 1:].mean(axis=1) + 1e-6)
            
            return gdf
        except Exception as e:
            print(f"增强空间特征时出错: {str(e)}")
            raise

    def _spatial_clustering(self, gdf):
        """空间聚类分析（直接使用投影坐标）"""
        try:
            # 直接使用坐标进行聚类，不做范围验证
            coords = np.array([gdf.geometry.x, gdf.geometry.y]).T
            
            if len(coords) < self.min_cluster_size:
                print(f"警告：样本数({len(coords)})小于最小聚类数({self.min_cluster_size})")
                gdf['cluster'] = 0  # 所有点归为一类
                return gdf
            
            # 动态计算EPS参数（米单位）
            n_neighbors = min(5, len(coords)-1)
            nn = NearestNeighbors(n_neighbors=n_neighbors).fit(coords)
            distances, _ = nn.kneighbors(coords)
            eps = np.percentile(distances[:, -1], 75)
            
            # 执行DBSCAN聚类
            db = DBSCAN(
                eps=eps,
                min_samples=min(self.min_cluster_size, len(coords)//2)
            )
            clusters = db.fit_predict(coords)
            gdf['cluster'] = clusters
            
            # 打印聚类统计信息
            n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
            print(f"形成了 {n_clusters} 个空间簇")
            print(f"噪声点数量: {sum(clusters == -1)}")
            
            return gdf
            
        except Exception as e:
            print(f"空间聚类失败: {str(e)}")
            print("使用备用方案：基于网格的聚类")
            gdf['cluster'] = (
                (gdf.geometry.x // self.spatial_resolution).astype(str) + "_" +
                (gdf.geometry.y // self.spatial_resolution).astype(str)
            )
            return gdf

    def _calculate_env_similarity(self, gdf1, gdf2=None):
        """计算环境因素相似性"""
        env_features = [
            'dem', 'slope', 'aspect', 'slopepostion', # 地形因子
            'pre22_mean', 'tmp22_mean','etp22_mean'  # 气候因子
            'ndvi', 'evi','pc1',  # 植被因子
            'clay_minerals', 'carbonate','dl','ferrous_minerals','rock_outcrop'  # 母质特征
        ]
        
        # 提取有效的环境特征
        valid_features = [f for f in env_features if f in self.feature_cols]
        if not valid_features:
            return None
            
        # 标准化环境特征
        scaler = RobustScaler()
        features1 = scaler.fit_transform(gdf1[valid_features])
        
        if gdf2 is not None:
            features2 = scaler.transform(gdf2[valid_features])
            # 计算欧氏距离
            return cdist(features1, features2, metric='euclidean')
        return features1

    def _generate_samples(self, gdf, target_count):
        """改进的样本生成方法"""
        needed = target_count - len(gdf)
        if needed <= 0:
            return gdf
            
        try:
            # 1. 增强特征预处理
            coords = np.array([gdf.geometry.x, gdf.geometry.y]).T
            
            # 处理环境特征（增加缺失值处理和特征筛选）
            if self.feature_cols:
                # 选择数值型特征并处理缺失值
                numeric_features = gdf[self.feature_cols].select_dtypes(include=[np.number])
                numeric_features = numeric_features.apply(lambda x: x.fillna(x.median()) if x.notnull().any() else x)
                numeric_features = numeric_features.fillna(0)
                
                # 特征筛选（保留与目标类别相关性强的特征）
                corr_matrix = numeric_features.corrwith(gdf[self.class_col].astype('category').cat.codes)
                # 确保至少保留min_features个特征
                selected_features = corr_matrix[abs(corr_matrix) > 0.1].index.tolist() or numeric_features.columns.tolist()
                selected_features = selected_features[:max(self.min_features, len(selected_features))]
                
                if selected_features:
                    # 增加特征交互项
                    poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
                    interaction_features = poly.fit_transform(numeric_features[selected_features])
                    
                    # 标准化
                    scaler = RobustScaler()
                    scaled_features = scaler.fit_transform(interaction_features)
                    
                    # 动态调整权重（样本越少越依赖环境特征）
                    dynamic_weight = min(0.9, self.env_weight + (1 - len(gdf)/self.min_samples)*0.2)
                    combined_features = np.hstack([
                        coords * (1 - dynamic_weight),
                        scaled_features * dynamic_weight
                    ])
                else:
                    print("警告：没有显著相关特征，仅使用空间特征")
                    combined_features = coords
            else:
                combined_features = coords
                
            # 2. 使用贝叶斯GMM生成样本（增加稳定性）
            n_components = min(5, max(2, len(combined_features)//2))
            gmm = BayesianGaussianMixture(
                n_components=n_components,
                covariance_type='diag',
                weight_concentration_prior_type='dirichlet_process',
                init_params='random_from_data',  # 修正初始化方法
                max_iter=500,
                random_state=42  # 新增随机种子
            )
            
            # 在生成样本前增加数据验证
            if len(combined_features) < 2:
                raise ValueError("有效特征不足，无法生成样本")
                
            gmm.fit(combined_features)
            new_samples, _ = gmm.sample(needed*2)  # 生成双倍样本用于后续筛选
            
            # 3. 验证生成的样本
            env_features = new_samples[:, 2:] if combined_features.shape[1] > 2 else None
            valid_mask = self._validate_generated_samples(
                new_samples[:, :2], 
                env_features,
                gdf
            )
            valid_samples = new_samples[valid_mask][:needed]
            
            # 4. 如果有效样本不足，进行二次生成
            if len(valid_samples) < needed:
                print(f"初次生成有效样本不足({len(valid_samples)}/{needed})，进行补偿生成")
                compensation = gmm.sample((needed - len(valid_samples))*3)[0]
                valid_compensation = compensation[self._validate_generated_samples(
                    compensation[:, :2], 
                    compensation[:, 2:] if combined_features.shape[1] > 2 else None,
                    gdf
                )]
                valid_samples = np.vstack([valid_samples, valid_compensation[:needed-len(valid_samples)]])
            
            # 5. 创建新的GeoDataFrame
            new_coords = valid_samples[:, :2] / (1 - dynamic_weight)
            new_points = [Point(x, y) for x, y in new_coords]
            new_gdf = gpd.GeoDataFrame(
                geometry=new_points,
                crs=self.crs
            )
            
            # 6. 智能属性赋值（基于最近邻特征）
            nn_model = NearestNeighbors(n_neighbors=3).fit(combined_features)
            _, indices = nn_model.kneighbors(valid_samples)
            for col in gdf.columns:
                if col not in ['geometry', 'cluster', 'grid_x', 'grid_y', 'local_density']:
                    new_gdf[col] = gdf[col].iloc[indices[:, 0]].values
            
            return pd.concat([gdf, new_gdf])
            
        except Exception as e:
            print(f"样本生成失败: {str(e)}")
            print("使用简单复制作为备用方案")
            return self._simple_copy(gdf, target_count)

    def _validate_generated_samples(self, coords, env_features, original_gdf):
        """增强的样本验证"""
        # 增加对极小数据量的处理
        if len(original_gdf) < 5:
            return np.ones(len(coords), dtype=bool)
            
        # 1. 空间约束
        bbox = original_gdf.total_bounds
        spatial_valid = (
            (coords[:, 0] >= bbox[0]) & (coords[:, 0] <= bbox[2]) &
            (coords[:, 1] >= bbox[1]) & (coords[:, 1] <= bbox[3])
        )
        
        if env_features is None:
            return spatial_valid
            
        # 2. 环境特征约束
        orig_env = self._calculate_env_similarity(original_gdf)
        env_range = np.percentile(orig_env, [5, 95], axis=0)
        env_valid = np.all(
            (env_features >= env_range[0]) & 
            (env_features <= env_range[1]),
            axis=1
        )
        
        return spatial_valid & env_valid

    def _adaptive_downsample(self, gdf):
        """改进的降采样方法"""
        try:
            # 创建数据的深拷贝，避免 SettingWithCopyWarning
            gdf = gdf.copy()
            
            # 1. 空间聚类
            clusters = self._spatial_clustering(gdf)
            sampled_features = []
            
            # 2. 对每个聚类进行处理
            for cluster_id in clusters['cluster'].unique():
                # 使用 loc 进行索引，避免 SettingWithCopyWarning
                cluster_gdf = clusters.loc[clusters['cluster'] == cluster_id].copy()
                
                # 计算采样大小
                sample_size = self._calculate_cluster_sample_size(
                    cluster_gdf, len(gdf)
                )
                
                if sample_size >= len(cluster_gdf):
                    sampled_features.append(cluster_gdf)
                    continue
                
                # 3. 基于环境特征和空间分布进行采样
                if self.feature_cols:
                    # 选择数值型特征
                    numeric_features = cluster_gdf[self.feature_cols].select_dtypes(include=[np.number])
                    # 处理缺失值
                    numeric_features = numeric_features.fillna(numeric_features.median())
                    numeric_features = numeric_features.fillna(0)
                    
                    if not numeric_features.empty:
                        # 结合环境和空间特征
                        coords = np.array([cluster_gdf.geometry.x, cluster_gdf.geometry.y]).T
                        scaler = RobustScaler()
                        scaled_features = scaler.fit_transform(numeric_features)
                        
                        combined_features = np.hstack([
                            coords * (1 - self.env_weight),
                            scaled_features * self.env_weight
                        ])
                        
                        # 使用K-means++的思想选择代表性样本
                        selected_indices = self._kmeans_plus_plus_sampling(
                            combined_features, sample_size
                        )
                    else:
                        # 仅使用空间特征
                        selected_indices = self._density_based_sampling(
                            cluster_gdf, sample_size
                        )
                else:
                    # 仅使用空间特征
                    selected_indices = self._density_based_sampling(
                        cluster_gdf, sample_size
                    )
                
                sampled_features.append(cluster_gdf.iloc[selected_indices])
            
            return pd.concat(sampled_features, ignore_index=True)
            
        except Exception as e:
            print(f"降采样失败: {str(e)}")
            return gdf.sample(min(len(gdf), self.target_samples))

    def _calculate_cluster_sample_size(self, cluster_gdf, total_size):
        """计算聚类采样大小"""
        # 实现聚类采样大小的计算逻辑
        # 这里可以根据聚类的大小、类别不平衡等因素来计算采样大小
        # 这里只是一个示例，实际实现需要根据具体情况来调整
        return min(len(cluster_gdf), total_size // len(cluster_gdf))

    def _kmeans_plus_plus_sampling(self, features, sample_size):
        """K-means++采样方法"""
        # 实现K-means++采样方法的逻辑
        # 这里可以根据特征的分布来选择代表性样本
        # 这里只是一个示例，实际实现需要根据具体情况来调整
        return np.random.choice(len(features), sample_size, replace=False)

    def _density_based_sampling(self, cluster_gdf, sample_size):
        """密度采样方法"""
        # 实现密度采样方法的逻辑
        # 这里可以根据聚类的大小、密度等因素来选择代表性样本
        # 这里只是一个示例，实际实现需要根据具体情况来调整
        return np.random.choice(len(cluster_gdf), sample_size, replace=False)

    def _simple_copy(self, gdf, target_count):
        """改进的简单复制方法"""
        if len(gdf) == 0:
            return gdf
        # 增加随机扰动避免完全重复
        repeats = target_count // len(gdf) + 1
        copied = pd.concat([gdf] * repeats)[:target_count]
        copied['geometry'] = copied.geometry.apply(
            lambda p: Point(p.x + np.random.normal(0, 50),  # 50米随机扰动
                            p.y + np.random.normal(0, 50))
        )
        return copied

    def _copy_attributes(self, original_gdf, new_gdf):
        """复制属性"""
        for col in original_gdf.columns:
            if col not in ['geometry', 'cluster', 'grid_x', 'grid_y', 'local_density']:
                new_gdf[col] = original_gdf[col].iloc[0]

    def balance_dataset(self, input_path, output_path):
        """执行完整的样本平衡流程"""
        try:
            # 1. 加载数据
            print("正在加载数据...")
            df = pd.read_csv(input_path)
            print(f"原始数据: {len(df)} 条记录")
            
            # 2. 验证必要字段
            required_fields = [self.x_col, self.y_col, self.class_col] + self.feature_cols
            missing_fields = [f for f in required_fields if f not in df.columns]
            if missing_fields:
                raise ValueError(f"缺少必要字段: {', '.join(missing_fields)}")
            
            # 3. 创建GeoDataFrame（使用投影坐标）
            gdf = gpd.GeoDataFrame(
                df, 
                geometry=gpd.points_from_xy(df[self.x_col], df[self.y_col]),
                crs=self.crs
            )
            
            # 保存原始坐标列
            gdf['longitude'] = df[self.x_col]
            gdf['latitude'] = df[self.y_col]
            
            # 后续处理...
            print("正在增强空间特征...")
            gdf = self._enhance_spatial_features(gdf)
            
            print("正在进行空间聚类...")
            gdf = self._spatial_clustering(gdf)
            
            # 处理各类别...
            print("正在平衡各类别样本...")
            balanced_dfs = []
            category_counts = gdf[self.class_col].value_counts()
            
            print("\n原始类别分布:")
            print(category_counts)
            
            for category, count in tqdm(category_counts.items(), desc="处理类别"):
                # 使用 loc 进行索引，避免 SettingWithCopyWarning
                category_gdf = gdf.loc[gdf[self.class_col] == category].copy()
                
                if count > self.target_samples * 1.5:  # 样本数显著过多
                    print(f"\n类别 {category} 样本过多({count})，进行降采样")
                    sampled = self._adaptive_downsample(category_gdf)
                    print(f"降采样后数量: {len(sampled)}")
                elif count < self.min_samples:  # 样本数过少
                    print(f"\n类别 {category} 样本不足({count})，进行生成")
                    sampled = self._generate_samples(category_gdf, target_count=self.min_samples)
                    print(f"生成后数量: {len(sampled)}")
                else:
                    sampled = category_gdf
                    print(f"\n类别 {category} 样本数适中({count})，保持不变")
                
                balanced_dfs.append(sampled)
            
            # 合并结果
            final_gdf = pd.concat(balanced_dfs, ignore_index=True)
            final_gdf = final_gdf[final_gdf['cluster'] != -1]  # 移除噪声点
            
            # 保存结果（确保保留原始坐标）
            final_df = pd.DataFrame(final_gdf.drop(columns='geometry'))
            # 确保输出的lon和lat列使用原始投影坐标
            final_df['lon'] = final_df['longitude']
            final_df['lat'] = final_df['latitude']
            final_df = final_df.drop(columns=['longitude', 'latitude'])  # 删除临时列
            
            final_df.to_csv(output_path, index=False)
            
            print("\n处理完成:")
            print(f"原始样本数: {len(df)}")
            print(f"处理后样本数: {len(final_df)}")
            print("\n各类别样本数:")
            print(final_df[self.class_col].value_counts())
            
            # 可视化
            # self._visualize_results(gdf, final_gdf)
            
            return final_df
            
        except Exception as e:
            print(f"数据处理失败: {str(e)}")
            raise

    def _visualize_results(self, original_gdf, balanced_gdf):
        """可视化空间分布对比"""
        # 增加图形高度以适应图例
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 12))
        
        # 原始分布
        original_gdf.plot(
            ax=ax1, 
            column=self.class_col, 
            markersize=5, 
            legend=True,
            cmap='tab20',
            alpha=0.7,
            legend_kwds={
                'bbox_to_anchor': (1.05, 1),  # 将图例放在图形右侧
                'loc': 'upper left',
                'borderaxespad': 0.,
                'fontsize': 8  # 减小图例字体大小
            }
        )
        ax1.set_title("原始空间分布", pad=20)  # 增加标题和图形的间距
        
        # 平衡后分布
        balanced_gdf.plot(
            ax=ax2, 
            column=self.class_col, 
            markersize=5, 
            legend=True,
            cmap='tab20',
            alpha=0.7,
            legend_kwds={
                'bbox_to_anchor': (1.05, 1),
                'loc': 'upper left',
                'borderaxespad': 0.,
                'fontsize': 8
            }
        )
        ax2.set_title("平衡后空间分布", pad=20)
        
        # 调整子图之间的间距
        plt.subplots_adjust(
            wspace=0.3,  # 增加子图之间的水平间距
            right=0.85   # 为图例留出空间
        )
        
        # 保存图片而不是显示
        plt.savefig('soil_distribution_comparison.png', 
                    bbox_inches='tight',  # 自动调整边界
                    dpi=300,             # 提高分辨率
                    pad_inches=0.5)      # 增加边距
        
        plt.close()  # 关闭图形，释放内存

# 使用示例
if __name__ == "__main__":
    # 初始化优化器
    optimizer = SoilSampleOptimizer(
        target_samples=385,
        min_samples=30,
        spatial_resolution=1000,  # 约1km
        min_cluster_size=5,
        pca_variance=0.95,
        x_col='lon',
        y_col='lat',
        class_col='NEW_TZ',
        feature_cols=['aspect',
       'carbonate', 'channelnetworkbaselevel', 'channelnetworkdistance',
       'clay_minerals', 'contrast', 'convergenceindex', 'correlation', 'dem',
       'dissimilarity', 'dl', 'entropy', 'etp22_3', 'etp22_mean', 'evi',
       'ferrous_minerals', 'hillshade', 'homogeneity', 'lat', 'lon',
       'lsfactor', 'lswi', 'mean', 'mndwi', 'mrrtf', 'mrvbf', 'ndmi', 'ndvi',
       'ndwi', 'night22_', 'pc1', 'pc2', 'plancurvature', 'pre22_3', 'pre22_mean',
       'profilecurvature', 'relativeslopeposition', 'rock_outcrop', 'savi',
       'secondmoment', 'slope', 'slopepostion', 'terrainruggednessindex',
       'tmp22_3', 'tmp22_mean', 'topographicwetnessindex', 'totalcatchmentarea',
       'valleydepth', 'vari', 'variance'],
        env_weight=0.7
    )
    
    # 执行样本平衡
    input_path = r"F:\cache_data\zone_ana\qz\train_data\soil_type_train_point.csv"
    output_path = r"F:\cache_data\zone_ana\qz\train_data\optimized_soil_samples.csv"
    
    optimized_df = optimizer.balance_dataset(input_path, output_path)
    