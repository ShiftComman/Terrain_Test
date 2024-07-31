import numpy as np
from scipy.spatial.distance import pdist, squareform

def create_soil_graph(X, y, max_distance=1000):
    edge_index = []
    
    # 计算样本之间的地理距离
    coordinates = X[['Centroid_X', 'Centroid_Y']].values
    distances = squareform(pdist(coordinates))
    
    for i in range(len(X)):
        for j in range(i+1, len(X)):
            connect = False
            
            # 条件1：地理距离在阈值内
            if distances[i][j] <= max_distance:
                connect = True
            
            # 条件2-4：相同TL、YL、TS
            elif (y['TL'].iloc[i] == y['TL'].iloc[j] or 
                  y['YL'].iloc[i] == y['YL'].iloc[j] or 
                  y['TS'].iloc[i] == y['TS'].iloc[j]):
                connect = True
            
            # 条件5：相似的环境因素（这里以DLMC和母质为例）
            elif X['DLMC'].iloc[i] == X['DLMC'].iloc[j] and X['母质'].iloc[i] == X['母质'].iloc[j]:
                connect = True
            
            if connect:
                edge_index.append([i, j])
                edge_index.append([j, i])  # 无向图
    
    return np.array(edge_index).T

# 使用示例
# edge_index = create_soil_graph(X_train, y_train)



# 不考虑距离，只考虑TL、YL、TS
# import numpy as np

# def create_soil_graph(X, y, similarity_threshold=0.7):
#     edge_index = []
    
#     for i in range(len(X)):
#         for j in range(i+1, len(X)):
#             connect = False
            
#             # 条件1-4：相同TL、YL、TS、TZ
#             if (y['TL'].iloc[i] == y['TL'].iloc[j] or 
#                 y['YL'].iloc[i] == y['YL'].iloc[j] or 
#                 y['TS'].iloc[i] == y['TS'].iloc[j] or 
#                 y['TZ'].iloc[i] == y['TZ'].iloc[j]):
#                 connect = True
            
#             # 条件5：环境因素的相似性
#             else:
#                 # 计算环境因素的相似性
#                 env_factors = ['DLMC', '母质', 'DEM_MEAN', 'Slope_MEAN', 'AspectMEAN', 'PRE2022_mean_MEAN', 'TMP2022_mean_MEAN']
#                 similarity = calculate_similarity(X.iloc[i][env_factors], X.iloc[j][env_factors])
                
#                 if similarity >= similarity_threshold:
#                     connect = True
            
#             if connect:
#                 edge_index.append([i, j])
#                 edge_index.append([j, i])  # 无向图
    
#     return np.array(edge_index).T

# def calculate_similarity(sample1, sample2):
#     # 这里使用一个简单的相似性计算方法
#     # 对于分类变量，检查是否相同
#     # 对于数值变量，计算归一化后的欧氏距离
#     similarity = 0
#     total_features = len(sample1)
    
#     for feature, value1 in sample1.items():
#         value2 = sample2[feature]
#         if feature in ['DLMC', '母质']:
#             similarity += 1 if value1 == value2 else 0
#         else:
#             # 假设数值特征已经过标准化
#             similarity += 1 - abs(value1 - value2)
    
#     return similarity / total_features

# # 使用示例
# # edge_index = create_soil_graph(X_train, y_train)


import numpy as np
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial.distance import pdist, squareform

def gower_distance(X):
    """计算Gower距离"""
    num_features = X.shape[1]
    ranges = np.ptp(X, axis=0)
    ranges[ranges == 0] = 1  # 避免除以零

    # 对于分类变量，我们使用简单的匹配/不匹配度量
    categorical_features = ['DLMC', '母质', 'SlopeClass_MAJORITY']
    cat_indices = [X.columns.get_loc(col) for col in categorical_features if col in X.columns]
    
    def gower_dist(xi, xj):
        dists = np.abs(xi - xj) / ranges
        dists[cat_indices] = (xi[cat_indices] != xj[cat_indices]).astype(float)
        return np.sum(dists) / num_features

    return pdist(X, metric=gower_dist)

def create_soil_graph(X, y, similarity_threshold=0.2):
    edge_index = []
    
    # 移除不需要的列
    X_filtered = X.drop(['Centroid_X', 'Centroid_Y'], axis=1, errors='ignore')
    
    # 计算Gower距离
    distances = gower_distance(X_filtered)
    similarities = 1 - squareform(distances)
    
    for i in range(len(X)):
        for j in range(i+1, len(X)):
            connect = False
            
            # 条件1-4：相同TL、YL、TS、TZ
            if (y['TL'].iloc[i] == y['TL'].iloc[j] or 
                y['YL'].iloc[i] == y['YL'].iloc[j] or 
                y['TS'].iloc[i] == y['TS'].iloc[j] or 
                y['TZ'].iloc[i] == y['TZ'].iloc[j]):
                connect = True
            
            # 条件5：环境因素的相似性
            elif similarities[i, j] >= similarity_threshold:
                connect = True
            
            if connect:
                edge_index.append([i, j])
                edge_index.append([j, i])  # 无向图
    
    return np.array(edge_index).T

# 使用示例
# edge_index = create_soil_graph(X_train, y_train, similarity_threshold=0.2)