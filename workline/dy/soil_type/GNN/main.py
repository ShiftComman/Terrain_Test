import pandas as pd
import torch

# 假设您已经定义了上述所有函数和类

# 加载数据
df = pd.read_csv('train_polygon.csv')

# 预处理数据
X_train, X_test, y_train, y_test, label_encoders = preprocess_soil_data(df)

# 构建图结构
edge_index_train = create_soil_graph(X_train, y_train)
edge_index_test = create_soil_graph(X_test, y_test)

# 初始化模型
in_channels = X_train.shape[1]
hidden_channels = 64
num_classes = {
    'TL': len(y_train['TL'].unique()),
    'YL': len(y_train['YL'].unique()),
    'TS': len(y_train['TS'].unique()),
    'TZ': len(y_train['TZ'].unique())
}
model = MultiTaskGATSoilClassifier(in_channels, hidden_channels, num_classes)

# 训练模型
model = train_multitask_gat(model, X_train, y_train, edge_index_train, epochs=200)

# 评估模型
evaluate_multitask_gat(model, X_test, y_test, edge_index_test, label_encoders)