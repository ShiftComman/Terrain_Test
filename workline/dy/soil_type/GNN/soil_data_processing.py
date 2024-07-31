import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

def preprocess_soil_data(df):
    df = df.copy()
    # 分离特征和标签
    label_columns = ['TL', 'YL', 'TS', 'TZ']
    feature_columns = [col for col in df.columns if col not in label_columns + ['Centroid_X', 'Centroid_Y']]
    
    X = df[feature_columns + ['Centroid_X', 'Centroid_Y']]
    y = df[label_columns]
    
    # 处理类别型特征
    categorical_features = ['DLMC', '母质', 'SlopeClass_MAJORITY']
    label_encoders = {}
    for feature in categorical_features:
        le = LabelEncoder()
        # X[feature] = le.fit_transform(X[feature])
        X.loc[:, feature] = le.fit_transform(X[feature])
        label_encoders[feature] = le
    
    # 确保所有特征都是数值型
    for col in X.columns:
        if X[col].dtype == 'object':
            try:
                X[col] = X[col].astype(float)
            except ValueError:
                print(f"警告: 无法将列 '{col}' 转换为数值型。将使用 LabelEncoder。")
                le = LabelEncoder()
                # X[col] = le.fit_transform(X[col].astype(str))
                X.loc[:, col] = le.fit_transform(X[col].astype(str))

                label_encoders[col] = le
    
    # 标准化数值型特征
    scaler = StandardScaler()
    X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    
    # 编码标签
    for col in label_columns:
        le = LabelEncoder()
        # y[col] = le.fit_transform(y[col].astype(str))
        y.loc[:, col] = le.fit_transform(y[col].astype(str))
        label_encoders[col] = le
    
    # 分割训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test, label_encoders

# 使用示例
# X_train, X_test, y_train, y_test, label_encoders = preprocess_soil_data(df)