import numpy as np

from sklearn.gaussian_process import GaussianProcessRegressor

from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel

import matplotlib.pyplot as plt


# 假设你已经有一个包含PH值和对应x, y坐标的数据帧df

# df['x_coords']

# df['y_coords']

# df['ph']

points_coordinates = df[["x_coords", "y_coords"]].values

ph_values = df["ph"].values


kernel = DotProduct() + WhiteKernel()

gpr = GaussianProcessRegressor(kernel=kernel, random_state=0).fit(
    points_coordinates, ph_values
)


# 网格化域

x = np.linspace(min(df["x_coords"]), max(df["x_coords"]), num=500)

y = np.linspace(min(df["y_coords"]), max(df["y_coords"]), num=500)

xv, yv = np.meshgrid(x, y)

grid_points = np.c_[xv.ravel(), yv.ravel()]


grid_predictions = gpr.predict(grid_points)
