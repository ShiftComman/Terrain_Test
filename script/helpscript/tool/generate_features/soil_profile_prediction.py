import os
import numpy as np
from osgeo import gdal
import rasterio
from rasterio.transform import Affine
from rasterio.windows import Window
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from multiprocessing import freeze_support
import json

class SoilProfilePredictor:
    """土壤剖面构型预测类"""
    
    def __init__(self, workspace):
        self.workspace = workspace
        gdal.AllRegister()
        
        # 质地构型编码映射
        self.profile_codes = {
            1: "海绵型",
            2: "夹层型",
            3: "紧实型",
            4: "松散型",
            5: "薄层型",
            6: "上松下紧型",
            7: "上紧下松型"
        }
        
        # 土壤质地与构型的关系规则（使用编码）
        self.texture_profile_rules = {
            1: {  # 壤土
                "主要构型": [1, 2],  # 海绵型、夹层型
                "次要构型": [3]      # 紧实型
            },
            2: {  # 粉(砂)质壤土
                "主要构型": [1, 4],  # 海绵型、松散型
                "次要构型": [2]      # 夹层型
            },
            # 砂质壤土
            3: {
                "主要构型": [4],     # 松散型
                "次要构型": [1]      # 海绵型
            },
            # 粉(砂)质黏壤土
            4: {
                "主要构型": [2, 3],  # 夹层型、紧实型
                "次要构型": [1]      # 海绵型
            },
            # 黏壤土
            5: {
                "主要构型": [1, 2],  # 海绵型、夹层型
                "次要构型": [3]      # 紧实型
            },
            # 砂质黏壤土
            6: {
                "主要构型": [2, 4],  # 夹层型、松散型
                "次要构型": [3]      # 紧实型
            },
            # 壤质黏土
            7: {
                "主要构型": [3],     # 紧实型
                "次要构型": [2]      # 夹层型
            },
            # 黏土
            8: {
                "主要构型": [3],     # 紧实型
                "次要构型": [6]      # 上松下紧型
            },
            # 砂质黏土
            9: {
                "主要构型": [3, 6],  # 紧实型、上松下紧型
                "次要构型": [2]      # 夹层型
            },
            # 重黏土
            10: {
                "主要构型": [3],     # 紧实型
                "次要构型": [6]      # 上松下紧型
            },
            # 粉(砂)质黏土
            11: {
                "主要构型": [3, 2],  # 紧实型、夹层型
                "次要构型": [6]      # 上松下紧型
            },
            # 砂土及壤质砂土
            12: {
                "主要构型": [4],     # 松散型
                "次要构型": [5]      # 薄层型
            }
        }
        
        # 地形位置对构型的影响规则（使用编码）
        self.terrain_rules = {
            6: {  # 坡顶
                "适宜构型": [4, 5], # 松散型、薄层型
                "不适构型": [1, 3] # 海绵型、紧实型
            },
            5: {  # 坡上
                "适宜构型": [3, 5], # 紧实型、薄层型
                "不适构型": [1, 4] # 海绵型、松散型
            },
            4: {  # 坡中
                "适宜构型": [2, 3],    # 夹层型、紧实型
                "不适构型": [4, 5]     # 松散型、薄层型
            },
            3: {  # 坡中平地
                "适宜构型": [1, 6],    # 海绵型、上松下紧型
                "不适构型": [3, 5]     # 紧实型、薄层型
            },
            2: {  # 谷底
                "适宜构型": [1, 2],    # 海绵型、夹层型
                "不适构型": [4, 5]     # 松散型、薄层型
            },
            1: {  # 平地
                "适宜构型": [1, 2],  # 海绵型、夹层型
                "不适构型": [3, 4]  # 紧实型、松散型
            }
        }
    
    def predict_profile(self, texture_raster, position_raster):
        """预测土壤剖面构型"""
        try:
            # 读取栅格信息
            with rasterio.open(texture_raster) as src:
                profile = src.profile
                
            # 准备输出文件
            profile.update(nodata=0)
            output_path = os.path.join(self.workspace, "soil_profile_prediction.tif")
            
            # 分块处理
            chunk_size = 1000
            with rasterio.open(texture_raster) as texture_src, \
                 rasterio.open(position_raster) as position_src, \
                 rasterio.open(output_path, 'w', **profile) as dst:
                
                windows = [Window(col_off, row_off, 
                                min(chunk_size, profile['width'] - col_off),
                                min(chunk_size, profile['height'] - row_off))
                          for row_off in range(0, profile['height'], chunk_size)
                          for col_off in range(0, profile['width'], chunk_size)]
                
                with ProcessPoolExecutor(max_workers=8) as executor:
                    futures = [
                        executor.submit(
                            self.process_chunk,
                            window,
                            texture_src.read(1, window=window),
                            position_src.read(1, window=window),
                            self.texture_profile_rules,
                            self.terrain_rules
                        )
                        for window in windows
                    ]
                    
                    for future in tqdm(as_completed(futures), 
                                     total=len(futures),
                                     desc="预测土壤构型"):
                        window, result = future.result()
                        dst.write(result, 1, window=window)
            
            # 保存编码映射字典
            code_dict = {
                'profile_codes': {str(k): v for k, v in self.profile_codes.items()},
                'description': '土壤剖面构型编码说明'
            }
            dict_path = os.path.join(self.workspace, "soil_profile_codes.json")
            with open(dict_path, 'w', encoding='utf-8') as f:
                json.dump(code_dict, f, ensure_ascii=False, indent=4)
            
            print(f"构型编码说明已保存至: {dict_path}")
            return output_path
            
        except Exception as e:
            raise Exception(f"预测失败: {str(e)}")
    
    @staticmethod
    def process_chunk(window, texture_data, position_data, texture_rules, terrain_rules):
        """处理数据块"""
        try:
            result = np.zeros_like(texture_data)
            
            # 创建有效数据掩膜
            valid_mask = (~np.isnan(texture_data)) & (~np.isnan(position_data)) & \
                        (texture_data > 0) & (position_data > 0)
            
            # 获取有效像素位置
            valid_indices = np.where(valid_mask)
            
            for idx in range(len(valid_indices[0])):
                i, j = valid_indices[0][idx], valid_indices[1][idx]
                
                try:
                    # 转换为整数
                    texture_code = int(round(texture_data[i, j]))
                    position_code = int(round(position_data[i, j]))
                    
                    # 检查编码是否在规则中
                    if texture_code in texture_rules:
                        # 首先尝试使用质地和地形位置的组合
                        if position_code in terrain_rules:
                            texture_profiles = texture_rules[texture_code]["主要构型"]
                            terrain_profiles = terrain_rules[position_code]["适宜构型"]
                            
                            # 找出共同适宜的构型
                            common_profiles = list(set(texture_profiles) & set(terrain_profiles))
                            
                            if common_profiles:
                                result[i, j] = common_profiles[0]
                                continue
                        
                        # 如果没有合适的组合或地形位置无效，使用质地的主要构型
                        result[i, j] = texture_rules[texture_code]["主要构型"][0]
                    else:
                        # 如果质地编码无效，设置为默认值
                        result[i, j] = 1  # 使用最常见的构型作为默认值
                    
                except (ValueError, TypeError) as e:
                    # 如果处理出错，设置为默认值
                    result[i, j] = 1
            
            return window, result
            
        except Exception as e:
            raise Exception(f"处理数据块失败: {str(e)}\n"
                           f"texture_data范围: {texture_data.min()}-{texture_data.max()}\n"
                           f"position_data范围: {position_data.min()}-{position_data.max()}")

if __name__ == '__main__':
    freeze_support()
    
    workspace = r"C:\Users\Runker\Desktop\genarate_feature"
    predictor = SoilProfilePredictor(workspace)
    
    try:
        # 预测土壤构型
        texture_raster = os.path.join(workspace, "soil_texture_prediction.tif")
        position_raster = os.path.join(workspace, "raster_file", "slope_position.tif")
        
        profile_raster = predictor.predict_profile(texture_raster, position_raster)
        print(f"土壤构型预测完成: {profile_raster}")
        
    except Exception as e:
        print(f"处理过程出错: {str(e)}") 