import os
import sys
from pathlib import Path
import json
import logging
from typing import Dict, Tuple, Optional
from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import seaborn as sns

# 日志配置
def setup_logger(log_dir: str) -> logging.Logger:
    """配置并返回日志记录器"""
    logger = logging.getLogger("grade_evaluation")
    if logger.hasHandlers():
        return logger

    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "grade_evaluation.log"

    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger

# 配置类
@dataclass
class EvaluationConfig:
    region: str
    sub_region: str
    area_coefficient: float
    crop_seasons: int
    actual_planting_ratio: float
    area_field: str
    latitude_field: str  # 新增经度列名参数
    longitude_field: str  # 新增纬度列名参数
    yield_dict: Dict[str, int] = None

    @classmethod
    def from_defaults(cls, **kwargs):
        """从默认值和自定义参数创建配置实例"""
        defaults = {
            "region": "西南区",
            "sub_region": "黔桂高原山地林农牧区",
            "area_coefficient": 15.0,
            "crop_seasons": 2,
            "actual_planting_ratio": 0.8,
            "area_field": "project_Area",
            "latitude_field": "latitude",  # 默认经纬度列名
            "longitude_field": "longitude",
            "yield_dict": {
                "一等": 1500, "二等": 1478, "三等": 1320, "四等": 1218,
                "五等": 1116, "六等": 1018, "七等": 996, "八等": 826,
                "九等": 735, "十等": 665
            }
        }
        defaults.update(kwargs)
        return cls(**defaults)

# 隶属度函数
class MembershipFunctions:
    @staticmethod
    def negative_linear(value: np.ndarray, u_min: float, u_max: float, a: float, b: float) -> np.ndarray:
        return np.where(value <= u_min, 1.0,
                        np.where(value >= u_max, 0.0, b - a * value))

    @staticmethod
    def upper_bound(value: np.ndarray, u_min: float, u_max: float, a: float, c: float) -> np.ndarray:
        return np.where(value <= u_min, 0.0,
                        np.where(value >= u_max, 1.0, 1 / (1 + a * (value - c) ** 2)))

    @staticmethod
    def peak(value: np.ndarray, u_min: float, u_max: float, a: float, c: float) -> np.ndarray:
        return np.where((value <= u_min) | (value >= u_max), 0.0, 1 / (1 + a * (value - c) ** 2))

    @staticmethod
    def lower_bound(value: np.ndarray, u_min: float, u_max: float, a: float, c: float) -> np.ndarray:
        return np.where(value <= u_min, 1.0,
                        np.where(value >= u_max, 0.0, 1 / (1 + a * (value - c) ** 2)))

FUNCTION_MAP = {
    "负直线型": MembershipFunctions.negative_linear,
    "戒上型": MembershipFunctions.upper_bound,
    "峰型": MembershipFunctions.peak,
    "戒下型": MembershipFunctions.lower_bound
}

# 耕地质量评价类
class GradeEvaluation:
    def __init__(self, data_path: str, region_json_path: str, config: EvaluationConfig):
        self.data_path = data_path
        self.region_json_path = region_json_path
        self.config = config
        self.data_frame: Optional[pd.DataFrame] = None
        self.dictionary_data: Optional[Dict] = None
        
        # 设置matplotlib字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']  # 优先使用黑体，备选Arial Unicode MS
        plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

    def _check_column_type(self, col_name: str, expected_type, logger: logging.Logger) -> None:
        """检查列的数据类型"""
        actual_type = self.data_frame[col_name].dtype
        if not np.issubdtype(actual_type, expected_type):
            logger.error(f"{col_name} 的数据类型不是 {expected_type}: {actual_type}")
            raise ValueError(f"{col_name} 的数据类型必须是 {expected_type}")

    def validate_data(self, logger: logging.Logger) -> None:
        """验证数据完整性和有效性"""
        logger.info("开始数据验证...")
        gn_value = self.dictionary_data[self.config.region][self.config.sub_region]['隶属度函数']['概念型']
        sz_value = self.dictionary_data[self.config.region][self.config.sub_region]['隶属度函数']['数值型']

        for index_name, valid_values in gn_value.items():
            if index_name not in self.data_frame.columns:
                raise ValueError(f"缺失概念型指标列: {index_name}")
            invalid_values = set(self.data_frame[index_name].unique()) - set(valid_values.keys())
            if invalid_values:
                raise ValueError(f"指标 {index_name} 包含无效值: {invalid_values}")

        for index_name in sz_value.keys():
            if index_name not in self.data_frame.columns:
                raise ValueError(f"缺失数值型指标列: {index_name}")
            self._check_column_type(index_name, np.floating, logger)

        if self.config.area_field not in self.data_frame.columns:
            raise ValueError(f"缺失面积字段: {self.config.area_field}")
        self._check_column_type(self.config.area_field, np.floating, logger)

        # 检查经纬度字段
        for field in [self.config.latitude_field, self.config.longitude_field]:
            if field not in self.data_frame.columns:
                logger.warning(f"缺失地理字段 {field}，空间可视化将不可用")
            else:
                self._check_column_type(field, np.floating, logger)

        logger.info("数据验证完成")

    def load_data(self, logger: logging.Logger) -> None:
        """加载数据"""
        logger.info("开始加载数据...")
        self.data_frame = pd.read_csv(self.data_path)
        logger.info(f"成功加载数据文件: {self.data_path}, 记录数: {len(self.data_frame)}")

        with open(self.region_json_path, 'r', encoding='utf-8') as f:
            self.dictionary_data = json.load(f)
        logger.info(f"成功加载配置文件: {self.region_json_path}")

        self.validate_data(logger)
        logger.info("数据加载完成")

    def calculate_concept_index(self, logger: logging.Logger) -> None:
        """计算概念型指标隶属度"""
        logger.info("开始计算概念型指标隶属度...")
        gn_value = self.dictionary_data[self.config.region][self.config.sub_region]['隶属度函数']['概念型']
        for index_name in gn_value:
            self.data_frame[f'{index_name}_sub'] = self.data_frame[index_name].map(gn_value[index_name])
            logger.info(f"完成 {index_name} 隶属度计算")
        logger.info("概念型指标隶属度计算完成")

    def calculate_numeric_index(self, logger: logging.Logger) -> None:
        """计算数值型指标隶属度"""
        logger.info("开始计算数值型指标隶属度...")
        sz_value = self.dictionary_data[self.config.region][self.config.sub_region]['隶属度函数']['数值型']
        for index_name, params in sz_value.items():
            func = FUNCTION_MAP.get(params["类型"])
            if not func:
                raise ValueError(f"未知的隶属度函数类型: {params['类型']}")
            self.data_frame[f"{index_name}_sub"] = func(
                self.data_frame[index_name].values,
                params.get("最小值", -np.inf),
                params.get("最大值", np.inf),
                params.get("系数a", 0),
                params.get("截距b", 0) or params.get("标准值c", 0)
            )
            logger.info(f"完成 {index_name} 隶属度计算")
        logger.info("数值型指标隶属度计算完成")

    def calculate_comprehensive_index(self, logger: logging.Logger) -> None:
        """计算综合指数"""
        logger.info("开始计算综合指数...")
        index_value = self.dictionary_data[self.config.region][self.config.sub_region]['权重']
        weights = {f"{key}_sub": value for key, value in index_value.items()}
        self.data_frame['综合指数'] = (self.data_frame[list(weights.keys())] * pd.Series(weights)).sum(axis=1)
        logger.info(f"综合指数平均值: {self.data_frame['综合指数'].mean():.4f}")
        logger.info("综合指数计算完成")

    def grade_division(self, logger: logging.Logger) -> None:
        """等级划分"""
        logger.info("开始进行等级划分...")
        level_value = self.dictionary_data[self.config.region][self.config.sub_region]['等级划分']
        
        def calc_level(value: float) -> str:
            for level_dict in level_value:
                if level_dict['最小值'] <= value < level_dict['最大值']:
                    return level_dict['等级']
            return "未定义"

        self.data_frame['综合质量等级'] = self.data_frame['综合指数'].apply(calc_level)
        logger.info("等级划分完成")

    def fit_yield_model(self, scores: np.ndarray, yields: np.ndarray) -> Tuple[float, float, float]:
        """拟合产量模型"""
        def model(x, a, c):
            return max(yields) / (1 + a * (x - c)**2)
        
        try:
            params, _ = curve_fit(model, scores, yields, p0=[0.01, np.mean(scores)], maxfev=1000)
            return params[0], params[1], max(yields)
        except RuntimeError as e:
            raise ValueError(f"拟合失败: {str(e)}")

    def calculate_production_capacity(self, logger: logging.Logger) -> None:
        """计算产能"""
        logger.info("开始计算产能...")
        grouped = self.data_frame.groupby("综合质量等级").agg({
            self.config.area_field: "sum",
            "综合指数": "mean"
        }).reset_index()
        grouped["estimated_yield"] = grouped["综合质量等级"].map(self.config.yield_dict)

        a, c, b = self.fit_yield_model(grouped["综合指数"], grouped["estimated_yield"])
        logger.info(f"拟合系数: a={a:.4f}, c={c:.4f}, b={b:.4f}")

        max_score, min_score = grouped["综合指数"].max(), grouped["综合指数"].min()
        yield_max = b / (1 + a * (max_score - c)**2)
        yield_min = b / (1 + a * (min_score - c)**2)

        def calc_yield(score: float) -> float:
            if score > max_score:
                return yield_max
            if score < min_score:
                return yield_min
            return b / (1 + a * (score - c)**2)

        self.data_frame["predicted_yield"] = self.data_frame["综合指数"].apply(calc_yield)
        self.data_frame["total_yield"] = (self.data_frame["predicted_yield"] * 
                                         self.data_frame[self.config.area_field] * 
                                         self.config.area_coefficient)
        self.data_frame["seasonal_yield"] = self.data_frame["predicted_yield"] / self.config.crop_seasons

        correction_factor = self.config.actual_planting_ratio / self.config.crop_seasons
        self.data_frame["annual_production"] = self.data_frame["total_yield"] * correction_factor

        logger.info(f"校正系数: {correction_factor:.4f}")
        logger.info(f"年度粮食生产潜力总和: {self.data_frame['annual_production'].sum():.2f}")
        logger.info("产能计算完成")

    def plot_comprehensive_index_distribution(self, output_dir: str, logger: logging.Logger) -> None:
        """绘制综合指数分布直方图"""
        plt.figure(figsize=(10, 6))
        sns.histplot(self.data_frame['综合指数'], kde=True, bins=30, color='skyblue')
        plt.title('综合指数分布', fontsize=14)  # 改为中文标题
        plt.xlabel('综合指数', fontsize=12)
        plt.ylabel('频次', fontsize=12)
        
        output_path = os.path.join(output_dir, 'comprehensive_index_distribution.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"综合指数分布图已保存至: {output_path}")

    def plot_grade_area_pie(self, output_dir: str, logger: logging.Logger) -> None:
        """绘制各等级面积占比饼图"""
        grade_areas = self.data_frame.groupby('综合质量等级')[self.config.area_field].sum()
        plt.figure(figsize=(8, 8))
        plt.pie(grade_areas, labels=grade_areas.index, autopct='%1.1f%%', startangle=90, colors=plt.cm.Paired.colors)
        plt.title('各等级面积占比', fontsize=14)  # 改为中文标题
        
        output_path = os.path.join(output_dir, 'grade_area_pie.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"各等级面积占比图已保存至: {output_path}")

    def plot_index_vs_yield_scatter(self, output_dir: str, logger: logging.Logger) -> None:
        """绘制综合指数与预测产能散点图"""
        plt.figure(figsize=(10, 6))
        plt.scatter(self.data_frame['综合指数'], self.data_frame['predicted_yield'], alpha=0.5, c='green', label='数据点')
        
        scores = np.linspace(self.data_frame['综合指数'].min(), self.data_frame['综合指数'].max(), 100)
        grouped = self.data_frame.groupby('综合质量等级').agg({'综合指数': 'mean', 'predicted_yield': 'mean'}).reset_index()
        a, c, b = self.fit_yield_model(grouped['综合指数'], grouped['predicted_yield'])
        fitted_yields = b / (1 + a * (scores - c)**2)
        plt.plot(scores, fitted_yields, 'r-', label='拟合曲线')
        
        plt.title('综合指数与预测产量关系', fontsize=14)
        plt.xlabel('综合指数', fontsize=12)
        plt.ylabel('预测产量 (kg/亩)', fontsize=12)
        plt.legend()
        
        output_path = os.path.join(output_dir, 'index_vs_yield_scatter.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"综合指数与预测产能散点图已保存至: {output_path}")

    def plot_membership_boxplot(self, output_dir: str, logger: logging.Logger) -> None:
        """绘制各指标隶属度箱线图"""
        sub_cols = [col for col in self.data_frame.columns if col.endswith('_sub')]
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=self.data_frame[sub_cols], palette='Set2')
        plt.title('Distribution of Membership of Each Index', fontsize=14)
        plt.xlabel('Index', fontsize=12)
        plt.ylabel('Membership', fontsize=12)
        plt.xticks(rotation=45)
        
        output_path = os.path.join(output_dir, 'membership_boxplot.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"各指标隶属度箱线图已保存至: {output_path}")

    
    def plot_spatial_production_map(self, output_dir: str, logger: logging.Logger) -> None:
        """绘制优化后的年度粮食生产潜力空间分布图，使用六边形热力图"""
        if (self.config.latitude_field not in self.data_frame.columns or 
            self.config.longitude_field not in self.data_frame.columns):
            logger.warning("数据中缺少经纬度信息，跳过空间分布图绘制")
            return

        plt.figure(figsize=(12, 8))
        hb = plt.hexbin(self.data_frame[self.config.longitude_field], 
                        self.data_frame[self.config.latitude_field], 
                        C=self.data_frame['annual_production'], 
                        gridsize=50,  # 控制六边形的密度
                        cmap='viridis',  # 使用更清晰的颜色映射
                        mincnt=1,  # 最小计数，过滤零值点
                        alpha=0.7)  # 调整透明度
        plt.colorbar(hb, label='年度粮食生产潜力 (kg)')
        plt.title('年度粮食生产潜力空间分布', fontsize=14)
        plt.xlabel('经度', fontsize=12)
        plt.ylabel('纬度', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.3)  # 添加网格线

        # 调整坐标范围（根据数据动态调整）
        plt.xlim(self.data_frame[self.config.longitude_field].min(), 
                self.data_frame[self.config.longitude_field].max())
        plt.ylim(self.data_frame[self.config.latitude_field].min(), 
                self.data_frame[self.config.latitude_field].max())
        
        output_path = os.path.join(output_dir, 'spatial_production_map.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"优化后的空间分布图已保存至: {output_path}")

    def save_results(self, output_path: str, logger: logging.Logger) -> None:
        """保存结果并生成可视化"""
        logger.info("开始保存结果...")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        for col in ['QSDWDM', 'ZLDWDM']:
            if col in self.data_frame.columns:
                self.data_frame[col] = self.data_frame[col].astype(str)
        self.data_frame.to_excel(output_path, index=False)
        logger.info(f"结果已保存至: {output_path}")

        vis_dir = os.path.join(os.path.dirname(output_path), 'visualizations')
        os.makedirs(vis_dir, exist_ok=True)
        self.plot_comprehensive_index_distribution(vis_dir, logger)
        self.plot_grade_area_pie(vis_dir, logger)
        self.plot_index_vs_yield_scatter(vis_dir, logger)
        self.plot_membership_boxplot(vis_dir, logger)
        self.plot_spatial_production_map(vis_dir, logger)

# 主函数
def main(config_path: str = "config.json"):
    """执行耕地质量评价流程"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config_data = json.load(f)

    config = EvaluationConfig.from_defaults(**config_data["evaluation"])
    logger = setup_logger(config_data["log_dir"])
    evaluator = GradeEvaluation(config_data["data_path"], config_data["region_json_path"], config)

    steps = [
        evaluator.load_data,
        evaluator.calculate_concept_index,
        evaluator.calculate_numeric_index,
        evaluator.calculate_comprehensive_index,
        evaluator.grade_division,
        evaluator.calculate_production_capacity
    ]

    try:
        for step in steps:
            step(logger)
        evaluator.save_results(config_data["output_path"], logger)
    except Exception as e:
        logger.error(f"评价过程发生错误: {str(e)}")
        raise

if __name__ == "__main__":
    # 示例配置文件
    default_config = {
        "data_path": r"G:\soil_property_result\qzs\grade_evaluation\table\grade_evaluation_have_index.csv",
        "region_json_path": r"D:\worker_code\Terrain_Test\data\grade_evaluation\region_all.json",
        "output_path": r"G:\soil_property_result\qzs\grade_evaluation\result\grade_evaluation_result_have_channeng_auto.xlsx",
        "log_dir": r"G:\soil_property_result\qzs\grade_evaluation\logs",
        "evaluation": {
            "latitude_field": "Centroid_Y",  
            "longitude_field": "Centroid_X"  
        }
    }
    # 删除已有的log文件
    log_dir = Path(default_config["log_dir"])
    if log_dir.exists():
        for file in log_dir.glob("*.log"):
            file.unlink()
            
    with open("config.json", "w", encoding='utf-8') as f:
        json.dump(default_config, f, ensure_ascii=False, indent=2)
    main("config.json")