import os
import pandas as pd
import json
from datetime import datetime
from dotenv import load_dotenv
import logging
from tqdm import tqdm
from tqdm.auto import tqdm  # 自动选择合适的进度条显示方式

# 配置文件
class Config:
    # 数据列配置
    load_dotenv()
    BASE_COLUMNS = os.getenv('BASE_COLUMNS').split(',')
    LDTJ_COLUMNS = os.getenv('LDTJ_COLUMNS').split(',')
    PM_COLUMNS = os.getenv('PM_COLUMNS').split(',')
    CH_COLUMNS = os.getenv('CH_COLUMNS').split(',')
    NUMERIC_COLUMNS = os.getenv('NUMERIC_COLUMNS').split(',')
    BASE_DATA_PATH = os.getenv('BASE_DATA_PATH')
    CH_DATA_PATH = os.getenv('CH_DATA_PATH')
    SAVE_PATH = os.getenv('SAVE_PATH')
    AREA_DICT_PATH = os.getenv('AREA_DICT_PATH')
    DTYPE_PATH = os.getenv('DTYPE_PATH')
    
    
class DataProcessor:
    def __init__(self, province_name, province_code):
        self.province_name = province_name
        self.province_code = province_code
        self.save_path = rf'{Config.SAVE_PATH}\{self.province_code}_{self.province_name}_result_{self.get_today_date()}.csv'
        self.setup_logging()
        
    def setup_logging(self):
        # 日志
        log_dir = Config.SAVE_PATH
        os.makedirs(log_dir, exist_ok=True)
        
        # 设置日志文件路径
        log_file = os.path.join(log_dir, f'data_export_{self.get_today_date()}.log')
        
        # 配置根日志记录器
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler()  # 同时输出到控制台
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def load_excel_dtype(self):
        """加载excel数据类型"""
        with open(Config.DTYPE_PATH, 'r', encoding='utf-8') as f:
            dtype_dict = json.load(f)
        return dtype_dict
    def get_today_date(self):
        """获取日期"""
        return datetime.now().strftime("%Y%m%d")
    
    def find_file_with_string(self, path, string):
        """获取指定路径下包含指定字符串的文件路径"""
        try:
            result_files = []
            for root, _, files in os.walk(path):
                # 优先添加CSV文件
                csv_files = [f for f in files if string in f and f.endswith('.csv')]
                result_files.extend([os.path.join(root, f) for f in csv_files])
                
                # 添加不重名的XLSX文件
                csv_names = {os.path.splitext(os.path.basename(f))[0] for f in result_files}
                xlsx_files = [f for f in files if string in f and f.endswith('.xlsx') 
                            and os.path.splitext(f)[0] not in csv_names]
                result_files.extend([os.path.join(root, f) for f in xlsx_files])
            
            if not result_files:
                raise FileNotFoundError(f"在{path}路径下未找到包含{string}的文件")
            
            return result_files
            
        except Exception as e:
            self.logger.error(f"查找文件时出错: {str(e)}")
            raise
    
    def merge_excel_files(self, file_list, use_columns, desc="合并文件"):
        """合并多个Excel/CSV文件"""
        try:
            df_list = []
            chunk_size = 100000
            
            # 添加数据预处理函数
            def preprocess_numeric_columns(df):
                numeric_cols = [col for col in df.columns if col in self.load_excel_dtype() 
                              and self.load_excel_dtype()[col] in ['float32', 'float64']]
                for col in numeric_cols:
                    # 将特殊字符替换为NaN
                    df[col] = df[col].replace(['/', '-', '#', '', ' '], pd.NA)
                    # 转换为数值类型
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                return df
            
            for file in tqdm(file_list, desc=desc):
                try:
                    if file.endswith('.csv'):
                        chunks = pd.read_csv(file, usecols=use_columns, chunksize=chunk_size,dtype=self.load_excel_dtype())
                        for chunk in tqdm(chunks, desc=f"读取{os.path.basename(file)}", leave=False):
                            chunk = preprocess_numeric_columns(chunk)
                            df_list.append(chunk)
                    else:
                        # 首先读取文件的所有列名
                        actual_columns = pd.read_excel(file, nrows=0).columns.tolist()
                        missing_columns = set(use_columns) - set(actual_columns)
                        if missing_columns:
                            self.logger.error(f"""
                            文件 {file} 缺少以下列:
                            缺失的列: {list(missing_columns)}
                            实际的列: {actual_columns}
                            期望的列: {use_columns}
                            """)
                            continue  # 跳过这个文件而不是直接失败
                            
                        df = pd.read_excel(file, usecols=use_columns, dtype=self.load_excel_dtype())
                        df = preprocess_numeric_columns(df)
                        df_list.append(df)
                except Exception as e:
                    self.logger.error(f"读取{os.path.basename(file)}时出错: {str(e)}")
            return pd.concat(df_list, ignore_index=True)
            
        except Exception as e:
            self.logger.error(f"合并文件时出错: {str(e)}")
            raise
    
    def process_data(self):
        """处理数据的主函数"""
        try:
            # 设置路径
            base_path = rf'{Config.BASE_DATA_PATH}\{self.province_name}'
            ch_data_path = rf'{Config.CH_DATA_PATH}\{self.province_name}'
            
            # 获取文件列表
            self.logger.info("开始获取文件列表...")
            with tqdm(total=4, desc="获取数据文件") as pbar:
                file_lists = {}
                # 获取base文件
                file_lists['base'] = self.find_file_with_string(base_path, 'base_info')
                pbar.update(1)
                
                # 获取ldtj文件
                file_lists['ldtj'] = self.find_file_with_string(base_path, 'ldtj_info') 
                pbar.update(1)
                
                # 获取pm文件
                file_lists['pm'] = self.find_file_with_string(base_path, 'pm_info')
                pbar.update(1)
                
                # 获取ch文件
                file_lists['ch'] = self.find_file_with_string(ch_data_path, 'all_info')
                pbar.update(1)
            
            # 读取并合并数据，显示每个数据集的处理进度
            self.logger.info("开始读取和合并数据...")
            dfs = {}
            # 定义列映射关系
            column_mapping = {
                'base': 'BASE_COLUMNS',
                'ldtj': 'LDTJ_COLUMNS',
                'pm': 'PM_COLUMNS',
                'ch': 'CH_COLUMNS'  
            }
            
            for key, files in tqdm(file_lists.items(), desc="处理数据集"):
                dfs[key] = self.merge_excel_files(
                    files, 
                    getattr(Config, column_mapping[key]),  # 使用映射获取正确的列名配置
                    desc=f"合并{key}数据"
                )
            
            # 优化合并策略，添加进度显示
            self.logger.info("开始合并所有数据集...")
            with tqdm(total=3, desc="合并数据集") as pbar:
                df_final = dfs['base'].merge(
                    dfs['ldtj'], on='ydbh', how='left'
                )
                pbar.update(1)
                
                df_final = df_final.merge(
                    dfs['pm'], on='ydbh', how='left'
                )
                pbar.update(1)
                
                df_final = df_final.merge(
                    dfs['ch'], on='ydbh', how='left'
                )
                pbar.update(1)
            
            return self.clean_data(df_final)
            
        except Exception as e:
            self.logger.error(f"处理数据时出错: {str(e)}")
            raise
    
    def clean_data(self, df):
        """清理和转换数据"""
        try:
            self.logger.info("开始清理数据...")
            
            # 转换数值列
            with tqdm(total=len(Config.NUMERIC_COLUMNS), desc="处理数值列") as pbar:
                for col in Config.NUMERIC_COLUMNS:
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0001)
                    pbar.update(1)
            
            # 添加T前缀
            df['ydbht'] = 'T' + df['ydbh'].astype('str')
            df['yypbht'] = 'T' + df['yypbh'].astype('str')
            
            # 转换数据类型
            int_cols = df.select_dtypes(include=['int32', 'int64']).columns
            float_cols = df.select_dtypes(include=['float32', 'float64']).columns
            string_cols = df.select_dtypes(include=['object']).columns
            
            with tqdm(total=len(int_cols) + len(float_cols) + len(string_cols), desc="转换数据类型") as pbar:
                # 整数列保持整数类型
                for col in int_cols:
                    df[col] = df[col].astype('int64').fillna(0)
                    pbar.update(1)
                
                # 浮点数列转为float32
                for col in float_cols:
                    df[col] = df[col].astype('float32').fillna(0.0001)
                    pbar.update(1)
                
                # 字符串列转换
                for col in string_cols:
                    df[col] = df[col].astype('string').fillna('')
                    pbar.update(1)
            
            return df
            
        except Exception as e:
            self.logger.error(f"清理数据时出错: {str(e)}")
            raise
    def save_data(self, df):
        """保存数据"""
        try:
            # 确保保存目录存在
            os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
            df.to_csv(self.save_path, index=False, encoding='utf-8-sig')
            print(f"数据已保存到: {self.save_path}")
        except Exception as e:
            self.logger.error(f"保存数据时出错: {str(e)}")
            raise
if __name__ == "__main__":
    # 获取省份列表名
    area_dict_path = r'D:\worker_code\spiderdir\collection_spb\data\all_region_codes.json'
    area_dict = json.load(open(area_dict_path, 'r', encoding='utf-8'))
    area_name_list = [[_, area_dict[_]['code']] for _ in area_dict.keys()]
    
    for area_name in area_name_list:
        print(area_name)
        # 设置省份名称
        province_name = area_name[0]
        province_code = area_name[1]
        # 创建数据处理器实例
        processor = DataProcessor(province_name, province_code)
        # 如果文件存在，则跳过
        if os.path.exists(processor.save_path):
            continue
        # 处理数据
        df_result = processor.process_data()
        
        # 保存结果
        processor.save_data(df_result)
        
