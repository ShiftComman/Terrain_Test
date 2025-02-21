import geopandas as gpd
import numpy as np
from shapely.geometry import Polygon, MultiPolygon
from shapely import oriented_envelope
import logging
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def calculate_min_width(geometry):
    """
    计算单个多边形的最小宽度
    使用shapely的oriented_envelope直接计算最小外接矩形
    """
    try:
        # 如果是MultiPolygon，取面积最大的那个
        if isinstance(geometry, MultiPolygon):
            geometry = max(geometry.geoms, key=lambda x: x.area)
        
        # 获取最小旋转矩形
        min_rect = oriented_envelope(geometry)
        
        # 获取矩形的坐标
        coords = np.array(min_rect.exterior.coords)[:-1]  # 去掉最后一个重复点
        
        # 计算四条边的长度
        edges = np.diff(coords, axis=0, append=[coords[0]])
        lengths = np.sqrt(np.sum(edges**2, axis=1))
        
        # 返回较短的边长
        return min(lengths)
        
    except Exception as e:
        logger.warning(f"计算最小宽度时出错: {str(e)}")
        return None

def process_shapefile(input_shp, output_shp=None):
    """
    处理shapefile，计算每个多边形的最小宽度
    """
    logger.info(f"开始处理文件: {input_shp}")
    
    # 读取shapefile
    gdf = gpd.read_file(input_shp)
    # 转换为单部件
    gdf = gdf.explode(index_parts=True).reset_index(drop=True)
    total_polygons = len(gdf)
    logger.info(f"总多边形数量: {total_polygons}")
    
    # 计算每个多边形的最小宽度
    logger.info("计算最小宽度...")
    gdf['min_width'] = [calculate_min_width(geom) for geom in tqdm(gdf.geometry)]
    
    # 计算面积
    gdf['area'] = gdf.geometry.area
    
    # 保存到输入路径
    logger.info(f"保存结果到: {output_shp}")
    gdf.to_file(output_shp, encoding='utf-8')
    # 输出统计信息
    logger.info("\n统计信息:")
    logger.info(f"最小宽度: {gdf['min_width'].min():.2f}")
    logger.info(f"最大宽度: {gdf['min_width'].max():.2f}")
    logger.info(f"平均宽度: {gdf['min_width'].mean():.2f}")
    logger.info(f"总面积: {gdf['area'].sum():.2f}")

if __name__ == "__main__":
    
    input_shp = r"F:\cache_data\shp_file\qz\ele_qz\qz_merge_data_single.shp"
    output_shp = r"F:\cache_data\shp_file\qz\ele_qz\qz_merge_data_single_width.shp"
    process_shapefile(input_shp, output_shp)