import ee
import geemap
import os
import math
import geopandas as gpd
from typing import List, Optional, Dict, Any
import logging
import backoff
from pathlib import Path
import aiohttp
import asyncio
from shapely.geometry import box, Polygon, MultiPolygon
import json

# 常量定义
DEFAULT_MAX_SIZE = 50000000  # 50MB
DOWNLOAD_TIMEOUT = 600  # 10分钟
CHUNK_SIZE = 1024 * 1024  # 1MB

class GEEDownloadError(Exception):
    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)

@backoff.on_exception(backoff.expo, Exception, max_tries=5)
def authenticate_gee(logger: logging.Logger) -> None:
    logger.info("尝试验证Google Earth Engine")
    try:
        ee.Initialize()
        logger.info("验证成功")
    except Exception as e:
        error_msg = f"验证失败：{str(e)}"
        logger.error(error_msg)
        raise GEEDownloadError(error_msg) from e

class GEEDatasetConfig:
    """简化的GEE数据集配置类"""
    def __init__(self, collection_name: str, bands: List[str], scale: int = 30):
        self.collection_name = collection_name
        self.bands = bands
        self.scale = scale
        self.crs = 'EPSG:4326'  # 使用默认值
        self.format = 'GEO_TIFF'  # 使用默认值
        self.max_size = DEFAULT_MAX_SIZE  # 使用默认值

    @classmethod
    def srtm_dem(cls):
        return cls(
            collection_name='USGS/SRTMGL1_003',
            bands=['elevation'],
            scale=30,
        )

def estimate_image_size(geometry: Polygon, scale: int = 30) -> int:
    """估算DEM影像大小（字节）"""
    # 获取区域面积（平方米）
    bounds = geometry.bounds
    width = abs(bounds[2] - bounds[0]) * 111319.49079327358  # 经度转米
    height = abs(bounds[3] - bounds[1]) * 111319.49079327358  # 纬度转米
    area = width * height
    
    # 计算像素数量（DEM只有一个波段）
    pixels = area / (scale * scale)
    # 估算文件大小（每个像素4字节，增加50%安全余量）
    estimated_size = pixels * 4 * 1.5
    return int(estimated_size)

def adaptive_split_geometry(geometry: Polygon, max_size: int, scale: int, logger: logging.Logger) -> List[Polygon]:
    """自适应分割几何体的边界框"""
    # 估算当前区域大小
    estimated_size = estimate_image_size(geometry, scale)
    
    if estimated_size <= max_size:
        return [geometry]
    
    # 计算需要的分割因子（增加安全系数）
    split_factor = math.ceil(math.sqrt(estimated_size / (max_size * 0.8))) + 1
    logger.info(f"区域预计大小: {estimated_size/1024/1024:.2f}MB, 分割因子: {split_factor}")
    
    bounds = geometry.bounds
    dx = (bounds[2] - bounds[0]) / split_factor
    dy = (bounds[3] - bounds[1]) / split_factor
    
    sub_geometries = []
    for i in range(split_factor):
        for j in range(split_factor):
            minx = bounds[0] + i * dx
            miny = bounds[1] + j * dy
            maxx = bounds[0] + (i + 1) * dx
            maxy = bounds[1] + (j + 1) * dy
            sub_box = box(minx, miny, maxx, maxy)
            sub_geometries.append(sub_box)
    
    return sub_geometries

@backoff.on_exception(backoff.expo, 
                     (aiohttp.ClientError, asyncio.TimeoutError, GEEDownloadError),
                     max_tries=5,
                     max_time=300)  # 最多重试5次，总时间不超过5分钟
async def download_file(url: str, output_path: str, logger: logging.Logger) -> None:
    """异步下载文件，增加重试机制"""
    try:
        # 增加连接超时和总超时时间
        timeout = aiohttp.ClientTimeout(
            total=DOWNLOAD_TIMEOUT,
            connect=60,  # 连接超时时间设为60秒
            sock_connect=60,  # socket连接超时
            sock_read=60  # socket读取超时
        )
        
        connector = aiohttp.TCPConnector(
            ssl=False,  # 禁用SSL验证
            force_close=True,  # 强制关闭连接
            enable_cleanup_closed=True,  # 清理关闭的连接
            limit=1  # 限制并发连接数
        )
        
        async with aiohttp.ClientSession(
            timeout=timeout,
            connector=connector,
            trust_env=True  # 信任环境变量中的代理设置
        ) as session:
            async with session.get(url) as response:
                if response.status != 200:
                    error_msg = f"下载失败，HTTP状态码: {response.status}"
                    logger.error(error_msg)
                    raise GEEDownloadError(error_msg)

                output_dir = os.path.dirname(output_path)
                os.makedirs(output_dir, exist_ok=True)

                with open(output_path, 'wb') as f:
                    async for chunk in response.content.iter_chunked(CHUNK_SIZE):
                        f.write(chunk)

        logger.info(f"文件已下载到: {output_path}")
    except asyncio.TimeoutError as e:
        error_msg = f"下载超时: {str(e)}"
        logger.error(error_msg)
        raise GEEDownloadError(error_msg) from e
    except aiohttp.ClientError as e:
        error_msg = f"网络错误: {str(e)}"
        logger.error(error_msg)
        raise GEEDownloadError(error_msg) from e
    except Exception as e:
        error_msg = f"下载文件时出错: {str(e)}"
        logger.error(error_msg)
        raise GEEDownloadError(error_msg) from e
    finally:
        if 'session' in locals():
            await session.close()

def get_download_url(image: ee.Image, region: Dict[str, Any], config: GEEDatasetConfig) -> str:
    """获取下载URL，使用边界框区域"""
    try:
        url = image.getDownloadURL({
            'region': region,
            'scale': config.scale,
            'crs': config.crs,
            'format': config.format,
            'bands': config.bands
        })
        return url
    except Exception as e:
        raise GEEDownloadError(f"获取下载URL时出错: {str(e)}") from e

def download_gee_data(region: gpd.GeoDataFrame,
                     dataset_config: GEEDatasetConfig,
                     output_folder: str,
                     logger: logging.Logger) -> None:
    """下载函数，增加错误处理"""
    try:
        os.makedirs(output_folder, exist_ok=True)
        image = ee.Image(dataset_config.collection_name)
        
        for idx, row in region.iterrows():
            try:
                geom = row.geometry
                max_pixels = int(50000000 / 4)
                sub_geometries = adaptive_split_geometry(
                    geometry=geom,
                    max_size=max_pixels,
                    scale=dataset_config.scale,
                    logger=logger
                )
                
                logger.info(f"区域 {idx} 被分割为 {len(sub_geometries)} 个子区域")
                
                for sub_idx, sub_geom in enumerate(sub_geometries):
                    try:
                        output_path = os.path.join(output_folder, f"dem_part_{idx}_{sub_idx}.tif")
                        
                        if os.path.exists(output_path):
                            logger.info(f"文件已存在，跳过: {output_path}")
                            continue
                        
                        coords = [[[x, y] for x, y in zip(*sub_geom.exterior.coords.xy)]]
                        ee_geometry = ee.Geometry.Polygon(coords)
                        region_dict = ee_geometry.getInfo()
                        
                        url = get_download_url(image, region_dict, dataset_config)
                        asyncio.run(download_file(url, output_path, logger))
                        
                    except Exception as e:
                        logger.error(f"处理子区域 {sub_idx} 时出错: {str(e)}")
                        continue  # 继续处理下一个子区域
                        
            except Exception as e:
                logger.error(f"处理区域 {idx} 时出错: {str(e)}")
                continue  # 继续处理下一个区域

    except Exception as e:
        raise GEEDownloadError(f"下载GEE数据时出错: {str(e)}") from e

def download_dem(region_path: str, output_folder: str, log_file: Optional[str] = None) -> None:
    """简化的主函数"""
    # 配置日志
    log_dir = Path(log_file).parent if log_file else None
    if log_dir:
        log_dir.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        encoding='utf-8'
    )
    
    logger = logging.getLogger(__name__)
    logger.info("开始下载DEM数据")
    
    try:
        authenticate_gee(logger)
        region = gpd.read_file(region_path)
        dem_config = GEEDatasetConfig.srtm_dem()
        
        download_gee_data(
            region=region,
            dataset_config=dem_config,
            output_folder=output_folder,
            logger=logger
        )
        
        logger.info("DEM数据下载成功完成")
    except Exception as e:
        logger.error(f"下载过程发生错误：{str(e)}")
        raise GEEDownloadError(str(e)) from e

if __name__ == "__main__":
    region_path = r'E:\GuiZhouProvinces\shp\gz_1000m.shp'
    output_folder = r'E:\GuiZhouProvinces\dem'
    log_file = r'E:\GuiZhouProvinces\logs\download_gee_dem.log'
    
    download_dem(region_path, output_folder, log_file)