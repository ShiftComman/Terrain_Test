import ee
import geemap
import os
import math
import geopandas as gpd
from typing import List, Tuple, Optional
import logging
import time
import backoff
from pathlib import Path
import aiohttp
import asyncio
from urllib.parse import urlparse
import aiofiles
import ssl

# 常量定义
SENTINEL2_BANDS = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B11', 'B12'] # 哨兵2波段
QA_BANDS = ['QA60', 'SCL'] # 质量控制波段
ALL_BANDS = SENTINEL2_BANDS + QA_BANDS # 所有波段
DEFAULT_SCALE = 10 # 默认像元大小
DEFAULT_MAX_SIZE = 30000000  # 40MB # 每个子区域的最大大小
DEFAULT_CLOUD_COVER = 20 # 云层覆盖率阈值
MAX_CONCURRENT_DOWNLOADS = 5 # 并发下载的最大数量
DOWNLOAD_TIMEOUT = 600  # 10分钟
CHUNK_SIZE = 1024 * 1024  # 1MB

class GEEDownloadError(Exception):
    """Google Earth Engine数据下载异常类。
    用于处理在下载GEE数据过程中可能出现的各种错误，包括但不限于：
    - 认证失败
    - 网络连接错误
    - 下载超时
    - 数据格式错误
    - 存储空间不足
    Attributes:
        message (str): 错误信息
        error_code (int, optional): 错误代码
        details (dict, optional): 详细错误信息
    """

    def __init__(self, message: str, error_code: int = None, details: dict = None):
        """初始化GEEDownloadError实例。

        Args:
            message: 错误描述信息
            error_code: 错误代码（可选）
            details: 详细错误信息字典（可选）
        """
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        super().__init__(self.message)

    def __str__(self) -> str:
        """返回格式化的错误信息。"""
        error_msg = self.message
        if self.error_code:
            error_msg = f"[错误代码 {self.error_code}] {error_msg}"
        if self.details:
            error_msg = f"{error_msg}\n详细信息: {self.details}"
        return error_msg

@backoff.on_exception(backoff.expo, Exception, max_tries=5)
def authenticate_gee(logger: logging.Logger) -> None:
    """
    验证Google Earth Engine服务。
    
    Args:
        logger: 日志记录器实例
    
    Raises:
        GEEDownloadError: 当验证失败时抛出
    """
    logger.info("尝试验证Google Earth Engine")
    try:
        ee.Initialize()
        logger.info("验证成功")
    except Exception as e:
        logger.warning(f"验证失败：{str(e)}。尝试重新验证...")
        try:
            ee.Authenticate()
            ee.Initialize()
            logger.info("重新验证成功")
        except Exception as e:
            error_msg = f"重新验证失败：{str(e)}"
            logger.error(error_msg)
            raise GEEDownloadError(error_msg) from e

def estimate_image_size(geometry: ee.Geometry, scale: int = 10) -> int:
    """估算影像大小（字节）"""
    # 获取区域面积（平方米）
    area = geometry.area().getInfo()
    # 计算像素数量（考虑所有波段）
    num_bands = 12  # B1-B12
    pixels = (area / (scale * scale)) * num_bands
    # 估算文件大小（每个像素4字节，增加50%安全余量）
    estimated_size = pixels * 4 * 1.5
    return estimated_size

def adaptive_split_geometry(geometry: ee.Geometry, max_size: int = 30000000, scale: int = 10) -> List[ee.Geometry]:
    """自适应分割几何体，确保每个子区域的预计大小不超过最大限制"""
    estimated_size = estimate_image_size(geometry, scale)
    
    if estimated_size <= max_size:
        return [geometry]
    
    # 计算需要的分割数（向上取整，并增加一个额外的分割以确保安全）
    split_factor = math.ceil(math.sqrt(estimated_size / max_size)) + 1
    
    bounds = geometry.bounds().getInfo()['coordinates'][0]
    min_x, min_y = bounds[0]
    max_x, max_y = bounds[2]
    
    width = max_x - min_x
    height = max_y - min_y
    
    x_step = width / split_factor
    y_step = height / split_factor
    
    sub_geometries = []
    for i in range(split_factor):
        for j in range(split_factor):
            sub_geometry = ee.Geometry.Rectangle([
                min_x + i * x_step,
                min_y + j * y_step,
                min_x + (i + 1) * x_step,
                min_y + (j + 1) * y_step
            ])
            sub_geometries.append(sub_geometry)
    
    return sub_geometries

def mask_s2_clouds(image):
    """在Sentinel-2影像中遮蔽云层。"""
    qa = image.select('QA60')
    cloudBitMask = 1 << 10
    cirrusBitMask = 1 << 11
    mask = qa.bitwiseAnd(cloudBitMask).eq(0).And(
        qa.bitwiseAnd(cirrusBitMask).eq(0))
    return image.updateMask(mask).divide(10000)

def get_sentinel2_collection(region: ee.Geometry, start_date: str, end_date: str, cloud_cover: int = 20) -> ee.ImageCollection:
    """获取给定区域和时间段的Sentinel-2影像集合。"""
    bands = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B11', 'B12','QA60','SCL']  # 选择要下载的波段
    return (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
            .filterBounds(region)
            .filterDate(start_date, end_date)
            .select(bands)
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', cloud_cover))
            .map(mask_s2_clouds))

@backoff.on_exception(backoff.expo, Exception, max_tries=5)
async def get_download_url(image: ee.Image, geometry: ee.Geometry) -> str:
    """获取下载链接，带重试机制"""
    return image.getDownloadURL({
        'scale': 10,
        'region': geometry,
        'format': 'GEO_TIFF',
        'crs': 'EPSG:4326'
    })

async def download_file(url: str, output_path: str, logger: logging.Logger) -> None:
    """异步下载文件"""
    timeout = aiohttp.ClientTimeout(total=DOWNLOAD_TIMEOUT)
    max_retries = 3
    retry_delay = 5  # 重试等待时间（秒）
    
    for attempt in range(max_retries):
        try:
            # 配置 SSL 上下文
            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE
            
            # 使用自定义的连接器
            conn = aiohttp.TCPConnector(
                ssl=ssl_context,
                force_close=True,
                enable_cleanup_closed=True,
                limit=10  # 限制并发连接数
            )
            
            async with aiohttp.ClientSession(
                connector=conn, 
                timeout=timeout,
                trust_env=True  # 信任环境变量中的代理设置
            ) as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        async with aiofiles.open(output_path, 'wb') as f:
                            async for chunk in response.content.iter_chunked(CHUNK_SIZE):
                                await f.write(chunk)
                        logger.info(f"文件下载成功：{output_path}")
                        return
                    else:
                        error_msg = f"下载失败，HTTP状态码：{response.status}"
                        logger.error(error_msg)
                        
        except asyncio.TimeoutError as e:
            error_msg = f"下载超时 (尝试 {attempt + 1}/{max_retries})"
            logger.warning(error_msg)
        except (aiohttp.ClientError, ssl.SSLError) as e:
            error_msg = f"网络错误 (尝试 {attempt + 1}/{max_retries}): {str(e)}"
            logger.warning(error_msg)
        except Exception as e:
            error_msg = f"未知错误 (尝试 {attempt + 1}/{max_retries}): {str(e)}"
            logger.warning(error_msg)
            
        if attempt < max_retries - 1:
            await asyncio.sleep(retry_delay)
            continue
            
    # 所有重试都失败后抛出异常
    raise GEEDownloadError(f"在 {max_retries} 次尝试后下载仍然失败")

async def download_subregion(composite: ee.Image, sub_geometry: ee.Geometry, output_path: str, logger):
    """下载子区域的影像"""
    try:
        url = await get_download_url(composite, sub_geometry)
        await download_file(url, output_path, logger)
    except Exception as e:
        logger.error(f"获取下载链接或下载失败：{str(e)}")
        raise

def split_geometry_by_groups(geometry: ee.Geometry, num_groups: int) -> List[ee.Geometry]:
    """将研究区域划分为指定数量的组
    
    Args:
        geometry: 输入的几何体
        num_groups: 需要划分的组数
    
    Returns:
        划分后的几何体列表
    """
    bounds = geometry.bounds().getInfo()['coordinates'][0]
    min_x, min_y = bounds[0]
    max_x, max_y = bounds[2]
    
    # 计算行列数（尽量使分割后的区域接近正方形）
    aspect_ratio = (max_x - min_x) / (max_y - min_y)
    num_cols = round(math.sqrt(num_groups * aspect_ratio))
    num_rows = math.ceil(num_groups / num_cols)
    
    width = max_x - min_x
    height = max_y - min_y
    
    x_step = width / num_cols
    y_step = height / num_rows
    
    groups = []
    for i in range(num_cols):
        for j in range(num_rows):
            group_geometry = ee.Geometry.Rectangle([
                min_x + i * x_step,
                min_y + j * y_step,
                min_x + (i + 1) * x_step,
                min_y + (j + 1) * y_step
            ])
            groups.append(group_geometry)
    
    return groups[:num_groups]  # 确保返回指定数量的组

def download_sentinel2_data(region: gpd.GeoDataFrame, start_date: str, end_date: str, 
                          output_folder: str, logger, cloud_cover_threshold: int = 20,
                          num_groups: int = 4):
    """下载给定区域和时间段的Sentinel-2数据。
    
    Args:
        region: 研究区域的GeoDataFrame
        start_date: 开始日期
        end_date: 结束日期
        output_folder: 输出文件夹
        logger: 日志记录器
        cloud_cover_threshold: 云覆盖阈值
        num_groups: 将研究区域划分为多少组
    """
    logger.info(f"下载{start_date}到{end_date}期间的Sentinel-2数据")
    
    if region.crs != 'EPSG:4326':
        logger.info(f"将坐标从{region.crs}转换为WGS84(EPSG:4326)")
        region = region.to_crs('EPSG:4326')
    
    os.makedirs(output_folder, exist_ok=True)
    
    # 获取研究区域的ee.Geometry对象
    region_geometry = ee.Geometry(region.geometry.iloc[0].__geo_interface__)
    
    try:
        # 将区域划分为指定数量的组
        group_geometries = split_geometry_by_groups(region_geometry, num_groups)
        logger.info(f"研究区域已被划分为 {len(group_geometries)} 组")
        
        async def process_group(group_geometry, group_index):
            try:
                logger.info(f"处理第 {group_index + 1}/{num_groups} 组")
                
                # 获取该组的影像集合
                s2_group = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
                    .filterBounds(group_geometry) \
                    .filterDate(start_date, end_date) \
                    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', cloud_cover_threshold))
                
                image_count = s2_group.size().getInfo()
                if image_count == 0:
                    logger.warning(f"组 {group_index + 1} 没有找到符合条件的影像")
                    return
                
                logger.info(f"组 {group_index + 1} 找到 {image_count} 张影像")
                
                # 对该组进行影像合成
                composite = s2_group.select(ALL_BANDS) \
                    .map(mask_s2_clouds) \
                    .reduce(ee.Reducer.median())
                
                # 将该组区域划分为适合下载的子区域
                sub_geometries = adaptive_split_geometry(group_geometry)
                logger.info(f"组 {group_index + 1} 被划分为 {len(sub_geometries)} 个下载子区域")
                
                # 下载该组的所有子区域
                for i, sub_geometry in enumerate(sub_geometries):
                    output_path = os.path.join(
                        output_folder, 
                        f'sentinel2_group_{group_index + 1}_part_{i + 1}.tif'
                    )
                    
                    # 检查文件是否已存在
                    if os.path.exists(output_path):
                        file_size = os.path.getsize(output_path)
                        if file_size > 0:  # 确保文件不是空文件
                            logger.info(f"文件已存在且非空，跳过下载：{output_path}")
                            continue
                        else:
                            logger.warning(f"发现空文件，将重新下载：{output_path}")
                            os.remove(output_path)  # 删除空文件
                    
                    await download_subregion(composite, sub_geometry, output_path, logger)
                
            except Exception as e:
                logger.error(f"处理组 {group_index + 1} 时发生错误: {str(e)}")
                raise
        
        # 使用信号量控制并发
        async def process_all_groups():
            semaphore = asyncio.Semaphore(MAX_CONCURRENT_DOWNLOADS)
            async def process_with_semaphore(group_geometry, index):
                async with semaphore:
                    return await process_group(group_geometry, index)
            
            tasks = [
                process_with_semaphore(geometry, i)
                for i, geometry in enumerate(group_geometries)
            ]
            await asyncio.gather(*tasks)
        
        # 执行异步下载
        asyncio.run(process_all_groups())
        
        logger.info("所有区域处理完成")
            
    except Exception as e:
        logger.error(f"处理过程发生错误: {str(e)}")
        raise

def main(
    region_path: str,
    output_folder: str,
    start_date: str,
    end_date: str,
    log_file: Optional[str] = None,
    num_groups: int=4
) -> None:
    """
    下载Sentinel-2数据的主函数。

    Args:
        region_path: 研究区域矢量文件路径
        output_folder: 输出文件夹路径
        start_date: 开始日期 (YYYY-MM-DD)
        end_date: 结束日期 (YYYY-MM-DD)
        log_file: 日志文件路径，可选

    Raises:
        GEEDownloadError: 当下载过程发生错误时抛出
    """
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
    logger.info("开始下载Sentinel-2数据")
    
    try:
        authenticate_gee(logger)
        region = gpd.read_file(region_path)
        download_sentinel2_data(region, start_date, end_date, output_folder, logger, num_groups)
        logger.info("数据下载成功完成")
    except Exception as e:
        error_msg = f"下载过程发生错误：{str(e)}"
        logger.error(error_msg)
        raise GEEDownloadError(error_msg) from e

# 测试
if __name__ == "__main__":
    region_path = r'E:\GuiZhouProvinces\shp\gz_1000m.shp'
    output_folder = r'E:\GuiZhouProvinces\s2'
    start_date = '2023-01-01'
    end_date = '2023-12-31'
    log_file = r'E:\GuiZhouProvinces\logs\download_gee.log'
    num_groups = 80
    # 将研究区域划分为80组进行处理
    main(region_path, output_folder, start_date, end_date, log_file, num_groups)