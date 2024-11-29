import geopandas as gpd
from shapely.geometry import Polygon, MultiPolygon, LineString
from shapely.ops import unary_union, polygonize
import logging
from datetime import datetime
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s')

def fix_self_crossing_boundaries(input_file, output_file):
    """
    修复面边界自相交的函数
    
    原理：
    1. 将面转换为边界线
    2. 将所有线段打碎成不相交的线段集合
    3. 重新构建面
    """
    try:
        start_time = datetime.now()
        logging.info(f"开始处理: {start_time}")
        
        # 读取数据
        logging.info(f"正在读取文件: {input_file}")
        gdf = gpd.read_file(input_file)
        initial_count = len(gdf)
        initial_area = gdf.geometry.area.sum()
        
        def fix_polygon(geom):
            try:
                # 只处理无效或自相交的几何体
                if geom.is_valid and not geom.is_simple:
                    boundary = geom.boundary
                    
                    # 如果是MultiLineString，需要获取所有线段
                    if hasattr(boundary, 'geoms'):
                        lines = list(boundary.geoms)
                    else:
                        lines = [boundary]
                    
                    # 将所有线段合并并重新分割成不相交的线段
                    merged = unary_union(lines)
                    
                    # 从不相交的线段重建面
                    new_polygons = list(polygonize(merged))
                    
                    if new_polygons:
                        # 选择面积最接近原始面积的多边形
                        result = unary_union(new_polygons)
                        if result.is_valid:
                            area_before = geom.area
                            area_after = result.area
                            # 如果修复后的面积大于原始面积，保持原样
                            if area_after > area_before:
                                return geom
                            return result
                
                return geom
                
            except Exception as e:
                logging.warning(f"处理单个几何体时出错: {str(e)}")
                return geom
        
        # 处理每个几何体
        logging.info("开始修复自相交边界...")
        fixed_geometries = []
        problem_count = 0
        
        for idx, row in tqdm(gdf.iterrows(), total=len(gdf)):
            original_geom = row.geometry
            fixed_geom = fix_polygon(original_geom)
            
            if fixed_geom.wkt != original_geom.wkt:
                problem_count += 1
                
            fixed_geometries.append(fixed_geom)
        
        # 更新几何体
        gdf['geometry'] = fixed_geometries
        
        # 转换为单部件
        gdf = gdf.explode(index_parts=True).reset_index(drop=True)
        
        # 计算最终统计信息
        final_count = len(gdf)
        final_area = gdf.geometry.area.sum()
        area_diff = abs(final_area - initial_area)
        
        # 保存结果
        logging.info("正在保存修复后的数据...")
        gdf.to_file(output_file, encoding='utf-8')
        
        # 输出统计信息
        end_time = datetime.now()
        processing_time = end_time - start_time
        
        logging.info(f"\n处理完成:")
        logging.info(f"发现并修复的问题几何体数量: {problem_count}")
        logging.info(f"初始要素数量: {initial_count}")
        logging.info(f"最终要素数量: {final_count}")
        logging.info(f"要素数量变化: {final_count - initial_count}")
        logging.info(f"初始总面积: {initial_area:.2f}")
        logging.info(f"最终总面积: {final_area:.2f}")
        logging.info(f"面积差异: {area_diff:.2f}")
        logging.info(f"总处理时间: {processing_time}")
        logging.info(f"结果已保存至: {output_file}")
        
        return True
        
    except Exception as e:
        logging.error(f"处理过程出错: {str(e)}")
        return False

if __name__ == "__main__":
    # 使用示例
    input_shp = r"C:\Users\Runker\Desktop\ele_sb\gl_merge_data_singles_5_split_curve.shp"
    output_shp = r"C:\Users\Runker\Desktop\ele_sb\gl_merge_data_singles_5_split_curve_repairs.shp"
    fix_self_crossing_boundaries(input_shp, output_shp)
    