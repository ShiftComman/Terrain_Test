import arcpy
from arcpy.sa import *
from arcpy import env

work_path = "D:\ArcGISProjects\syraster\END.gdb"
env.workspace = work_path
# DEM数据
dem_path = "SY_DEM_5_DT_3"


# 获取栅格数据窗口内的相对高程位置
def calc_dem_raster(dem_raster, focus_size):
    min_raster = FocalStatistics(
        dem_raster, NbrRectangle(focus_size, focus_size, "CELL"), "MINIMUM", "DATA"
    )
    max_raster = FocalStatistics(
        dem_raster, NbrRectangle(focus_size, focus_size, "CELL"), "MAXIMUM", "DATA"
    )
    mood_raster = max_raster - min_raster
    meet_raster = dem_raster - min_raster
    result_raster = Con(
        (((meet_raster / mood_raster) <= 0.25) | (mood_raster <= 0)),
        1,
        Con(
            (
                ((meet_raster / mood_raster) > 0.25)
                & ((meet_raster / mood_raster) <= 0.5)
            ),
            2,
            Con(
                (
                    ((meet_raster / mood_raster) > 0.5)
                    & ((meet_raster / mood_raster) <= 0.85)
                ),
                3,
                4,
            ),
        ),
    )
    return result_raster


for one in range(11, 51, 10):
    result_raster = calc_dem_raster(dem_path, one)
    result_raster.save(f"SY_DEM_5M_RECLASS_{one}")
