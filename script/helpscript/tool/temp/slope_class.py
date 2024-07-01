import arcpy
from arcpy.sa import *
from arcpy.ia import *
from arcpy import env
import time

work_path = r"D:\ArcGISProjects\workspace\shbyq\feature_raster_file\features_data_dy.gdb"
env.workspace = work_path


# 焦点统计计算
def calc_foc(dem_raster, size):
    nbrhood = NbrRectangle(size, size, "CELL")
    out_raster = FocalStatistics(dem_raster, nbrhood, "MEAN", "DATA")
    return out_raster


# TPI计算并保存
def calc_tpi_save(dem_raster, foc_raster, out_file):
    tpi_raster = dem_raster - foc_raster
    tpi_raster.save(out_file)
    print("save success")


# TPI计算
def calc_tpi(dem_raster, foc_raster):
    tpi_raster = dem_raster - foc_raster
    return tpi_raster


# 获取栅格值范围
def get_values_round(raster):
    mimimum = arcpy.GetRasterProperties_management(raster, "MINIMUM")
    maximum = arcpy.GetRasterProperties_management(raster, "MAXIMUM")
    return (mimimum, maximum)


# 重分类
def reclass(tpi_raster_min, tpi_raster_max, slope_raster, factor, asp_factor):
    # 表达式
    # output_raster = arcpy.sa.RasterCalculator(f' Con(({tpi_raster_min}<=-1)&({tpi_raster_max}<=-1),1, Con(({tpi_raster_min}<=-1)&(({tpi_raster_max}>-1)&({tpi_raster_max}<1)),2,))')
    # 表达式
    # output_raster = arcpy.sa.RasterCalculator(
    #     f" Con(({tpi_raster_min}<=-1)&({tpi_raster_max}<=-1),1,"
    #     f"Con(({tpi_raster_min}<=-1)&(({tpi_raster_max}>-1)&({tpi_raster_max}<1)),2,"
    #     f"Con(({tpi_raster_min}<=-1)&({tpi_raster_max}>=1),3,"
    #     f"Con((({tpi_raster_min}>-1)&({tpi_raster_min}<1))&({tpi_raster_max}<=-1),4,"
    #     f"Con((({tpi_raster_min}>-1)&({tpi_raster_min}<1))&(({tpi_raster_max}>-1)&({tpi_raster_max}<1))&({slope_raster}<=5),5,"
    #     f"Con((({tpi_raster_min}>-1)&({tpi_raster_min}<1))&(({tpi_raster_max}>-1)&({tpi_raster_max}<1))&({slope_raster}>5),6,"
    #     f"Con((({tpi_raster_min}>-1)&({tpi_raster_min}<1))&({tpi_raster_max}>=1),7,"
    #     f"Con(({tpi_raster_min}>=1)&({tpi_raster_max}<=-1),8,"
    #     f"Con(({tpi_raster_min}>=1)&(({tpi_raster_max}>-1)&({tpi_raster_max}<1)),9,"
    #     f"Con(({tpi_raster_min}>=1)&({tpi_raster_max>=1}),10,11)))))))))"
    # )

    output_raster = Con(
        ((tpi_raster_min <= -factor) & (tpi_raster_max <= -factor)),
        1,
        Con(
            (
                (tpi_raster_min <= -factor)
                & ((tpi_raster_max > -factor) & (tpi_raster_max < factor))
            ),
            2,
            Con(
                ((tpi_raster_min <= -factor) & (tpi_raster_max >= factor)),
                3,
                Con(
                    (
                        ((tpi_raster_min > -factor) & (tpi_raster_min < factor))
                        & (tpi_raster_max <= -factor)
                    ),
                    4,
                    Con(
                        (
                            ((tpi_raster_min > -factor) & (tpi_raster_min < factor))
                            & ((tpi_raster_max > -factor) & (tpi_raster_max < factor))
                            & (slope_raster <= asp_factor)
                        ),
                        5,
                        Con(
                            (
                                ((tpi_raster_min > -factor) & (tpi_raster_min < factor))
                                & (
                                    (tpi_raster_max > -factor)
                                    & (tpi_raster_max < factor)
                                )
                                & (slope_raster > asp_factor)
                            ),
                            6,
                            Con(
                                (
                                    (
                                        (tpi_raster_min > -factor)
                                        & (tpi_raster_min < factor)
                                    )
                                    & (tpi_raster_max >= factor)
                                ),
                                7,
                                Con(
                                    (
                                        (tpi_raster_min >= factor)
                                        & (tpi_raster_max <= -factor)
                                    ),
                                    8,
                                    Con(
                                        (
                                            (tpi_raster_min >= factor)
                                            & (
                                                (tpi_raster_max > -factor)
                                                & (tpi_raster_max < factor)
                                            )
                                        ),
                                        9,
                                        Con(
                                            (
                                                (tpi_raster_min >= factor)
                                                & (tpi_raster_max >= factor)
                                            ),
                                            10,
                                            11,
                                        ),
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        ),
    )

    return output_raster


def reclass_factor(tpi, slope, asp):
    # ((tpi > -0.5) & (tpi < 0.5)) & (slope < 6)
    # (tpi > 1) & ((slope < 24) & (slope > 6))
    # ((tpi > 0.5) & (tpi > 1)) & ((slope < 24) & (slope > 6))
    # (tpi > 1) & (slope < 6)
    # (((tpi > -0.5) & (tpi < 0.5))& ((slope < 24) & (slope > 6))& ((asp < 135) & (asp > 315) & (asp < 360)))
    # (((tpi > -0.5) & (tpi < 0.5))& ((slope < 24) & (slope > 6))& ((asp > 135) & (asp < 315)))
    # (tpi < -0.5) & ((slope < 24) & (slope > 6)) & ((asp < 135) & (asp > 315) & (asp < 360))
    # (tpi < -0.5) & ((slope < 24) & (slope > 6)) & ((asp > 135) & (asp < 315))
    # ((asp > 24) & (asp < 90)) & ((asp < 135) & (asp > 315) & (asp < 360))
    # ((asp > 24) & (asp < 90)) & ((asp > 135) & (asp < 315))
    # (tpi < -0.5) & (slope < 6)
    output_raster = Con(
        (((tpi > -0.5) & (tpi < 0.5)) & (slope < 6)),
        1,
        Con(
            ((tpi > 1) & ((slope < 24) & (slope > 6))),
            2,
            Con(
                (((tpi > 0.5) & (tpi > 1)) & ((slope < 24) & (slope > 6))),
                3,
                Con(
                    ((tpi > 1) & (slope < 6)),
                    4,
                    Con(
                        (
                            ((tpi > -0.5) & (tpi < 0.5))
                            & ((slope < 24) & (slope > 6))
                            & ((asp < 135) & (asp > 315) & (asp < 360))
                        ),
                        5,
                        Con(
                            (
                                ((tpi > -0.5) & (tpi < 0.5))
                                & ((slope < 24) & (slope > 6))
                                & ((asp > 135) & (asp < 315))
                            ),
                            6,
                            Con(
                                (
                                    (tpi < -0.5)
                                    & ((slope < 24) & (slope > 6))
                                    & ((asp < 135) & (asp > 315) & (asp < 360))
                                ),
                                7,
                                Con(
                                    (
                                        (tpi < -0.5)
                                        & ((slope < 24) & (slope > 6))
                                        & ((asp > 135) & (asp < 315))
                                    ),
                                    8,
                                    Con(
                                        (
                                            ((asp > 24) & (asp < 90))
                                            & ((asp < 135) & (asp > 315) & (asp < 360))
                                        ),
                                        9,
                                        Con(
                                            (
                                                ((asp > 24) & (asp < 90))
                                                & ((asp > 135) & (asp < 315))
                                            ),
                                            10,
                                            Con(((tpi < -0.5) & (slope < 6)), 11, 12),
                                        ),
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        ),
    )
    return output_raster


# 得到坡度


def calc_slope(dem_raster):
    out_raster = arcpy.sa.Slope(dem_raster, "DEGREE", 1, "PLANAR", "METER")
    return out_raster


# 得到坡向
def calc_asp(dem_raster):
    out_raster = arcpy.sa.Aspect(dem_raster, "PLANAR", "METER", "GEODESIC_AZIMUTHS")
    return out_raster


# 结果输出
def get_result_raster(dem_raster, min_size, max_size, factor, slope_factor):
    foc_raster_min = calc_foc(dem_raster, min_size)
    foc_raster_max = calc_foc(dem_raster, max_size)
    tpi_raster_min = calc_tpi(dem_raster, foc_raster_min)
    tpi_raster_max = calc_tpi(dem_raster, foc_raster_max)
    slope_raster = calc_slope(dem_raster)
    result_raster = reclass(
        tpi_raster_min, tpi_raster_max, slope_raster, factor, slope_factor
    )
    return result_raster


def get_result_raster_factor(dem_raster, tpi_size):
    slop_raster = calc_slope(dem_raster)
    asp_raster = calc_asp(dem_raster)
    foc_raster = calc_foc(dem_raster, tpi_size)
    tpi_raster = calc_tpi(dem_raster, foc_raster)
    result_raster = reclass_factor(tpi_raster, slop_raster, asp_raster)
    return result_raster


# 按条执行并输出

# 5mDEM min 5  max 45  range 5*5 45*5

dem_raster_5m = arcpy.Raster(r"D:\ArcGISProjects\workspace\shbyq\feature_raster_file\features_data_dy.gdb\DEM")


def main():
    print("Start!")
    start_time = time.time()
    # result_5m = get_result_raster_factor(dem_raster_5m, 200)
    result_5m = get_result_raster(dem_raster_5m, 5, 45, 1, 5)
    result_5m.save("SY_FACTOR_5_45_1_5")
    print(f"耗时{time.time()-start_time} S")


main()
