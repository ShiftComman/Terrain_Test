# coding:utf-8
import arcpy
from arcpy.sa import *

# from arcpy.ia import *
from arcpy import env

work_path = "D:\ArcGISProjects\syraster\SJSG.gdb"
env.workspace = work_path

dem_file_path = "D:\ArcGISProjects\MyProject\LBCL.gdb\SY_DEM_5_DT_3"
dem = arcpy.Raster(dem_file_path)


# 山谷山脊线
def get_line(dem):
    with arcpy.EnvManager(parallelProcessingFactor="0"):
        print("计算坡向坡度")
        z_asp = Aspect(dem, "PLANAR", "METER", "GEODESIC_AZIMUTHS")
        z_slope = Slope(z_asp, "DEGREE", 1, "PLANAR", "METER")
        print("计算反地形")
        # fdx = RasterCalculator("fdx", f"3090 - {dem}")
        fdx = 3090 - dem
        print("计算负地形坡向坡度")
        f_asp = Aspect(fdx, "PLANAR", "METER", "GEODESIC_AZIMUTHS")
        f_slope = Slope(f_asp, "DEGREE", 1, "PLANAR", "METER")

        print("计算坡度变化率")
        # pd_rg = RasterCalculator(
        #     "pd_rg", f"(({z_slope} + {f_slope}) - Abs({z_slope} - {f_slope})) / 2"
        # )
        pd_rg = ((z_slope + f_slope) - Abs(z_slope - f_slope)) / 2

        print("焦点统计")
        mean_dem = FocalStatistics(dem, NbrRectangle(3, 3, "CELL"), "MEAN", "DATA")
        print("得到正负地形")

        # zf_dem = RasterCalculator("zf_dem", f"{dem} - {mean_dem}")
        zf_dem = dem - mean_dem
        print("得到山脊线")

        sj_raster = Con((zf_dem > 0) & (pd_rg > 70), 1, 0)

        sg_raster = Con((zf_dem < 0) & (pd_rg > 70), 1, 0)

        return sj_raster, sg_raster


result_sj = get_line(dem)[0]
result_sj.save("SY_DEM_5_SJ_PD")
result_sg = get_line(dem)[1]
result_sg.save("SY_DEM_5_SG_PD")
