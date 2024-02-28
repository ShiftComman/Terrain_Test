# 按坡度获取山脊山谷
import arcpy
from arcpy.sa import *

# from arcpy.ia import *
from arcpy import env

work_path = "D:\ArcGISProjects\syraster\SJSG.gdb"
env.workspace = work_path

dem_file_path = "D:\ArcGISProjects\MyProject\LBCL.gdb\SY_DEM_5_DT_3"
dem = arcpy.Raster(dem_file_path)


# 山谷线
def get_sg_line(dem):
    with arcpy.EnvManager(parallelProcessingFactor="0"):
        print("计算负地形")
        fd = Test(
            dem - FocalStatistics(dem, NbrRectangle(11, 11, "CELL"), "MEAN", "DATA"),
            "value<0",
        )
        print("绝对值")
        a = Abs(Minus(5000, dem))
        print("流向")
        b = FlowDirection(a, "NORMAL", None, "D8")
        print("流量")
        c = FlowAccumulation(b, None, "FLOAT", "D8")
        print("end")
        d = Test(c, "value=0")
        e = Con(fd * d, 1)
        print("Success")
        return e


# 山脊线
def get_sj_line(dem):
    with arcpy.EnvManager(parallelProcessingFactor="0"):
        print("计算正地形")
        zd = Test(
            dem - FocalStatistics(dem, NbrRectangle(11, 11, "CELL"), "MEAN", "DATA"),
            "value>0",
        )
        print("填洼")
        a = Fill(dem, None)
        print("流向")
        b = FlowDirection(a, "FORCE", None, "D8")
        print("流量")
        c = FlowAccumulation(b, None, "FLOAT", "D8")
        print("end")
        d = Test(c, "value=0")
        e = Con(zd * d, 1)
        print("Success")
        return e


result = get_sj_line(dem)
result.save("SY_DEM_5_SJ")
