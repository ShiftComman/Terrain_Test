import os
import sys
import glob
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from osgeo import gdal, gdalconst
import numpy as np
import threading
from datetime import datetime

# 设置环境变量
# os.environ['GDAL_DATA'] = r'D:\Python38\Lib\site-packages\osgeo\data\gdal'
# os.environ['PROJ_LIB'] = r'D:\Python38\Lib\site-packages\fiona\proj_data'

# # 将 GDAL 目录添加到系统路径
# gdal_path = r'D:\Python38\Lib\site-packages\osgeo'
# if gdal_path not in sys.path:
#     sys.path.append(gdal_path)

class AerialImageMergerApp:
    def __init__(self, master):
        self.master = master
        master.title("Aerial Image Merger Application")
        master.geometry("800x600")

        # 初始化实例变量
        self.input_folder = tk.StringVar()
        self.output_file = tk.StringVar()
        self.target_srs = tk.StringVar(value="EPSG:4544")
        self.target_res_x = tk.DoubleVar(value=0.1)
        self.target_res_y = tk.DoubleVar(value=0.1)
        self.apply_color_balance = tk.BooleanVar(value=True)
        self.low_percentile = tk.DoubleVar(value=2.0)
        self.high_percentile = tk.DoubleVar(value=98.0)
        print(f"初始化时 apply_color_balance 的类型: {type(self.apply_color_balance)}")
        print(f"初始化时 apply_color_balance 的值: {self.apply_color_balance.get()}")
        # 设置样式
        self.style = ttk.Style()
        self.style.theme_use('clam')
        self.style.configure('TButton', padding=5, relief="flat", background="#4CAF50", foreground="white")
        self.style.map('TButton', background=[('active', '#45a049')])
        self.style.configure('TEntry', padding=5)
        self.style.configure('TCheckbutton', background="#f0f0f0")
        self.style.configure('TProgressbar', thickness=20)

        # 创建主框架
        main_frame = ttk.Frame(master, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # 输入文件夹选择
        ttk.Label(main_frame, text="输入文件夹:").grid(row=0, column=0, sticky=tk.W, pady=5)
        ttk.Entry(main_frame, textvariable=self.input_folder, width=50).grid(row=0, column=1, pady=5)
        ttk.Button(main_frame, text="浏览", command=self.browse_input_folder).grid(row=0, column=2, padx=5, pady=5)

        # 输出文件选择
        ttk.Label(main_frame, text="输出文件:").grid(row=1, column=0, sticky=tk.W, pady=5)
        ttk.Entry(main_frame, textvariable=self.output_file, width=50).grid(row=1, column=1, pady=5)
        ttk.Button(main_frame, text="浏览", command=self.browse_output_file).grid(row=1, column=2, padx=5, pady=5)

        # 目标坐标系选择
        ttk.Label(main_frame, text="目标坐标系:").grid(row=2, column=0, sticky=tk.W, pady=5)
        ttk.Entry(main_frame, textvariable=self.target_srs, width=50).grid(row=2, column=1, pady=5)

        # 目标分辨率设置
        ttk.Label(main_frame, text="目标分辨率 (米):").grid(row=3, column=0, sticky=tk.W, pady=5)
        res_frame = ttk.Frame(main_frame)
        res_frame.grid(row=3, column=1, sticky=tk.W, pady=5)
        ttk.Entry(res_frame, textvariable=self.target_res_x, width=10).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Label(res_frame, text="x").pack(side=tk.LEFT, padx=(0, 5))
        ttk.Entry(res_frame, textvariable=self.target_res_y, width=10).pack(side=tk.LEFT, padx=(0, 5))

        # 色彩平衡选项
        ttk.Checkbutton(main_frame, text="应用色彩平衡", variable=self.apply_color_balance).grid(row=4, column=0, columnspan=2, sticky=tk.W, pady=5)

        # 色彩平衡百分位数设置
        ttk.Label(main_frame, text="色彩平衡百分位数:").grid(row=5, column=0, sticky=tk.W, pady=5)
        percentile_frame = ttk.Frame(main_frame)
        percentile_frame.grid(row=5, column=1, sticky=tk.W, pady=5)
        ttk.Entry(percentile_frame, textvariable=self.low_percentile, width=10).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Label(percentile_frame, text="-").pack(side=tk.LEFT, padx=(0, 5))
        ttk.Entry(percentile_frame, textvariable=self.high_percentile, width=10).pack(side=tk.LEFT, padx=(0, 5))

        # 处理按钮
        self.process_button = ttk.Button(main_frame, text="开始处理", command=self.start_processing)
        self.process_button.grid(row=6, column=0, columnspan=2, pady=10)

        # 帮助按钮
        help_button = ttk.Button(main_frame, text="帮助", command=self.show_help)
        help_button.grid(row=6, column=2, pady=10, padx=5)

        # 进度条
        self.progress = ttk.Progressbar(main_frame, orient=tk.HORIZONTAL, length=300, mode='indeterminate')
        self.progress.grid(row=7, column=0, columnspan=3, pady=10)

        # 状态标签
        self.status_label = ttk.Label(main_frame, text="")
        self.status_label.grid(row=8, column=0, columnspan=3, pady=5)

    def browse_input_folder(self):
        folder = filedialog.askdirectory()
        if folder:
            self.input_folder.set(folder)

    def browse_output_file(self):
        file = filedialog.asksaveasfilename(defaultextension=".tif", filetypes=[("TIFF files", "*.tif")])
        if file:
            self.output_file.set(file)

    def validate_inputs(self):
        """验证用户输入"""
        if not self.input_folder.get():
            raise ValueError("请选择输入文件夹")
        if not self.output_file.get():
            raise ValueError("请选择输出文件")
        if not self.target_srs.get():
            raise ValueError("请指定目标坐标系")
        if self.target_res_x.get() <= 0 or self.target_res_y.get() <= 0:
            raise ValueError("目标分辨率必须大于0")
        if self.low_percentile.get() < 0 or self.low_percentile.get() > 100:
            raise ValueError("低百分位数必须在0到100之间")
        if self.high_percentile.get() < 0 or self.high_percentile.get() > 100:
            raise ValueError("高百分位数必须在0到100之间")
        if self.low_percentile.get() >= self.high_percentile.get():
            raise ValueError("低百分位数必须小于高百分位数")

    def start_processing(self):
        try:
            self.validate_inputs()
            input_folder = self.input_folder.get()
            output_file = self.output_file.get()
            target_srs = self.target_srs.get()
            target_res = (self.target_res_x.get(), self.target_res_y.get())

            self.process_button.config(state=tk.DISABLED)
            self.progress.start()
            self.status_label.config(text="处理中...")

            # 在新线程中运行处理过程
            threading.Thread(target=self.process_and_merge_images, 
                             args=(input_folder, output_file, target_srs, target_res), 
                             daemon=True).start()
        except ValueError as e:
            messagebox.showerror("输入错误", str(e))

    def process_file(self, input_file, output_file, target_srs, target_res):
        try:
            print(f"开始处理文件: {input_file}")
            print(f"apply_color_balance 类型: {type(self.apply_color_balance)}")
            print(f"apply_color_balance 值: {self.apply_color_balance.get()}")

            if os.path.exists(output_file):
                self.update_status(f"输出文件已存在，跳过处理: {output_file}")
                return output_file

            # 检查输出文件夹的写入权限
            output_dir = os.path.dirname(output_file)
            if not os.access(output_dir, os.W_OK):
                raise PermissionError(f"没有写入权限: {output_dir}")

            ds = gdal.Open(input_file, gdalconst.GA_ReadOnly)
            if ds is None:
                self.update_status(f"无法打开文件: {input_file}", error=True)
                return None

            # 获取图像信息
            band_count = min(ds.RasterCount, 3)  # 使用前三个波段，如果少于3个则使用所有可用波段
            data_type = ds.GetRasterBand(1).DataType

            print(f"输入图像波段数: {ds.RasterCount}, 使用波段数: {band_count}")

            # 读取所有波段的数据
            data = [ds.GetRasterBand(i + 1).ReadAsArray() for i in range(band_count)]

            # 应用色彩平衡（如果启用）
            print("准备应用色彩平衡...")
            apply_balance = self.apply_color_balance.get()
            print(f"是否应用色彩平衡: {apply_balance}")
            
            if apply_balance:
                print("开始应用色彩平衡...")
                balanced_data = self.apply_color_balance_to_data(data)
                print("色彩平衡应用完成")
            else:
                print("跳过色彩平衡")
                balanced_data = data

            # 创建临时文件
            temp_file = output_file + "_temp.tif"
            # 获取输入图像的 NODATA 值
            nodata_value = ds.GetRasterBand(1).GetNoDataValue()
            if nodata_value is None:
                nodata_value = 255  # 如果输入没有 NODATA 值，使用 255

            # 创建新的数据集以保存处理后的数据
            driver = gdal.GetDriverByName('GTiff')
            balanced_ds = driver.Create(temp_file, ds.RasterXSize, ds.RasterYSize, max(band_count, 3), data_type)
            balanced_ds.SetGeoTransform(ds.GetGeoTransform())
            balanced_ds.SetProjection(ds.GetProjection())

            # 写入处理后的数据
            for i in range(band_count):
                balanced_band = balanced_ds.GetRasterBand(i+1)
                balanced_band.WriteArray(balanced_data[i])
                balanced_band.SetNoDataValue(nodata_value)
                balanced_ds.GetRasterBand(i+1).WriteArray(balanced_data[i])
                # 复制原始波段的元数据
                original_band = ds.GetRasterBand(i+1)
                balanced_band = balanced_ds.GetRasterBand(i+1)
                balanced_band.SetNoDataValue(original_band.GetNoDataValue())
                balanced_band.SetScale(original_band.GetScale())
                balanced_band.SetOffset(original_band.GetOffset())

            # 如果原始图像少于3个波段，用最后一个波段填充剩余的波段
            if band_count < 3:
                for i in range(band_count, 3):
                    balanced_ds.GetRasterBand(i+1).WriteArray(balanced_data[-1])
                    balanced_ds.GetRasterBand(i+1).SetNoDataValue(original_band.GetNoDataValue())
                    balanced_ds.GetRasterBand(i+1).SetScale(original_band.GetScale())
                    balanced_ds.GetRasterBand(i+1).SetOffset(original_band.GetOffset())

            # 复制数据集级别的元数据
            balanced_ds.SetMetadata(ds.GetMetadata())

            # 关闭数据集
            ds = None
            balanced_ds = None

            # 进行坐标转换和重采样
            options = gdal.WarpOptions(
                dstSRS=target_srs,
                xRes=target_res[0], yRes=target_res[1],
                resampleAlg=gdal.GRA_Bilinear,
                format='GTiff',
                creationOptions=['COMPRESS=LZW', 'TILED=YES', 'BLOCKXSIZE=256', 'BLOCKYSIZE=256', 'BIGTIFF=YES'],
                 srcNodata=nodata_value,
                dstNodata=nodata_value,
                warpOptions=['CUTLINE_ALL_TOUCHED=TRUE'],
                multithread=True,
                warpMemoryLimit=4096
            )

            self.update_status(f"正在处理文件: {input_file}")
            gdal.PushErrorHandler('CPLQuietErrorHandler')
            gdal.Warp(output_file, temp_file, options=options)
            gdal.PopErrorHandler()
            error_msg = gdal.GetLastErrorMsg()
            if error_msg:
                raise RuntimeError(f"GDAL错误: {error_msg}")
            
            # 删除临时文件
            os.remove(temp_file)
            
            self.update_status(f"成功处理文件: {input_file}")
            return output_file

        except Exception as e:
            import traceback
            error_msg = f"处理文件 {input_file} 时出错: {str(e)}\n{traceback.format_exc()}"
            self.update_status(error_msg, error=True)
            print(error_msg)
            return None

    def apply_color_balance_to_data(self, data):
        print("进入 apply_color_balance_to_data 方法")
        balanced_data = []
        low_percentile = self.low_percentile.get()
        high_percentile = self.high_percentile.get()

        print(f"低百分位数: {low_percentile}, 高百分位数: {high_percentile}")

        for i, band in enumerate(data):
            print(f"处理波段 {i+1}")
            # 计算有效数据的掩码（排除NoData值）
            valid_mask = np.isfinite(band)
            
            if np.any(valid_mask):
                # 只对有效数据进行处理
                valid_data = band[valid_mask]
                
                # 计算指定的百分位数
                low_val = np.percentile(valid_data, low_percentile)
                high_val = np.percentile(valid_data, high_percentile)
                
                print(f"波段 {i+1} 的低值: {low_val}, 高值: {high_val}")
                
                # 线性拉伸
                balanced_band = np.clip(band, low_val, high_val)
                balanced_band = (balanced_band - low_val) / (high_val - low_val)
                
                # 将结果缩放回原始数据类型的范围
                if band.dtype == np.uint16:
                    balanced_band = (balanced_band * 65535).astype(np.uint16)
                elif band.dtype == np.float32:
                    # 对于浮点型数据，我们保持原始范围
                    balanced_band = (balanced_band * (high_val - low_val) + low_val).astype(np.float32)
                else:
                    balanced_band = (balanced_band * 255).astype(np.uint8)
                
                # 保留原始的NoData值
                balanced_band[~valid_mask] = band[~valid_mask]
            else:
                print(f"波段 {i+1} 没有有效数据")
                balanced_band = band
            
            balanced_data.append(balanced_band)
        
        print("完成色彩平衡")
        return balanced_data

    def process_and_merge_images(self, input_folder, output_file, target_srs, target_res):
        try:
            # 获取输入文件夹中所有的tif和ecw文件
            input_files = glob.glob(os.path.join(input_folder, '**', '*.tif'), recursive=True) + \
                        glob.glob(os.path.join(input_folder, '**', '*.ecw'), recursive=True)

            if not input_files:
                self.update_status("没有找到TIF或ECW文件", error=True)
                return

            # 处理每个文件并收集它们的边界
            processed_files = []
            all_bounds = []
            for i, input_file in enumerate(input_files):
                output_file_temp = os.path.splitext(input_file)[0] + '_processed.tif'
                processed = self.process_file(input_file, output_file_temp, target_srs, target_res)
                if processed:
                    processed_files.append(processed)
                    ds = gdal.Open(processed)
                    if ds is not None:
                        bounds = self.get_raster_bounds(ds)
                        all_bounds.append(bounds)
                        print(f"处理后的文件 {processed} 有 {ds.RasterCount} 个波段")
                        ds = None
                    else:
                        print(f"无法打开处理后的文件: {processed}")
                self.update_status(f"处理进度: {i+1}/{len(input_files)}")

            if not processed_files:
                self.update_status("没有成功处理的文件", error=True)
                return

            # 计算所有图像的总体边界
            min_x = min(bound[0] for bound in all_bounds)
            min_y = min(bound[1] for bound in all_bounds)
            max_x = max(bound[2] for bound in all_bounds)
            max_y = max(bound[3] for bound in all_bounds)

            # 计算输出图像的大小
            width = int((max_x - min_x) / target_res[0])
            height = int((max_y - min_y) / target_res[1])

            # 设置 NODATA 值
            nodata_value = 255  # 对于 8 位无符号整数，255 通常用作 NODATA

            # 创建输出图像
            driver = gdal.GetDriverByName('GTiff')
            out_ds = driver.Create(output_file, width, height, 3, gdal.GDT_Byte,
                                options=['COMPRESS=LZW', 'TILED=YES', 'BIGTIFF=YES'])

            if out_ds is None:
                self.update_status(f"无法创建输出文件: {output_file}", error=True)
                return

            # 设置输出图像的地理变换和投影
            out_ds.SetGeoTransform((min_x, target_res[0], 0, max_y, 0, -target_res[1]))
            out_ds.SetProjection(target_srs)

            # 初始化输出数据并设置 NODATA 值
            for i in range(3):
                band = out_ds.GetRasterBand(i+1)
                band.SetNoDataValue(nodata_value)
                band.Fill(nodata_value)

            # 将每个处理后的图像写入输出图像
            for processed_file in processed_files:
                self.update_status(f"正在合并: {processed_file}")
                gdal.PushErrorHandler('CPLQuietErrorHandler')
                result = gdal.Warp(out_ds, processed_file, format='VRT', 
                                options=gdal.WarpOptions(
                                    resampleAlg=gdal.GRA_Bilinear,
                                    srcNodata=nodata_value,
                                    dstNodata=nodata_value,
                                    multithread=True,
                                    warpOptions=['CUTLINE_ALL_TOUCHED=TRUE']
                                ))
                gdal.PopErrorHandler()
                if result is None:
                    print(f"合并文件 {processed_file} 时出错: {gdal.GetLastErrorMsg()}")

            out_ds = None  # 关闭数据集，确保数据写入磁盘

            # 清理临时文件
            for file in processed_files:
                try:
                    os.remove(file)
                except Exception as e:
                    self.update_status(f"删除临时文件 {file} 时出错: {str(e)}", error=True)

            self.update_status("处理完成")
            self.master.after(0, self.show_completion_message)
        except Exception as e:
            import traceback
            error_msg = f"处理过程中出错: {str(e)}\n{traceback.format_exc()}"
            self.update_status(error_msg, error=True)
            print(error_msg)

    def get_raster_bounds(self, ds):
        """获取栅格数据的地理边界"""
        gt = ds.GetGeoTransform()
        ulx = gt[0]
        uly = gt[3]
        lrx = ulx + gt[1] * ds.RasterXSize
        lry = uly + gt[5] * ds.RasterYSize
        return [ulx, lry, lrx, uly]

    def update_status(self, message, error=False):
        def _update():
            if error:
                self.status_label.config(text=message, foreground="red")
                messagebox.showerror("错误", message)
            else:
                self.status_label.config(text=message, foreground="black")
            
            if "处理完成" in message or error:
                self.progress.stop()
                self.process_button.config(state=tk.NORMAL)

        self.master.after(0, _update)

    def show_completion_message(self):
        messagebox.showinfo("处理完成", "图像合并和处理已完成！")

    def show_help(self):
        help_text = """
        使用说明：
        1. 选择包含航拍图像的输入文件夹
        2. 选择输出文件位置
        3. 指定目标坐标系（例如：EPSG:4544）
        4. 设置目标分辨率（米）
        5. 选择是否应用色彩平衡
        6. 如果应用色彩平衡，设置低和高百分位数
        7. 点击"开始处理"按钮
        
        注意：处理大量或大尺寸图像可能需要较长时间。
        """
        messagebox.showinfo("帮助", help_text)

def log_error(message):
    """记录错误信息到日志文件"""
    with open("error_log.txt", "a") as log_file:
        log_file.write(f"{datetime.now()}: {message}\n")

if __name__ == "__main__":
    # 设置GDAL错误处理
    gdal.UseExceptions()
    gdal.PushErrorHandler('CPLQuietErrorHandler')

    try:
        root = tk.Tk()
        app = AerialImageMergerApp(root)
        root.mainloop()
    except Exception as e:
        print(f"程序运行出错: {str(e)}")
        messagebox.showerror("错误", f"程序运行出错: {str(e)}")
        log_error(f"程序运行出错: {str(e)}")
    finally:
        # 恢复GDAL错误处理
        gdal.PopErrorHandler()