import os
import sys
import glob
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from osgeo import gdal
import threading

# # 设置环境变量
# os.environ['GDAL_DATA'] = r'D:\Python38\Lib\site-packages\osgeo\data\gdal'
# os.environ['PROJ_LIB'] = r'D:\Python38\Lib\site-packages\fiona\proj_data'

# # 将 GDAL 目录添加到系统路径
# gdal_path = r'D:\Python38\Lib\site-packages\osgeo'
# if gdal_path not in sys.path:
#     sys.path.append(gdal_path)


class TIFMergerApp:
    def __init__(self, master):
        self.master = master
        master.title("TIF Merger Application")
        master.geometry("800x500")  # 减小窗口高度，因为不再需要预览区域

        # 设置样式
        self.style = ttk.Style()
        self.style.theme_use('clam')
        self.style.configure('TButton', padding=5, relief="flat", background="#4CAF50", foreground="white")
        self.style.map('TButton', background=[('active', '#45a049')])
        self.style.configure('TEntry', padding=5)
        self.style.configure('TProgressbar', thickness=20)

        # 创建主框架
        main_frame = ttk.Frame(master, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # 输入文件夹选择
        ttk.Label(main_frame, text="输入文件夹:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.input_folder = tk.StringVar()
        ttk.Entry(main_frame, textvariable=self.input_folder, width=50).grid(row=0, column=1, pady=5)
        ttk.Button(main_frame, text="浏览", command=self.browse_input_folder).grid(row=0, column=2, padx=5, pady=5)

        # 输出文件选择
        ttk.Label(main_frame, text="输出文件:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.output_file = tk.StringVar()
        ttk.Entry(main_frame, textvariable=self.output_file, width=50).grid(row=1, column=1, pady=5)
        ttk.Button(main_frame, text="浏览", command=self.browse_output_file).grid(row=1, column=2, padx=5, pady=5)

        # 目标坐标系选择
        ttk.Label(main_frame, text="目标坐标系:").grid(row=2, column=0, sticky=tk.W, pady=5)
        self.target_srs = tk.StringVar(value="EPSG:4544")
        ttk.Entry(main_frame, textvariable=self.target_srs, width=50).grid(row=2, column=1, pady=5)

        # 目标分辨率设置
        ttk.Label(main_frame, text="目标分辨率 (米):").grid(row=3, column=0, sticky=tk.W, pady=5)
        res_frame = ttk.Frame(main_frame)
        res_frame.grid(row=3, column=1, sticky=tk.W, pady=5)
        self.target_res_x = tk.DoubleVar(value=0.1)
        self.target_res_y = tk.DoubleVar(value=0.1)
        ttk.Entry(res_frame, textvariable=self.target_res_x, width=10).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Label(res_frame, text="x").pack(side=tk.LEFT, padx=(0, 5))
        ttk.Entry(res_frame, textvariable=self.target_res_y, width=10).pack(side=tk.LEFT, padx=(0, 5))

        # 处理按钮
        self.process_button = ttk.Button(main_frame, text="开始处理", command=self.start_processing)
        self.process_button.grid(row=4, column=0, columnspan=3, pady=10)

        # 进度条
        self.progress = ttk.Progressbar(main_frame, orient=tk.HORIZONTAL, length=300, mode='indeterminate')
        self.progress.grid(row=5, column=0, columnspan=3, pady=10)

        # 状态标签
        self.status_label = ttk.Label(main_frame, text="")
        self.status_label.grid(row=6, column=0, columnspan=3, pady=5)

    def browse_input_folder(self):
        folder = filedialog.askdirectory()
        if folder:
            self.input_folder.set(folder)

    def browse_output_file(self):
        file = filedialog.asksaveasfilename(defaultextension=".tif", filetypes=[("TIFF files", "*.tif")])
        if file:
            self.output_file.set(file)

    def start_processing(self):
        input_folder = self.input_folder.get()
        output_file = self.output_file.get()
        target_srs = self.target_srs.get()
        target_res = (self.target_res_x.get(), self.target_res_y.get())

        if not input_folder or not output_file:
            messagebox.showerror("错误", "请选择输入文件夹和输出文件")
            return

        self.process_button.config(state=tk.DISABLED)
        self.progress.start()
        self.status_label.config(text="处理中...")

        # 在新线程中运行处理过程
        threading.Thread(target=self.process_and_merge_images, args=(input_folder, output_file, target_srs, target_res), daemon=True).start()

    def process_and_merge_images(self, input_folder, output_file, target_srs, target_res):
        try:
            input_files = glob.glob(os.path.join(input_folder, '**', '*.tif'), recursive=True) + \
                          glob.glob(os.path.join(input_folder, '**', '*.ecw'), recursive=True)

            if not input_files:
                self.update_status("没有找到TIF或ECW文件", error=True)
                return

            processed_files = []
            nodata_value = None

            for i, input_file in enumerate(input_files):
                output_file_temp = os.path.splitext(input_file)[0] + '_processed.tif'
                processed, file_nodata = self.process_file(input_file, output_file_temp, target_srs, target_res)
                if processed:
                    processed_files.append(processed)
                    if nodata_value is None:
                        nodata_value = file_nodata
                self.update_status(f"处理进度: {i+1}/{len(input_files)}")

            if not processed_files:
                self.update_status("没有成功处理的文件", error=True)
                return

            self.update_status("开始合并处理后的文件")
            options = gdal.WarpOptions(
                format='GTiff',
                dstSRS=target_srs,
                xRes=target_res[0], yRes=target_res[1],
                srcNodata=nodata_value,
                dstNodata=nodata_value,
                multithread=True,
                resampleAlg=gdal.GRA_Bilinear,
                creationOptions=['COMPRESS=LZW', 'TILED=YES', 'BLOCKXSIZE=256', 'BLOCKYSIZE=256', 'BIGTIFF=YES'],
                warpOptions=['CUTLINE_ALL_TOUCHED=TRUE'],
                warpMemoryLimit=4096
            )
            
            gdal.Warp(output_file, processed_files, options=options)
            self.update_status(f"成功合并文件，输出: {output_file}")

            for file in processed_files:
                try:
                    os.remove(file)
                except Exception as e:
                    self.update_status(f"删除临时文件 {file} 时出错: {str(e)}", error=True)

            self.update_status("处理完成")
        except Exception as e:
            self.update_status(f"处理过程中出错: {str(e)}", error=True)

    def process_file(self, input_file, output_file, target_srs, target_res):
        try:
            if os.path.exists(output_file):
                self.update_status(f"输出文件已存在，跳过处理: {output_file}")
                return output_file, None

            ds = gdal.Open(input_file)
            if ds is None:
                self.update_status(f"无法打开文件: {input_file}", error=True)
                return None, None

            # 获取输入图像的 NODATA 值
            band = ds.GetRasterBand(1)
            nodata_value = band.GetNoDataValue()
            if nodata_value is None:
                nodata_value = 0  # 如果没有设置 NODATA 值，使用 0

            options = gdal.WarpOptions(
                dstSRS=target_srs,
                xRes=target_res[0], yRes=target_res[1],
                srcNodata=nodata_value,
                dstNodata=nodata_value,
                resampleAlg=gdal.GRA_Bilinear,
                format='GTiff',
                creationOptions=['COMPRESS=LZW', 'TILED=YES', 'BLOCKXSIZE=256', 'BLOCKYSIZE=256', 'BIGTIFF=YES'],
                warpOptions=['CUTLINE_ALL_TOUCHED=TRUE'],
                multithread=True,
                warpMemoryLimit=4096
            )

            self.update_status(f"正在处理文件: {input_file}")
            gdal.Warp(output_file, ds, options=options)
            
            ds = None
            self.update_status(f"成功处理文件: {input_file}")
            return output_file, nodata_value

        except Exception as e:
            self.update_status(f"处理文件 {input_file} 时出错: {str(e)}", error=True)
            return None, None
    
    def update_status(self, message, error=False, show_preview=False):
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

if __name__ == "__main__":
    root = tk.Tk()
    app = TIFMergerApp(root)
    root.mainloop()