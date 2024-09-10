import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import os
import joblib
import threading
import queue
import logging
from logging.handlers import RotatingFileHandler
import rasterio
import geopandas as gpd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
from skimage.feature import graycomatrix, graycoprops
from skimage.color import rgb2hsv
import fiona
from scipy.stats import randint, uniform

# 配置日志记录
def setup_logging():
    log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    log_file = 'crop_classification.log'
    log_handler = RotatingFileHandler(log_file, maxBytes=1024 * 1024, backupCount=5)
    log_handler.setFormatter(log_formatter)
    log_handler.setLevel(logging.INFO)

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(log_handler)

setup_logging()

fiona.supported_drivers['ESRI Shapefile'] = 'rw'

def validate_shp_data(gdf, column_name='ZZZW'):
    """
    验证SHP数据
    """
    if column_name not in gdf.columns:
        raise ValueError(f"SHP文件中缺少'{column_name}'列")
    
    if gdf[column_name].isnull().any():
        raise ValueError(f"'{column_name}'列中存在空值")

def extract_features(image_path, gdf, progress_callback=None):
    features = []
    valid_geometries = []
    invalid_geometries = []
    total_geometries = len(gdf)

    with rasterio.open(image_path) as src:
        for idx, geometry in enumerate(gdf.geometry):
            try:
                window = rasterio.windows.from_bounds(*geometry.bounds, src.transform)
                masked_image = src.read(window=window, indexes=[1, 2, 3])
                
                if masked_image.shape[0] < 3 or masked_image.size == 0:
                    logging.warning(f"几何体 {idx}: 无数据")
                    invalid_geometries.append(idx)
                    continue
                
                rgb_means = np.nanmean(masked_image, axis=(1, 2))
                rgb_stds = np.nanstd(masked_image, axis=(1, 2))
                
                r, g, b = rgb_means
                exg = 2 * g - r - b
                vari = (g - r) / (g + r - b + 1e-8)
                
                hsv_image = rgb2hsv(np.moveaxis(masked_image, 0, -1))
                hsv_means = np.nanmean(hsv_image, axis=(0, 1))
                
                green_channel = masked_image[1].astype(np.uint8)
                if green_channel.size > 0:
                    glcm = graycomatrix(green_channel, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
                    contrast = graycoprops(glcm, 'contrast')[0, 0]
                    dissimilarity = graycoprops(glcm, 'dissimilarity')[0, 0]
                    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
                    energy = graycoprops(glcm, 'energy')[0, 0]
                    correlation = graycoprops(glcm, 'correlation')[0, 0]
                else:
                    contrast = dissimilarity = homogeneity = energy = correlation = 0
                
                feature = np.concatenate([rgb_means, rgb_stds, [exg, vari], hsv_means, 
                                          [contrast, dissimilarity, homogeneity, energy, correlation]])
                features.append(feature)
                valid_geometries.append(idx)

                if progress_callback:
                    progress_callback((idx + 1) / total_geometries * 100)
            except Exception as e:
                logging.error(f"处理几何体 {idx} 时出错: {str(e)}")
                invalid_geometries.append(idx)
    
    if not features:
        raise ValueError("没有成功提取到任何特征")
    
    return np.array(features), valid_geometries, invalid_geometries
    
def update_or_train_model(image_path, train_shp_path, model_output_path, le_output_path, data_output_path, force_new_model=False, progress_callback=None, hyperparameters=None, cancel_check=None):
    logging.info(f"开始{'创建新模型' if force_new_model else '更新模型'}")
    gdf_train = gpd.read_file(train_shp_path, encoding='utf-8')
    
    # 验证SHP数据
    validate_shp_data(gdf_train)
    
    X_new, valid_indices, _ = extract_features(image_path, gdf_train, progress_callback=progress_callback)
    gdf_train = gdf_train.iloc[valid_indices]
    
    le = LabelEncoder()
    y_new = le.fit_transform(gdf_train['ZZZW'])
    
    if os.path.exists(model_output_path) and os.path.exists(le_output_path) and os.path.exists(data_output_path) and not force_new_model:
        logging.info("加载现有模型并进行更新")
        clf = joblib.load(model_output_path)
        X_old, y_old = joblib.load(data_output_path)
        
        if X_old.shape[1] != X_new.shape[1]:
            logging.warning(f"新旧数据的特征数量不一致。旧数据：{X_old.shape[1]}，新数据：{X_new.shape[1]}")
            logging.info("将重新训练模型")
            X_combined, y_combined = X_new, y_new
        else:
            X_combined = np.vstack((X_old, X_new))
            y_combined = np.concatenate((y_old, y_new))
    else:
        logging.info("创建新模型")
        X_combined, y_combined = X_new, y_new

    # 定义随机搜索的参数范围
    if hyperparameters is None:
        hyperparameters = {
            'n_estimators': randint(10, 1000),
            'max_depth': randint(2, 50),
            'min_samples_split': randint(2, 50),
            'min_samples_leaf': randint(1, 20),
            'max_features': uniform(0.1, 0.9)
        }

    # 创建随机搜索对象
    random_search = RandomizedSearchCV(
        RandomForestClassifier(random_state=42),
        param_distributions=hyperparameters,
        n_iter=50,
        cv=5,
        random_state=42,
        n_jobs=-1
    )

    # 执行随机搜索
    random_search.fit(X_combined, y_combined)

    # 获取最佳模型
    clf = random_search.best_estimator_

    # 评估模型
    y_pred = clf.predict(X_combined)
    accuracy = accuracy_score(y_combined, y_pred)
    report = classification_report(y_combined, y_pred, target_names=le.classes_, output_dict=True)

    # 保存模型和数据
    joblib.dump(clf, model_output_path)
    joblib.dump(le, le_output_path)
    joblib.dump((X_combined, y_combined), data_output_path)

    logging.info(f"模型已保存到: {model_output_path}")
    logging.info(f"标签编码器已保存到: {le_output_path}")
    logging.info(f"训练数据已保存到: {data_output_path}")
    logging.info(f"最佳参数: {random_search.best_params_}")
    logging.info(f"模型整体精度: {accuracy}")
    logging.info("各类别精度:")
    for class_name, metrics in report.items():
        if isinstance(metrics, dict):
            logging.info(f"{class_name}: 精度 = {metrics['precision']:.2f}, 召回率 = {metrics['recall']:.2f}, F1分数 = {metrics['f1-score']:.2f}")

    if cancel_check and cancel_check():
        raise InterruptedError("操作被用户取消")

    return accuracy, report

def predict_new_data(model_path, le_path, new_image_path, new_shp_path, output_shp_path, progress_callback=None, cancel_check=None):
    logging.info("开始预测新数据")
    clf = joblib.load(model_path)
    le = joblib.load(le_path)

    gdf_new = gpd.read_file(new_shp_path, encoding='utf-8')
    X_new, valid_indices, invalid_indices = extract_features(new_image_path, gdf_new,progress_callback=progress_callback)
    
    if X_new.shape[1] != clf.n_features_in_:
        raise ValueError(f"错误：特征数量不匹配。模型期望 {clf.n_features_in_} 个特征，但提供了 {X_new.shape[1]} 个特征。")
    
    y_pred = clf.predict(X_new)
    y_proba = clf.predict_proba(X_new)
    
    # 为有效的几何体添加预测结果
    gdf_new.loc[valid_indices, 'ZZZW'] = le.inverse_transform(y_pred)
    gdf_new.loc[valid_indices, 'ZZZW_proba'] = np.max(y_proba, axis=1)
    
    # 为无效的几何体添加标记
    gdf_new.loc[invalid_indices, 'ZZZW'] = 'Invalid'
    gdf_new.loc[invalid_indices, 'ZZZW_proba'] = 0
    
    gdf_new.to_file(output_shp_path, encoding='utf-8')

    if cancel_check and cancel_check():
        raise InterruptedError("操作被用户取消")

    logging.info(f"预测结果已保存到: {output_shp_path}")
    return len(invalid_indices)

class CropClassificationApp:
    def __init__(self, master):
        self.master = master
        master.title("RGB分类模型")
        master.geometry("800x600")

        # 初始化所有需要的属性
        self.tif_path = tk.StringVar()
        self.shp_path = tk.StringVar()
        self.model_path = tk.StringVar(value=os.getcwd())
        self.model_action = tk.StringVar(value="update")
        self.predict_tif_path = tk.StringVar()
        self.predict_shp_path = tk.StringVar()
        self.predict_model_path = tk.StringVar(value=os.getcwd())
        self.output_path = tk.StringVar()

        # 添加这些行来初始化超参数变量
        self.n_estimators_min = tk.IntVar(value=50)
        self.n_estimators_max = tk.IntVar(value=300)
        self.max_depth_min = tk.IntVar(value=5)
        self.max_depth_max = tk.IntVar(value=50)

        self.create_widgets()
        self.create_menu()

        self.progress_queue = queue.Queue()
        self.master.after(100, self.check_progress_queue)

        self.cancel_operation = False

    def create_widgets(self):
        self.notebook = ttk.Notebook(self.master)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.train_tab = ttk.Frame(self.notebook)
        self.predict_tab = ttk.Frame(self.notebook)
        self.info_tab = ttk.Frame(self.notebook)

        self.notebook.add(self.train_tab, text="训练模型")
        self.notebook.add(self.predict_tab, text="模型预测")
        self.notebook.add(self.info_tab, text="模型信息")

        self.setup_train_tab()
        self.setup_predict_tab()
        self.setup_info_tab()

        self.status_bar = ttk.Label(self.master, text="就绪", relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

        self.progress = ttk.Progressbar(self.master, length=780, mode='determinate')
        self.progress.pack(pady=10)

        self.log_text = scrolledtext.ScrolledText(self.master, wrap=tk.WORD, width=95, height=10)
        self.log_text.pack(padx=10, pady=10)

    def create_menu(self):
        menubar = tk.Menu(self.master)
        self.master.config(menu=menubar)

        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="文件", menu=file_menu)
        file_menu.add_command(label="退出", command=self.master.quit)

        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="帮助", menu=help_menu)
        help_menu.add_command(label="关于", command=self.show_about)

    def setup_train_tab(self):
        ttk.Label(self.train_tab, text="选择TIF文件:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        self.tif_path = tk.StringVar()
        tif_entry = ttk.Entry(self.train_tab, textvariable=self.tif_path, width=50)
        tif_entry.grid(row=0, column=1, padx=5, pady=5)
        ttk.Button(self.train_tab, text="浏览", command=lambda: self.browse_file(self.tif_path, [("TIF files", "*.tif")])).grid(row=0, column=2, padx=5, pady=5)
        ttk.Button(self.train_tab, text="预览", command=lambda: self.preview_file(self.tif_path)).grid(row=0, column=3, padx=5, pady=5)

        ttk.Label(self.train_tab, text="选择带有标签的SHP文件:").grid(row=1, column=0, sticky="w", padx=5, pady=5)
        self.shp_path = tk.StringVar()
        shp_entry = ttk.Entry(self.train_tab, textvariable=self.shp_path, width=50)
        shp_entry.grid(row=1, column=1, padx=5, pady=5)
        ttk.Button(self.train_tab, text="浏览", command=lambda: self.browse_file(self.shp_path, [("SHP files", "*.shp")])).grid(row=1, column=2, padx=5, pady=5)
        ttk.Button(self.train_tab, text="预览", command=lambda: self.preview_file(self.shp_path)).grid(row=1, column=3, padx=5, pady=5)

        ttk.Label(self.train_tab, text="模型存储路径:").grid(row=2, column=0, sticky="w", padx=5, pady=5)
        self.model_path = tk.StringVar(value=os.getcwd())
        ttk.Entry(self.train_tab, textvariable=self.model_path, width=50).grid(row=2, column=1, padx=5, pady=5)
        ttk.Button(self.train_tab, text="浏览", command=self.browse_model_path).grid(row=2, column=2, padx=5, pady=5)

        self.model_action = tk.StringVar(value="update")
        ttk.Radiobutton(self.train_tab, text="更新现有模型", variable=self.model_action, value="update").grid(row=3, column=0, sticky="w", padx=5, pady=5)
        ttk.Radiobutton(self.train_tab, text="创建新模型", variable=self.model_action, value="new").grid(row=3, column=1, sticky="w", padx=5, pady=5)

        # 添加超参数设置控件
        ttk.Label(self.train_tab, text="n_estimators:").grid(row=4, column=0, sticky="w", padx=5, pady=5)
        ttk.Entry(self.train_tab, textvariable=self.n_estimators_min, width=5).grid(row=4, column=1, sticky="w", padx=5, pady=5)
        ttk.Label(self.train_tab, text="-").grid(row=4, column=1)
        ttk.Entry(self.train_tab, textvariable=self.n_estimators_max, width=5).grid(row=4, column=1, sticky="e", padx=5, pady=5)

        ttk.Label(self.train_tab, text="max_depth:").grid(row=5, column=0, sticky="w", padx=5, pady=5)
        ttk.Entry(self.train_tab, textvariable=self.max_depth_min, width=5).grid(row=5, column=1, sticky="w", padx=5, pady=5)
        ttk.Label(self.train_tab, text="-").grid(row=5, column=1)
        ttk.Entry(self.train_tab, textvariable=self.max_depth_max, width=5).grid(row=5, column=1, sticky="e", padx=5, pady=5)
        

        self.train_button = ttk.Button(self.train_tab, text="开始训练", command=self.start_training)
        self.train_button.grid(row=6, column=0, columnspan=3, pady=20)

    def setup_predict_tab(self):
        ttk.Label(self.predict_tab, text="选择TIF文件:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        self.predict_tif_path = tk.StringVar()
        predict_tif_entry = ttk.Entry(self.predict_tab, textvariable=self.predict_tif_path, width=50)
        predict_tif_entry.grid(row=0, column=1, padx=5, pady=5)
        ttk.Button(self.predict_tab, text="浏览", command=lambda: self.browse_file(self.predict_tif_path, [("TIF files", "*.tif")])).grid(row=0, column=2, padx=5, pady=5)
        ttk.Button(self.predict_tab, text="预览", command=lambda: self.preview_file(self.predict_tif_path)).grid(row=0, column=3, padx=5, pady=5)

        ttk.Label(self.predict_tab, text="选择SHP文件:").grid(row=1, column=0, sticky="w", padx=5, pady=5)
        self.predict_shp_path = tk.StringVar()
        predict_shp_entry = ttk.Entry(self.predict_tab, textvariable=self.predict_shp_path, width=50)
        predict_shp_entry.grid(row=1, column=1, padx=5, pady=5)
        ttk.Button(self.predict_tab, text="浏览", command=lambda: self.browse_file(self.predict_shp_path, [("SHP files", "*.shp")])).grid(row=1, column=2, padx=5, pady=5)
        ttk.Button(self.predict_tab, text="预览", command=lambda: self.preview_file(self.predict_shp_path)).grid(row=1, column=3, padx=5, pady=5)

        ttk.Label(self.predict_tab, text="模型路径:").grid(row=2, column=0, sticky="w", padx=5, pady=5)
        self.predict_model_path = tk.StringVar(value=os.getcwd())
        ttk.Entry(self.predict_tab, textvariable=self.predict_model_path, width=50).grid(row=2, column=1, padx=5, pady=5)
        ttk.Button(self.predict_tab, text="浏览", command=self.browse_predict_model_path).grid(row=2, column=2, padx=5, pady=5)

        ttk.Label(self.predict_tab, text="输出文件路径:").grid(row=3, column=0, sticky="w", padx=5, pady=5)
        self.output_path = tk.StringVar()
        ttk.Entry(self.predict_tab, textvariable=self.output_path, width=50).grid(row=3, column=1, padx=5, pady=5)
        ttk.Button(self.predict_tab, text="浏览", command=lambda: self.browse_save_file(self.output_path, [("SHP files", "*.shp")])).grid(row=3, column=2, padx=5, pady=5)


        self.predict_button = ttk.Button(self.predict_tab, text="开始预测", command=self.start_prediction)
        self.predict_button.grid(row=6, column=0, columnspan=3, pady=20)

    def setup_info_tab(self):
        self.info_text = scrolledtext.ScrolledText(self.info_tab, wrap=tk.WORD, width=90, height=20)
        self.info_text.pack(padx=10, pady=10)

        ttk.Button(self.info_tab, text="获取模型信息", command=self.refresh_info).pack(pady=10)

    def browse_file(self, path_var, file_types):
        filename = filedialog.askopenfilename(filetypes=file_types)
        if filename:
            path_var.set(filename)

    def browse_save_file(self, path_var, file_types):
        filename = filedialog.asksaveasfilename(filetypes=file_types, defaultextension=file_types[0][1])
        if filename:
            path_var.set(filename)

    def browse_model_path(self):
        path = filedialog.askdirectory()
        if path:
            self.model_path.set(path)

    def browse_predict_model_path(self):
        path = filedialog.askdirectory()
        if path:
            self.predict_model_path.set(path)

    def toggle_chunk_size(self):
        state = 'normal' if self.use_chunking.get() else 'disabled'
        self.chunk_size_label['state'] = state
        self.chunk_size_entry['state'] = state
        self.chunk_size_label_predict['state'] = state
        self.chunk_size_entry_predict['state'] = state

    def preview_file(self, path_var):
        file_path = path_var.get()
        if not file_path:
            messagebox.showwarning("警告", "请先选择文件")
            return
        
        try:
            info = self.get_file_info(file_path)
            messagebox.showinfo("文件预览", info)
        except Exception as e:
            messagebox.showerror("错误", f"无法预览文件：{str(e)}")

    @staticmethod
    def get_file_info(file_path):
        file_size = os.path.getsize(file_path) / (1024 * 1024)  # Size in MB
        file_type = os.path.splitext(file_path)[1].lower()
        
        info = f"文件路径: {file_path}\n"
        info += f"文件大小: {file_size:.2f} MB\n"
        info += f"文件类型: {file_type}\n"
        
        if file_type == '.tif':
            with rasterio.open(file_path) as src:
                info += f"图像大小: {src.width} x {src.height}\n"
                info += f"波段数: {src.count}\n"
                info += f"坐标系统: {src.crs}\n"
        elif file_type == '.shp':
            gdf = gpd.read_file(file_path)
            info += f"要素数量: {len(gdf)}\n"
            info += f"几何类型: {gdf.geom_type.iloc[0]}\n"
            info += f"属性列: {', '.join(gdf.columns)}\n"
        
        return info

    def start_training(self):
        errors = self.validate_inputs()
        if errors:
            messagebox.showerror("输入错误", "\n".join(errors))
            return

        self.disable_buttons()
        self.progress['value'] = 0
        self.status_bar['text'] = "训练中..."
        self.log_text.insert(tk.END, "开始训练...\n")
        self.cancel_operation = False

        tif_file = self.tif_path.get()
        shp_file = self.shp_path.get()
        model_dir = self.model_path.get()
        force_new = self.model_action.get() == "new"
        # 添加超参数设置
        hyperparameters = {
            'n_estimators': randint(self.n_estimators_min.get(), self.n_estimators_max.get()),
            'max_depth': randint(self.max_depth_min.get(), self.max_depth_max.get()),
            'min_samples_split': randint(2, 20),
            'min_samples_leaf': randint(1, 10),
            'max_features': uniform(0.1, 0.9)
        }
        def training_thread():
            try:
                accuracy, report = update_or_train_model(
                    tif_file, shp_file, 
                    os.path.join(model_dir, "model.joblib"), 
                    os.path.join(model_dir, "label_encoder.joblib"), 
                    os.path.join(model_dir, "training_data.joblib"), 
                    force_new,
                    self.update_progress,
                     hyperparameters=hyperparameters,
                    cancel_check=lambda: self.cancel_operation
                )
                if not self.cancel_operation:
                    self.master.after(0, lambda: self.training_complete(accuracy, report))
                else:
                    self.master.after(0, self.operation_cancelled)
            except Exception as e:
                logging.exception("训练过程中出错")
                error_message = str(e)
                self.master.after(0, lambda: self.training_error(error_message))

        self.current_thread = threading.Thread(target=training_thread)
        self.current_thread.start()

        # 添加取消按钮
        self.cancel_button = ttk.Button(self.train_tab, text="取消", command=self.cancel_current_operation)
        self.cancel_button.grid(row=7, column=0, columnspan=3, pady=10)

    def start_prediction(self):
        errors = self.validate_inputs()
        if errors:
            messagebox.showerror("输入错误", "\n".join(errors))
            return

        self.disable_buttons()
        self.progress['value'] = 0
        self.status_bar['text'] = "预测中..."
        self.log_text.insert(tk.END, "开始预测...\n")
        self.cancel_operation = False

        tif_file = self.predict_tif_path.get()
        shp_file = self.predict_shp_path.get()
        model_dir = self.predict_model_path.get()
        output_file = self.output_path.get()
        # 添加文件存在性检查
        if not self.check_file_exists(tif_file):
            self.prediction_error(f"TIF文件不存在: {tif_file}")
            return
        if not self.check_file_exists(shp_file):
            self.prediction_error(f"SHP文件不存在: {shp_file}")
            return
        if not self.check_file_exists(os.path.join(model_dir, "model.joblib")):
            self.prediction_error(f"模型文件不存在: {os.path.join(model_dir, 'model.joblib')}")
            return
        def prediction_thread():
            try:
                invalid_count = predict_new_data(
                    os.path.join(model_dir, "model.joblib"), 
                    os.path.join(model_dir, "label_encoder.joblib"), 
                    tif_file, shp_file, output_file,
                    self.update_progress,
                    cancel_check=lambda: self.cancel_operation
                )
                if not self.cancel_operation:
                    self.master.after(0, lambda: self.prediction_complete(invalid_count))
                else:
                    self.master.after(0, self.operation_cancelled)
            except Exception as e:
                logging.exception("预测过程中出错")
                self.master.after(0, lambda: self.prediction_error(str(e)))

        self.current_thread = threading.Thread(target=prediction_thread)
        self.current_thread.start()

        # 添加取消按钮
        self.cancel_button = ttk.Button(self.predict_tab, text="取消", command=self.cancel_current_operation)
        self.cancel_button.grid(row=7, column=0, columnspan=3, pady=10)
    def check_file_exists(self, file_path):
        return os.path.exists(file_path)
    def update_progress(self, value):
        self.progress_queue.put(value)

    def check_progress_queue(self):
        try:
            while True:
                value = self.progress_queue.get_nowait()
                self.progress['value'] = value
        except queue.Empty:
            pass
        finally:
            self.master.after(100, self.check_progress_queue)

    def training_complete(self, accuracy, report):
        self.enable_buttons()
        self.status_bar['text'] = "训练完成"
        action_text = "创建新模型" if self.model_action.get() == "new" else "更新现有模型"
        self.log_text.insert(tk.END, f"{action_text}完成\n")
        self.log_text.insert(tk.END, f"模型整体精度: {accuracy:.4f}\n")
        self.log_text.insert(tk.END, "各类别精度:\n")
        for class_name, metrics in report.items():
            if isinstance(metrics, dict):
                self.log_text.insert(tk.END, f"{class_name}: 精度 = {metrics['precision']:.2f}, 召回率 = {metrics['recall']:.2f}, F1分数 = {metrics['f1-score']:.2f}\n")
        messagebox.showinfo("成功", f"{action_text}完成\n模型整体精度: {accuracy:.4f}")
        self.refresh_info()
        if hasattr(self, 'cancel_button'):
            self.cancel_button.destroy()

    def prediction_complete(self, invalid_count):
        self.enable_buttons()
        self.status_bar['text'] = "预测完成"
        self.log_text.insert(tk.END, "预测完成\n")
        if invalid_count > 0:
            self.log_text.insert(tk.END, f"警告：{invalid_count}个几何体无法进行预测，已在输出中标记为'Invalid'\n")
        messagebox.showinfo("成功", f"预测完成\n{invalid_count}个几何体无法预测")
        if hasattr(self, 'cancel_button'):
            self.cancel_button.destroy()

    def training_error(self, error_message):
        self.enable_buttons()
        self.status_bar['text'] = "训练出错"
        self.log_text.insert(tk.END, f"训练过程中出错：{error_message}\n")
        messagebox.showerror("错误", f"训练过程中出错：{error_message}")
        if hasattr(self, 'cancel_button'):
            self.cancel_button.destroy()

    def prediction_error(self, error_message):
        self.enable_buttons()
        self.status_bar['text'] = "预测出错"
        self.log_text.insert(tk.END, f"预测过程中出错：{error_message}\n")
        messagebox.showerror("错误", f"预测过程中出错：{error_message}")
        if hasattr(self, 'cancel_button'):
            self.cancel_button.destroy()

    def refresh_info(self):
        self.info_text.delete('1.0', tk.END)
        model_dir = self.model_path.get()
        try:
            model = joblib.load(os.path.join(model_dir, "model.joblib"))
            le = joblib.load(os.path.join(model_dir, "label_encoder.joblib"))
            
            info = f"模型信息：\n"
            info += f"模型存储路径：{model_dir}\n"
            info += f"特征数量：{model.n_features_in_}\n"
            info += f"类别：{', '.join(le.classes_)}\n"
            info += f"树的数量：{model.n_estimators}\n"
            info += f"最大深度：{model.max_depth}\n"
            info += f"最小分裂样本数：{model.min_samples_split}\n"
            info += f"最小叶子节点样本数：{model.min_samples_leaf}\n"
            info += f"特征选择方式：{model.max_features}\n"
            
            # 尝试加载训练数据并获取样本数量
            try:
                X, y = joblib.load(os.path.join(model_dir, "training_data.joblib"))
                info += f"参训练样本数量：{len(X)}\n"
            except Exception as e:
                info += f"无法加载训练数据信息：{str(e)}\n"
            
            self.info_text.insert(tk.END, info)
        except Exception as e:
            self.info_text.insert(tk.END, f"无法加载模型信息：{str(e)}")

    def disable_buttons(self):
        self.train_button['state'] = 'disabled'
        self.predict_button['state'] = 'disabled'

    def enable_buttons(self):
        self.train_button['state'] = 'normal'
        self.predict_button['state'] = 'normal'

    def show_about(self):
        messagebox.showinfo("关于", "RGB分类模型 v1.0\n\n作者：AI 贵州雏阳\n\n版权所有 © 2024")

    def cancel_current_operation(self):
        self.cancel_operation = True
        self.log_text.insert(tk.END, "正在取消操作...\n")
        self.status_bar['text'] = "正在取消..."

    def operation_cancelled(self):
        self.enable_buttons()
        self.status_bar['text'] = "操作已取消"
        self.log_text.insert(tk.END, "操作已取消\n")
        messagebox.showinfo("已取消", "操作已被用户取消")
        if hasattr(self, 'cancel_button'):
            self.cancel_button.destroy()

    def validate_inputs(self):
        errors = []
        if self.notebook.index(self.notebook.select()) == 0:  # Training tab
            if not self.tif_path.get():
                errors.append("请选择训练用TIF文件")
            if not self.shp_path.get():
                errors.append("请选择训练用SHP文件")
        else:  # Prediction tab
            if not self.predict_tif_path.get():
                errors.append("请选择预测用TIF文件")
            if not self.predict_shp_path.get():
                errors.append("请选择预测用SHP文件")
            if not self.predict_model_path.get():
                errors.append("请选择模型路径")
            if not self.output_path.get():
                errors.append("请选择输出文件路径")
        return errors

def main():
    root = tk.Tk()
    app = CropClassificationApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()