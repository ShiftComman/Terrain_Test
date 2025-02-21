import os

def rename_tif_files(directory, target_str, replace_str="", to_lower=False, to_upper=False):
    """
    重命名目录中的所有tif文件
    
    参数:
        directory: 文件夹路径
        target_str: 要删除或替换的字符串
        replace_str: 替换后的字符串（默认为空，即删除）
        to_lower: 是否转换为小写
        to_upper: 是否转换为大写
    """
    # 确保目录存在
    if not os.path.exists(directory):
        print(f"目录不存在: {directory}")
        return
    
    # 获取所有tif文件
    files_renamed = 0
    for filename in os.listdir(directory):
        if filename.lower().endswith('.tif'):
            # 构建新文件名
            new_name = filename.replace(target_str, replace_str)
            
            # 处理大小写
            if to_lower:
                new_name = new_name.lower()
            elif to_upper:
                new_name = new_name.upper()
            
            # 如果文件名没有变化，跳过
            if new_name == filename:
                continue
            
            # 构建完整的文件路径
            old_path = os.path.join(directory, filename)
            new_path = os.path.join(directory, new_name)
            
            try:
                # 重命名文件
                os.rename(old_path, new_path)
                print(f"已重命名: {filename} -> {new_name}")
                files_renamed += 1
            except Exception as e:
                print(f"重命名 {filename} 时出错: {e}")
    
    print(f"\n共重命名了 {files_renamed} 个文件")

# 使用示例
if __name__ == "__main__":
    # 设置参数
    directory = r"G:\tif_features\county_feature\ky"  # 替换为你的文件夹路径
    target_str = "a_"                 # 要删除或替换的字符串
    replace_str = ""                # 替换后的字符串（如果只想删除，设为空字符串""）
    to_lower = True                     # 是否转换为小写
    to_upper = False                    # 是否转换为大写
    
    # 执行重命名
    rename_tif_files(directory, target_str, replace_str, to_lower, to_upper)