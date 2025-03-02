import os
import re
from argparse import ArgumentParser
import natsort

def rename_files_in_folder(folder_path):
    # 检查文件夹是否存在
    if not os.path.exists(folder_path):
        print(f"Folder not found: {folder_path}")
        return

    # 获取文件夹中的所有文件并排序
    files = natsort.natsorted(os.listdir(folder_path))
    # 遍历文件夹中的所有文件
    for index, file_name in enumerate(files):
        # 使用正则表达式匹配文件名
        match = re.match(r"(.+?)_(\d+)(\.\w+)", file_name)
        if match:
            # 提取文件名中的扩展名
            _, _, ext = match.groups()
            # 构造新的文件名，使用三位数格式
            new_name = f"{index:03d}{ext}"
            # 构造旧文件路径
            old_path = os.path.join(folder_path, file_name)
            # 构造新文件路径
            new_path = os.path.join(folder_path, new_name)

            if old_path != new_path:  # 避免重命名相同文件
                os.rename(old_path, new_path)
                print(f"Renamed: {old_path} -> {new_path}")

def main():
    # 创建一个ArgumentParser对象，用于解析命令行参数
    parser = ArgumentParser(description="Rename depth and RGB images in dataset.")
    # 添加一个命令行参数，用于指定数据集文件夹的路径
    parser.add_argument("--dataset_path", "-d", required=True, type=str, help="Path to dataset folder.")
    # 解析命令行参数
    args = parser.parse_args()

    # 获取RGB图像文件夹的路径
    rgb_folder = os.path.join(args.dataset_path, "images")
    pose_folder = os.path.join(args.dataset_path, "poses")
    # 重命名RGB图像文件夹中的文件
    rename_files_in_folder(rgb_folder)
    rename_files_in_folder(pose_folder)

if __name__ == "__main__":
    main()