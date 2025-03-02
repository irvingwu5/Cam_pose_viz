import os
import re
from argparse import ArgumentParser
import natsort

def rename_files_in_folder(folder_path):
    if not os.path.exists(folder_path):
        print(f"Folder not found: {folder_path}")
        return

    files = natsort.natsorted(os.listdir(folder_path))
    for index, file_name in enumerate(files):
        match = re.match(r"(.+?)_(\d+)(\.\w+)", file_name)
        if match:
            _, _, ext = match.groups()
            new_name = f"{index:03d}{ext}"
            old_path = os.path.join(folder_path, file_name)
            new_path = os.path.join(folder_path, new_name)

            if old_path != new_path: 
                os.rename(old_path, new_path)
                print(f"Renamed: {old_path} -> {new_path}")

def main():
    parser = ArgumentParser(description="Rename depth and RGB images in dataset.")
    parser.add_argument("--dataset_path", "-d", required=True, type=str, help="Path to dataset folder.")
    args = parser.parse_args()

    rgb_folder = os.path.join(args.dataset_path, "images")
    pose_folder = os.path.join(args.dataset_path, "poses")
    rename_files_in_folder(rgb_folder)
    rename_files_in_folder(pose_folder)

if __name__ == "__main__":
    main()
