import argparse
import os

from src.loader import load_quick, load_nerf, load_colmap
from src.utils import load_image, rescale_cameras, recenter_cameras
from src.visualizer import CameraVisualizer


parser = argparse.ArgumentParser()
parser.add_argument('--root', type=str)
parser.add_argument('--format', default='quick', choices=['quick', 'nerf', 'colmap'])
parser.add_argument('--type', default=None, choices=[None, 'sph', 'xyz', 'elu', 'c2w', 'w2c'])
parser.add_argument('--no_images', action='store_true')
parser.add_argument('--mesh_path', type=str, default=None)
parser.add_argument('--image_size', type=int, default=256)
parser.add_argument('--scene_size', type=int, default=5)
parser.add_argument('--y_up', action='store_true')
parser.add_argument('--recenter', action='store_true')
parser.add_argument('--rescale', type=float, default=None)
parser.add_argument('--ply_path', type=str, default=None)

args = parser.parse_args()

root_path = args.root

poses = []
legends = []
colors = []
images = None

if args.format == 'quick':
    poses, legends, colors, image_paths = load_quick(root_path, args.type)

elif args.format == 'nerf':
    poses, legends, colors, image_paths = load_nerf(root_path)

elif args.format == 'colmap':
    poses, legends, colors, image_paths = load_colmap(root_path)

if args.recenter:
    poses = recenter_cameras(poses)

if args.rescale is not None:
    poses = rescale_cameras(poses, args.rescale)

if args.y_up:
    for i in range(0, len(poses)):
        poses[i] = poses[i][[0, 2, 1, 3]]
        poses[i][1, :] *= -1
    
if not args.no_images:
    images = []
    for fpath in image_paths:  # ['inputs/quick/cam_c2w/images/011.png', 'inputs/quick/cam_c2w/images/077.png', 'inputs/quick/cam_c2w/images/001.png', 'inputs/quick/cam_c2w/images/065.png']
        if fpath is None:
            images.append(None)
            continue

        if not os.path.exists(fpath):
            images.append(None)
            print(f'Image not found at {fpath}')
            continue

        images.append(load_image(fpath, sz=args.image_size))  # 将图片缩放大小后转换为numpy数组，并返回前三个通道，添加到images列表中


viz = CameraVisualizer(poses, legends, colors, images=images, ply_path=args.ply_path)  # 所有poses矩阵array列表、['011.png', '077.png', '001.png', '065.png']、['blue', 'blue', 'blue', 'blue']、所有images矩阵array列表
fig = viz.update_figure(args.scene_size, base_radius=1, zoom_scale=0.5, show_grid=True, show_ticklabels=True, show_background=True, y_up=args.y_up)

fig.show()
