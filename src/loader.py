import os
import numpy as np
import json

from .utils import elu_to_c2w, spherical_to_cartesian, load_image, qvec_to_rotmat, rotmat


def load_quick(root_path, type):

    # 初始化变量
    poses = []
    legends = []
    colors = []
    image_paths = []

    # 如果type为None，则从poses.json文件中加载pose
    if type is None:
        pose_path = os.path.join(root_path, 'poses.json')
        print(f'Load poses from {pose_path}')
        with open(pose_path, 'r') as fin:
            jdata = json.load(fin)
        type = jdata['type']
        frame_list = jdata['frames']
    else:
        # 否则从poses文件夹中加载pose
        pose_root = os.path.join(root_path, 'poses')
        print(f'Load poses from {pose_root}')
        frame_list = os.listdir(pose_root)  # 获取pose文件列表['011.npy', '077.npy', '001.npy', '065.npy']

    # 加载图像
    image_root = os.path.join(root_path, 'images')
    print(f'Load images from {image_root}')

    # 遍历frame_list
    for idx, frame in enumerate(frame_list):

        fid = idx

        # 如果frame是字符串类型
        if isinstance(frame, str):

            fname = frame  # '011.npy'
            vals = fname.split('.')  # ['011', 'npy']
            fid, ext = vals[0], vals[-1]  # fid='011', ext='npy'

            fpath = os.path.join(pose_root, fname)  # 'inputs/quick/cam_c2w/poses/011.npy'

            # 根据文件后缀加载pose
            if ext == 'npy':
                mat = np.load(fpath)
            elif ext == 'txt':
                mat = np.loadtxt(fpath)

            # 加载图像路径
            img_paths = [os.path.join(image_root, f'{fid}.{ext}') for ext in ['png', 'jpg', 'jpeg']]  # ['inputs/quick/cam_c2w/images/011.png', 'inputs/quick/cam_c2w/images/011.jpg', 'inputs/quick/cam_c2w/images/011.jpeg']
            img_paths = [fpath for fpath in img_paths if os.path.exists(fpath)]   # ['inputs/quick/cam_c2w/images/011.png']

            img_path = img_paths[0] if len(img_paths) > 0 else None  # 'inputs/quick/cam_c2w/images/011.png'

        # 如果frame是字典类型
        elif isinstance(frame, dict):

            # 如果字典中有image_name键，则加载图像路径
            if 'image_name' in frame and frame['image_name']:
                fname = frame['image_name']
                img_path = os.path.join(image_root, fname)
            else:
                img_path = None

            # 加载pose
            mat = np.array(frame['pose'])

        # 根据type加载pose
        if type == 'c2w':
            c2w = mat
            if c2w.shape[0] == 3:
                c2w = np.concatenate([c2w, np.zeros((1, 4))], axis=0)
                c2w[-1, -1] = 1

        if type == 'w2c':
            w2c = mat
            if w2c.shape[0] == 3:
                w2c = np.concatenate([w2c, np.zeros((1, 4))], axis=0)
                w2c[-1, -1] = 1
            c2w = np.linalg.inv(w2c)

        elif type == 'elu':
            eye = mat[0, :]
            lookat = mat[1, :]
            up = mat[2, :]
            c2w = elu_to_c2w(eye, lookat, up)

        elif type == 'sph' or type == 'xyz':

            assert (mat.size == 3)
    
            if type == 'sph':
                eye = spherical_to_cartesian((np.deg2rad(mat[0]), np.deg2rad(mat[1]), mat[2]))
            else:
                eye = mat

            lookat = np.zeros(3)
            up = np.array([0, 0, 1])
            c2w = elu_to_c2w(eye, lookat, up)

        # 将pose添加到poses列表中
        poses.append(c2w)
        # 将图像路径添加到legends列表中
        legends.append( os.path.basename(img_path) if img_path else str(fid) )  # ['011.png', '077.png']
        # 将颜色添加到colors列表中
        colors.append('blue')
        # 将图像路径添加到image_paths列表中
        image_paths.append(img_path)  # ['inputs/quick/cam_c2w/images/011.png', 'inputs/quick/cam_c2w/images/077.png']

    # 返回poses、legends、colors、image_paths
    return poses, legends, colors, image_paths


def load_nerf(root_path):

    # 初始化变量
    poses = []
    legends = []
    colors = []
    image_paths = []

    # 获取pose文件路径
    pose_path = os.path.join(root_path, 'transforms.json')
    print(f'Load poses from {pose_path}')

    # 打开pose文件
    with open(pose_path, 'r') as fin:
        jdata = json.load(fin)

    # 遍历pose文件中的每一帧
    for fi, frm in enumerate(jdata['frames']):

        # 获取每一帧的变换矩阵
        c2w = np.array(frm['transform_matrix'])
        poses.append(c2w)
        colors.append('blue')

        # 如果帧中有文件路径，则获取文件名和路径
        if 'file_path' in frm:
            fpath = frm['file_path']
            fname = os.path.basename(fpath)
            
            legends.append(fname)
            image_paths.append(os.path.join(root_path, fpath))
        else:
            # 如果没有文件路径，则使用帧的索引作为文件名
            legends.append(str(fi))
            image_paths.append(None)

    # 返回pose、图例、颜色和图像路径
    return poses, legends, colors, image_paths


def load_colmap(root_path):

    # 初始化变量
    poses = []
    legends = []
    colors = []
    image_paths = []

    # 获取pose文件路径
    pose_path = os.path.join(root_path, 'images.txt')
    print(f'Load poses from {pose_path}')
    
    # 打开pose文件
    fin = open(pose_path, 'r')

    # 初始化up向量
    up = np.zeros(3)

    # 初始化计数器
    i = 0
    # 遍历pose文件中的每一行
    for line in fin:
        line = line.strip()
        # 如果行首为#，则跳过
        if line[0] == "#":
            continue
        i = i + 1
        # 如果计数器为偶数，则跳过
        if  i % 2 == 0:
            continue
        # 将行按空格分割
        elems = line.split(' ')

        # 获取文件名
        fname = '_'.join(elems[9:])
        legends.append(fname)

        # 获取文件路径
        fpath = os.path.join(root_path, 'images', fname)
        image_paths.append(fpath)

        # 将qvec和tvec转换为numpy数组
        qvec = np.array(tuple(map(float, elems[1:5])))
        tvec = np.array(tuple(map(float, elems[5:8])))
        rot = qvec_to_rotmat(-qvec)
        tvec = tvec.reshape(3)

        # 计算w2c矩阵
        w2c = np.eye(4)
        w2c[:3, :3] = rot
        w2c[:3, -1] = tvec
        c2w = np.linalg.inv(w2c)

        c2w[0:3,2] *= -1 # flip the y and z axis
        c2w[0:3,1] *= -1
        c2w = c2w[[1,0,2,3],:]
        c2w[2,:] *= -1 # flip whole world upside down

        up += c2w[0:3,1]

        poses.append(c2w)
        colors.append('blue')

    fin.close()

    up = up / np.linalg.norm(up)
    up_rot = rotmat(up,[0,0,1]) # rotate up vector to [0,0,1]
    up_rot = np.pad(up_rot,[0,1])
    up_rot[-1, -1] = 1

    for i in range(0, len(poses)):
        poses[i] = np.matmul(up_rot, poses[i])

    return poses, legends, colors, image_paths
