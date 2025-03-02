import os

from PIL import Image
import plotly.graph_objects as go
import numpy as np
from src.utils import fetchPly

def calc_cam_cone_pts_3d(c2w, fov_deg, zoom=0.1):

    # 将视场角从度转换为弧度
    fov_rad = np.deg2rad(fov_deg)

    # 获取相机在世界坐标系中的位置，外参矩阵的最后一列前三行即平移向量
    cam_x = c2w[0, -1]
    cam_y = c2w[1, -1]
    cam_z = c2w[2, -1]

    # 计算相机锥体的顶点，光心corn1+4个顶点，相机坐标系下的坐标，tan(fovy/2)=top/|near|,|near|=|-1|=1
    corn1 = [np.tan(fov_rad / 2.0), np.tan(fov_rad / 2.0), -1.0]
    corn2 = [-np.tan(fov_rad / 2.0), np.tan(fov_rad / 2.0), -1.0]
    corn3 = [-np.tan(fov_rad / 2.0), -np.tan(fov_rad / 2.0), -1.0]
    corn4 = [np.tan(fov_rad / 2.0), -np.tan(fov_rad / 2.0), -1.0]
    corn5 = [0, np.tan(fov_rad / 2.0), -1.0]

    # 将相机锥体的顶点从相机坐标系转换到世界坐标系
    corn1 = np.dot(c2w[:3, :3], corn1)  # 点乘外参矩阵的前三行前三列转换到世界坐标系
    corn2 = np.dot(c2w[:3, :3], corn2)
    corn3 = np.dot(c2w[:3, :3], corn3)
    corn4 = np.dot(c2w[:3, :3], corn4)
    corn5 = np.dot(c2w[:3, :3], corn5)

    # Now attach as offset to actual 3D camera position:
    corn1 = np.array(corn1) / np.linalg.norm(corn1, ord=2) * zoom  # 世界坐标系中视锥顶点坐标除以corn1的L2范数（即欧几里得范数），然后乘以缩放系数zoom
    corn_x1 = cam_x + corn1[0]
    corn_y1 = cam_y + corn1[1]
    corn_z1 = cam_z + corn1[2]
    corn2 = np.array(corn2) / np.linalg.norm(corn2, ord=2) * zoom
    corn_x2 = cam_x + corn2[0]
    corn_y2 = cam_y + corn2[1]
    corn_z2 = cam_z + corn2[2]
    corn3 = np.array(corn3) / np.linalg.norm(corn3, ord=2) * zoom
    corn_x3 = cam_x + corn3[0]
    corn_y3 = cam_y + corn3[1]
    corn_z3 = cam_z + corn3[2]
    corn4 = np.array(corn4) / np.linalg.norm(corn4, ord=2) * zoom
    corn_x4 = cam_x + corn4[0]
    corn_y4 = cam_y + corn4[1]
    corn_z4 = cam_z + corn4[2]
    corn5 = np.array(corn5) / np.linalg.norm(corn5, ord=2) * zoom
    corn_x5 = cam_x + corn5[0]
    corn_y5 = cam_y + corn5[1]
    corn_z5 = cam_z + corn5[2]

    xs = [cam_x, corn_x1, corn_x2, corn_x3, corn_x4, corn_x5]  # `xs` 列表存储相机位置和锥体顶点的 x 坐标
    ys = [cam_y, corn_y1, corn_y2, corn_y3, corn_y4, corn_y5]
    zs = [cam_z, corn_z1, corn_z2, corn_z3, corn_z4, corn_z5]

    return np.array([xs, ys, zs]).T


class CameraVisualizer:

    def __init__(self, poses, legends, colors, images=None, mesh_path=None, camera_x=1.0, ply_path=None):
        self._fig = None
        # 将poses赋值给self._poses是为了在类的实例中存储相机姿态数据。这样，类的其他方法可以访问和操作这些数据。self._poses是一个实例变量，它在类的整个生命周期内都可以被访问和修改
        # 设置相机位置
        self._camera_x = camera_x
        # 目的是为了在类的不同方法中使用相机姿态数据，例如在可视化相机位置和方向时
        # 设置相机姿态、标签和颜色
        self._poses = poses
        self._legends = legends  # 图片名称列表
        self._colors = colors

        # 初始化图像和颜色映射
        self._raw_images = None
        self._bit_images = None
        self._image_colorscale = None


        # 如果有图像，进行编码
        if images is not None:
            self._raw_images = images
            self._bit_images = []
            self._image_colorscale = []
            for img in images:
                if img is None:
                    self._bit_images.append(None)
                    self._image_colorscale.append(None)
                    continue
                # 对图像进行编码，返回编码后的位图bit_img和颜色比例colorscale
                bit_img, colorscale = self.encode_image(img)
                self._bit_images.append(bit_img)
                self._image_colorscale.append(colorscale)

        # 初始化网格
        self._mesh = None
        if mesh_path is not None and os.path.exists(mesh_path):
            import trimesh
            self._mesh = trimesh.load(mesh_path, force='mesh')

        # 初始化点云
        if ply_path is not None:
            self._ply = fetchPly(ply_path)


    def encode_image(self, raw_image):
        '''
        :param raw_image (H, W, 3) array of uint8 in [0, 255].
        '''
        # https://stackoverflow.com/questions/60685749/python-plotly-how-to-add-an-image-to-a-3d-scatter-plot

        # 创建一个3x3的白色图像
        dum_img = Image.fromarray(np.ones((3, 3, 3), dtype='uint8')).convert('P', palette='WEB')
        # 获取图像的调色板，并将其转换为numpy数组
        idx_to_color = np.array(dum_img.getpalette()).reshape((-1, 3))

        # 将原始图像转换为调色板图像
        bit_image = Image.fromarray(raw_image).convert('P', palette='WEB', dither=None)
        # bit_image = Image.fromarray(raw_image.clip(0, 254)).convert(
        #     'P', palette='WEB', dither=None)
        # 创建一个颜色映射表
        colorscale = [
            [i / 255.0, 'rgb({}, {}, {})'.format(*rgb)] for i, rgb in enumerate(idx_to_color)]

        return bit_image, colorscale


    def update_figure(
            self, scene_bounds,
            base_radius=0.0, zoom_scale=1.0, fov_deg=50.,
            mesh_z_shift=0.0, mesh_scale=1.0,
            show_background=False, show_grid=False, show_ticklabels=False, y_up=False
        ):

        fig = go.Figure()

        # 如果有网格，添加网格
        if self._mesh is not None:
            fig.add_trace(
                go.Mesh3d(
                    x=self._mesh.vertices[:, 0] * mesh_scale,
                    y=self._mesh.vertices[:, 2] * -mesh_scale,
                    z=(self._mesh.vertices[:, 1] + mesh_z_shift) * mesh_scale,
                    i=self._mesh.faces[:, 0],
                    j=self._mesh.faces[:, 1],
                    k=self._mesh.faces[:, 2],
                    color=None,
                    facecolor=None,
                    opacity=0.8,
                    lighting={'ambient': 1},
                )
            )
        # 添加 PLY 点云数据
        if self._ply is not None:
            points, colors = self._ply  # 获取点云坐标和颜色
            # 如果PLY文件有颜色数据，进行归一化处理
            if colors is not None:
                # 归一化颜色到 [0, 1]
                colors = colors / 255.0

            fig.add_trace(go.Scatter3d(
                x=points[:, 0],
                y=points[:, 2] * -1,
                z=points[:, 1],
                mode='markers',
                marker=dict(
                    size=2,
                    opacity=0.8,
                    color=colors,  # 直接传递归一化的颜色
                    colorscale='Plasma',  # 也可以选择其他调色板
                ),
                name='Point Cloud'
            ))


        # 遍历相机姿态
        for i in range(len(self._poses)):

            pose = self._poses[i]  # 获取第i个相机姿态array
            clr = self._colors[i]  # 获取第i个相机框颜色
            legend = self._legends[i]  # 获取第i个相机图片名称

            # 定义相机边缘 (0, 1) 表示从顶点 0 到顶点 1 的边、(0, 2) 表示从顶点 0 到顶点 2 的边
            edges = [(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (2, 3), (3, 4), (4, 1), (0, 5)]

            # 计算相机锥体顶点
            cone = calc_cam_cone_pts_3d(pose, fov_deg)
            radius = np.linalg.norm(pose[:3, -1])  # 计算相机位置到原点的距离

            # 如果有图像，添加图像 将图像添加到3D图形中
            if self._bit_images and self._bit_images[i]:

                raw_image = self._raw_images[i]
                bit_image = self._bit_images[i]
                colorscale = self._image_colorscale[i]

                (H, W, C) = raw_image.shape

                z = np.zeros((H, W)) + base_radius
                # 使用 np.meshgrid 函数创建了 x 和 y 坐标的网格。np.linspace 用于生成从 -self._camera_x 到 self._camera_x 的 W 个等间距点，以及从 1.0 到 -1.0 的 H 个等间距点，并乘以 H / W 以保持纵横比
                (x, y) = np.meshgrid(np.linspace(-1.0 * self._camera_x, 1.0 * self._camera_x, W), np.linspace(1.0, -1.0, H) * H / W)  # 在三维空间创建画出每个相机的视锥体近裁面

                xyz = np.concatenate([x[..., None], y[..., None], z[..., None]], axis=-1)  # xyz 代表在三维空间中每个相机视锥体近裁面的线上每个点坐标

                rot_xyz = np.matmul(xyz, pose[:3, :3].T) + pose[:3, -1]  # 将 xyz 坐标通过相机姿态矩阵 pose 进行旋转和平移

                x, y, z = rot_xyz[:, :, 0], rot_xyz[:, :, 1], rot_xyz[:, :, 2]

                fig.add_trace(go.Surface(
                    x=x, y=y, z=z,
                    surfacecolor=bit_image,
                    cmin=0,
                    cmax=255,
                    colorscale=colorscale,
                    showscale=False,
                    lighting_diffuse=1.0,
                    lighting_ambient=1.0,
                    lighting_fresnel=1.0,
                    lighting_roughness=1.0,
                    lighting_specular=0.3))

            # 添加相机边缘到3D图形中
            for (i, edge) in enumerate(edges):
                (x1, x2) = (cone[edge[0], 0], cone[edge[1], 0])
                (y1, y2) = (cone[edge[0], 1], cone[edge[1], 1])
                (z1, z2) = (cone[edge[0], 2], cone[edge[1], 2])
                fig.add_trace(go.Scatter3d(
                    x=[x1, x2], y=[y1, y2], z=[z1, z2], mode='lines',
                    line=dict(color=clr, width=3),
                    name=legend, showlegend=(i == 0)))

            # Add label.根据相机锥体顶点的z坐标值来决定标签的位置。如果z坐标小于0，则标签显示在锥体顶点的下方；否则，标签显示在锥体顶点的上方
            if cone[0, 2] < 0:
                fig.add_trace(go.Scatter3d(
                    x=[cone[0, 0]], y=[cone[0, 1]], z=[cone[0, 2] - 0.05], showlegend=False,
                    mode='text', text=legend, textposition='bottom center'))
            else:
                fig.add_trace(go.Scatter3d(
                    x=[cone[0, 0]], y=[cone[0, 1]], z=[cone[0, 2] + 0.05], showlegend=False,
                    mode='text', text=legend, textposition='top center'))

        # look at the center of scene
        fig.update_layout(
            height=1300,
            autosize=True,
            hovermode=False,
            margin=go.layout.Margin(l=0, r=0, b=0, t=0),
            showlegend=True,
            legend=dict(
                yanchor='bottom',
                y=0.01,
                xanchor='right',
                x=0.99,
            ),
            scene=dict(
                aspectmode='manual',
                aspectratio=dict(x=1, y=1, z=1),
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.0),
                    center=dict(x=0.0, y=0.0, z=0.0),
                    up=dict(x=0.0, y=0.0, z=1.0)),
                xaxis_title='X',
                yaxis_title='Z' if y_up else 'Y',
                zaxis_title='Y' if y_up else 'Z',
                xaxis=dict(
                    range=[-scene_bounds, scene_bounds],
                    showticklabels=show_ticklabels,
                    showgrid=show_grid,
                    zeroline=False,
                    showbackground=show_background,
                    showspikes=False,
                    showline=False,
                    ticks=''),
                yaxis=dict(
                    range=[-scene_bounds, scene_bounds],
                    showticklabels=show_ticklabels,
                    showgrid=show_grid,
                    zeroline=False,
                    showbackground=show_background,
                    showspikes=False,
                    showline=False,
                    ticks=''),
                zaxis=dict(
                    range=[-scene_bounds, scene_bounds],
                    showticklabels=show_ticklabels,
                    showgrid=show_grid,
                    zeroline=False,
                    showbackground=show_background,
                    showspikes=False,
                    showline=False,
                    ticks='')
            )
        )

        self._fig = fig
        return fig