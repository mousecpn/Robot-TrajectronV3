import os
os.environ["PYOPENGL_PLATFORM"] = "egl"
import subprocess
import trimesh
# import pyrender
import numpy as np
from PIL import Image
import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R

from utils_exp.grasp import Grasp
from utils_exp.transform import Transform, Rotation
from utils_exp.implicit import as_mesh
try:
    import open3d as o3d
except:
    pass
#########
# visualize affordance and graso
#########
cmap = plt.get_cmap('Reds')


def visualize_point_cloud_with_normals(points, normals=None, grasps=None):
    """
    Visualizes a point cloud with normals using Open3D.

    Args:
        points (torch.Tensor): Tensor of shape (n, 3) representing point cloud.
        normals (torch.Tensor, optional): Tensor of shape (n, 3) representing normals. Defaults to None.
    """
    # Convert points to Open3D format
    if not isinstance(points, np.ndarray):
        pcl = points
    else:
        pcl = o3d.geometry.PointCloud()
        pcl.points = o3d.utility.Vector3dVector(points)

    if normals is not None:
        pcl.normals = o3d.utility.Vector3dVector(normals)

    # Visualize the point cloud
    if grasps is None:
        o3d.visualization.draw_geometries([pcl],
                                        window_name="Point Cloud with Normals",
                                        point_show_normal=True)
    else:
        o3d.visualization.draw_geometries([pcl, ]+ grasps,
                                        window_name="Point Cloud with Normals",
                                        point_show_normal=True)


def color_print(text, fg="default", bg="default", style="normal"):
    """
    Print `text` in a specified foreground color, background color, and style
    using ANSI escape sequences.
    
    :param text:  The text to be printed
    :param fg:    The foreground color (string)
    :param bg:    The background color (string)
    :param style: The text style (string)
    """
    # ANSI styles
    styles = {
        "normal":      "0",
        "bold":        "1",
        "faint":       "2",
        "italic":      "3",
        "underline":   "4",
        "blink":       "5",
        "negative":    "7",
        "concealed":   "8",
        "strikethrough": "9"
    }

    # Foreground (text) colors
    fg_colors = {
        "default":     "39",
        "black":       "30",
        "red":         "31",
        "green":       "32",
        "yellow":      "33",
        "blue":        "34",
        "magenta":     "35",
        "cyan":        "36",
        "light_gray":  "37",
        "dark_gray":   "90",
        "light_red":   "91",
        "light_green": "92",
        "light_yellow":"93",
        "light_blue":  "94",
        "light_magenta":"95",
        "light_cyan":  "96",
        "white":       "97"
    }

    # Background colors
    bg_colors = {
        "default":     "49",
        "black":       "40",
        "red":         "41",
        "green":       "42",
        "yellow":      "43",
        "blue":        "44",
        "magenta":     "45",
        "cyan":        "46",
        "light_gray":  "47",
        "dark_gray":   "100",
        "light_red":   "101",
        "light_green": "102",
        "light_yellow":"103",
        "light_blue":  "104",
        "light_magenta":"105",
        "light_cyan":  "106",
        "white":       "107"
    }

    # Get the correct codes from the dictionaries (fallback to "normal"/"default" if not found)
    style_code = styles.get(style, styles["normal"])
    fg_code    = fg_colors.get(fg, fg_colors["default"])
    bg_code    = bg_colors.get(bg, bg_colors["default"])

    # Construct and print the ANSI escaped string
    print(f"\033[{style_code};{fg_code};{bg_code}m{text}\033[0m")

def draw_trajectory_in_pybullet(p, points, color=[1, 1, 0], line_width=2.0, life_time=0):
    """
    在 PyBullet 中可视化一个三维轨迹。

    参数：
        points : list[np.ndarray] 或 np.ndarray
            一系列 3D 点 (x, y, z)，可以是 shape=(N,3) 的数组或长度为 N 的向量列表。
        color : list[float]
            轨迹颜色，默认黄色 [1, 1, 0]
        line_width : float
            线段宽度
        life_time : float
            可视化持续时间，0 表示永久
    """
    points = np.array(points)
    assert points.ndim == 2 and points.shape[1] == 3, "points 应该是 (N,3) 的数组"

    for i in range(len(points) - 1):
        p.addUserDebugLine(
            points[i],
            points[i + 1],
            lineColorRGB=color,
            lineWidth=line_width,
            lifeTime=life_time
        )

def draw_frame_in_pybullet(p, T: np.ndarray, axis_len: float = 0.2, line_width: float = 2.0, life_time: float = 0):
    """
    在 PyBullet 中绘制一个坐标系（frame），输入为 4x4 齐次变换矩阵。

    参数：
        T : np.ndarray
            4x4 齐次变换矩阵（旋转 + 平移）
        axis_len : float
            每个坐标轴的长度
        line_width : float
            坐标轴线条粗细
        life_time : float
            可视化持续时间，0 表示永久
    """
    assert T.shape == (4, 4), "输入必须是 4x4 的齐次变换矩阵"

    R = T[:3, :3]  # 旋转矩阵
    t = T[:3, 3]   # 平移向量

    # 坐标轴颜色：X 红，Y 绿，Z 蓝
    colors = {
        'x': [1, 0, 0],
        'y': [0, 1, 0],
        'z': [0, 0, 1],
    }

    # 绘制三条轴线
    p.addUserDebugLine(t, t + R[:, 0]*axis_len, colors['x'], lineWidth=line_width, lifeTime=life_time)
    p.addUserDebugLine(t, t + R[:, 1]*axis_len, colors['y'], lineWidth=line_width, lifeTime=life_time)
    p.addUserDebugLine(t, t + R[:, 2]*axis_len, colors['z'], lineWidth=line_width, lifeTime=life_time)

def affordance_visual(qual_vol,
                      rot_vol,
                      scene_mesh,
                      size=0.3,
                      resolution=40,
                      th=0.5,
                      temp=150,
                      rad=0.02,
                      finger_depth=0.05,
                      finger_offset=0.5,
                      move_center=True,
                      aggregation='max'):
    # Transform voxel grid into point cloud
    x = np.linspace(0, size, num=resolution)
    y = np.linspace(0, size, num=resolution)
    z = np.linspace(0, size, num=resolution)
    X, Y, Z = np.meshgrid(x, y, z)
    grid = np.stack((Y, X, Z), axis=-1)
    # move center_vol to grasp center
    if move_center:
        z_axis = np.stack([
            2 * rot_vol[:, :, :, 0] * rot_vol[:, :, :, 2] +
            2 * rot_vol[:, :, :, 1] * rot_vol[:, :, :, 3],
            2 * rot_vol[:, :, :, 1] * rot_vol[:, :, :, 2] -
            2 * rot_vol[:, :, :, 0] * rot_vol[:, :, :, 3],
            1 - 2 * rot_vol[:, :, :, 0] * rot_vol[:, :, :, 0] -
            2 * rot_vol[:, :, :, 1] * rot_vol[:, :, :, 1]
        ],
                          axis=-1)
        grid += z_axis * finger_depth * finger_offset

    grid = grid[qual_vol > th]
    if grid.shape[0] <= 0:
        return scene_mesh
    qual_vol = qual_vol[qual_vol > th]
    pc_coordinate = np.reshape(grid, (-1, 3))
    pc_vector = np.expand_dims(np.reshape(qual_vol, (-1, )), axis=1)
    qual_pc = np.concatenate((pc_coordinate, pc_vector), axis=1)

    # Calculate the affordance value for each trimesh face
    # sum(exp(-dist_i * 150) * aff_i) / sum(exp(-dist_i)) (using 150 as the temperature term for exp)
    mesh = scene_mesh.copy()
    triangles_center = mesh.triangles_center
    centers = np.reshape(triangles_center, (triangles_center.shape[0], 1, 3))
    qual_pc_coords = np.reshape(qual_pc[:, 0:3], (1, -1, 3))
    diff = centers - qual_pc_coords
    dist = np.sqrt((diff**2).sum(axis=-1))

    if aggregation == 'mean':
        weight = np.exp(-dist * temp)
        affordance = weight.dot(qual_pc[:, 3]) / weight.sum(axis=-1)
    elif aggregation == 'max':
        # num_face, num_points
        mask = dist <= rad
        affordance = mask * qual_pc[:, 3][np.newaxis]
        # num_face
        affordance = affordance.max(axis=1)
    elif aggregation == 'softmax':
        # num_face, num_points
        mask = dist <= rad
        affordance_mask = mask * qual_pc[:, 3][np.newaxis]
        # mask out points outside radiance
        affordance_mask[np.logical_not(mask)] = -1e10
        # softmax
        weight = np.exp(affordance_mask * temp)
        affordance = weight.dot(qual_pc[:, 3]) / (weight.sum(axis=-1) + 1e-5)

    affordance = np.clip(affordance, a_min=th, a_max=1)
    affordance = (affordance - th) / (1 - th)
    # affordance = (affordance - affordance.min()) / (affordance.max() -
    #                                                 affordance.min())
    # different colormaps if need to change
    # cmap = plt.get_cmap('rainbow')
    # cmap = plt.get_cmap('nipy_spectral')
    colors = cmap(affordance ** 4)
    mesh.visual.face_colors = colors
    return mesh


def grasp2mesh(grasp, score, finger_depth=0.05, color = np.array([0, 250, 0, 180]).astype(np.uint8)):
    # color = cmap(float(score))
    # color = (np.array(color) * 255).astype(np.uint8)
    radius = 0.1 * finger_depth
    w, d = grasp.width, finger_depth
    scene = trimesh.Scene()
    # left finger
    pose = grasp.pose * Transform(Rotation.identity(), [0.0, -w / 2, d / 2])
    scale = [radius, radius, d]
    left_finger = trimesh.creation.cylinder(radius,
                                            d,
                                            transform=pose.as_matrix())
    scene.add_geometry(left_finger, 'left_finger')

    # right finger
    pose = grasp.pose * Transform(Rotation.identity(), [0.0, w / 2, d / 2])
    scale = [radius, radius, d]
    right_finger = trimesh.creation.cylinder(radius,
                                             d,
                                             transform=pose.as_matrix())
    scene.add_geometry(right_finger, 'right_finger')

    # wrist
    pose = grasp.pose * Transform(Rotation.identity(), [0.0, 0.0, -d / 4])
    scale = [radius, radius, d / 2]
    wrist = trimesh.creation.cylinder(radius,
                                      d / 2,
                                      transform=pose.as_matrix())
    scene.add_geometry(wrist, 'wrist')

    # palm
    pose = grasp.pose * Transform(
        Rotation.from_rotvec(np.pi / 2 * np.r_[1.0, 0.0, 0.0]),
        [0.0, 0.0, 0.0])
    scale = [radius, radius, w]
    palm = trimesh.creation.cylinder(radius, w, transform=pose.as_matrix())
    scene.add_geometry(palm, 'palm')
    scene = as_mesh(scene)
    colors = np.repeat(color[np.newaxis, :], len(scene.faces), axis=0)
    scene.visual.face_colors = colors
    return scene

#########
# Render
#########
def get_camera_pose(radius, center=np.zeros(3), ax=0, ay=0, az=0):
    rotation = R.from_euler('xyz', (ax, ay, az)).as_matrix()
    vec = np.array([0, 0, radius])
    translation = rotation.dot(vec) + center
    camera_pose = np.zeros((4, 4))
    camera_pose[3, 3] = 1
    camera_pose[:3, :3] = rotation
    camera_pose[:3, 3] = translation
    return camera_pose

def render_mesh(mesh, camera, light, camera_pose, light_pose, renderer):
    r_scene = pyrender.Scene()
    o_mesh = pyrender.Mesh.from_trimesh(mesh, smooth=False)
    r_scene.add(o_mesh)
    r_scene.add(camera, name='camera', pose=camera_pose)
    r_scene.add(light, name='light', pose=light_pose)
    color_img, _ = renderer.render(r_scene)
    return Image.fromarray(color_img)

#########
# Plot
#########


def plot_3d_point_cloud(x,
                        y,
                        z,
                        show=True,
                        show_axis=True,
                        in_u_sphere=False,
                        marker='.',
                        s=8,
                        alpha=.8,
                        figsize=(5, 5),
                        elev=10,
                        azim=240,
                        axis=None,
                        title=None,
                        lim=None,
                        *args,
                        **kwargs):

    if axis is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
    else:
        ax = axis
        fig = axis

    if title is not None:
        plt.title(title)

    sc = ax.scatter(x, y, z, marker=marker, s=s, alpha=alpha, *args, **kwargs)
    ax.view_init(elev=elev, azim=azim)

    if lim:
        ax.set_xlim3d(*lim[0])
        ax.set_ylim3d(*lim[1])
        ax.set_zlim3d(*lim[2])
    elif in_u_sphere:
        ax.set_xlim3d(-0.5, 0.5)
        ax.set_ylim3d(-0.5, 0.5)
        ax.set_zlim3d(-0.5, 0.5)
    else:
        lim = (min(np.min(x), np.min(y),
                   np.min(z)), max(np.max(x), np.max(y), np.max(z)))
        ax.set_xlim(1.3 * lim[0], 1.3 * lim[1])
        ax.set_ylim(1.3 * lim[0], 1.3 * lim[1])
        ax.set_zlim(1.3 * lim[0], 1.3 * lim[1])
        plt.tight_layout()

    if not show_axis:
        plt.axis('off')

    if show:
        plt.show()

    return fig


def plot_3d_point_cloud_dict(name_dict, lim, size=2):
    num_plots = len(name_dict)
    fig = plt.figure(figsize=(size * num_plots, size))
    ax = {}
    for i, (k, v) in enumerate(name_dict.items()):
        ax[k] = fig.add_subplot(1, num_plots, i + 1, projection='3d')
        plot_3d_point_cloud(v[2], -v[0], v[1], axis=ax[k], show=False, lim=lim)
        ax[k].set_title(k)
    plt.tight_layout()
    return fig


def plot_3d_voxel_cloud_dict(name_dict, size=5, *args, **kwargs):
    num_plots = len(name_dict)
    fig = plt.figure(figsize=(size * num_plots, size))
    ax = {}
    for i, (k, v) in enumerate(name_dict.items()):
        ax[k] = fig.add_subplot(1, num_plots, i + 1, projection='3d')
        plot_voxel_as_cloud(v, axis=ax[k], fig=fig, *args, **kwargs)
        ax[k].set_title(k)
    plt.tight_layout()
    return fig


def plot_voxel_as_cloud(voxel,
                        axis=None,
                        figsize=(5, 5),
                        marker='s',
                        s=8,
                        alpha=.8,
                        lim=[0.3, 0.3, 0.3],
                        elev=10,
                        azim=240,
                        fig=None,
                        *args,
                        **kwargs):
    cloud = convert_voxel_to_cloud(voxel, lim)
    points = cloud[:, :3]
    val = cloud[:, 3]
    if axis is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
    else:
        ax = axis

    # color = cm.coolwarm(val)
    sc = ax.scatter(*points.T,
                    marker=marker,
                    s=s,
                    alpha=alpha,
                    c=val,
                    cmap=plt.cm.get_cmap('RdYlBu_r'),
                    *args,
                    **kwargs)
    plt.colorbar(sc, ax=ax)
    ax.set_xlim3d(0, lim[0])
    ax.set_ylim3d(0, lim[1])
    ax.set_zlim3d(0, lim[2])

    return fig

def plot_tsdf_with_grasps(tsdf,
                          grasps,
                          axis=None,
                          figsize=(5, 5),
                          marker='s',
                          s=8,
                          alpha=.8,
                          lim=[0.3, 0.3, 0.3],
                          elev=10,
                          azim=240,
                          fig=None,
                          *args,
                          **kwargs):
    cloud = convert_voxel_to_cloud(tsdf, lim)
    points = cloud[:, :3]
    val = cloud[:, 3]
    grasp_meshes = [
        grasp2mesh(grasps[idx], 1) for idx in range(len(grasps))
    ]
    gripper_points = [grasp_mesh.sample(512) for grasp_mesh in grasp_meshes]
    gripper_points = np.concatenate(gripper_points, axis=0)
    cmap = plt.cm.get_cmap('RdYlBu_r')
    color_tsdf = cmap(val)
    color_gripper_points = np.tile(np.array([[0, 1, 0, 0.3]]),
                                   (gripper_points.shape[0], 1))

    points = np.concatenate((points, gripper_points), axis=0)
    color = np.concatenate((color_tsdf, color_gripper_points), axis=0)

    if axis is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
    else:
        ax = axis

    # color = cm.coolwarm(val)
    sc = ax.scatter(*points.T,
                    marker=marker,
                    s=s,
                    alpha=alpha,
                    c=color,
                    *args,
                    **kwargs)
    # plt.colorbar(sc, ax=ax)
    ax.set_xlim3d(0, lim[0])
    ax.set_ylim3d(0, lim[1])
    ax.set_zlim3d(0, lim[2])

    return fig

def convert_voxel_to_cloud(voxel, size):

    assert len(voxel.shape) == 3
    lx, ly, lz = voxel.shape
    lx = lx / voxel.shape[0] * size[0]
    ly = ly / voxel.shape[1] * size[1]
    lz = lz / voxel.shape[2] * size[2]
    points = []
    for x in range(voxel.shape[0]):
        for y in range(voxel.shape[1]):
            for z in range(voxel.shape[2]):
                if voxel[x, y, z] > 0:
                    points.append([
                        x / voxel.shape[0] * size[0],
                        y / voxel.shape[1] * size[1],
                        z / voxel.shape[2] * size[2], voxel[x, y, z]
                    ])

    return np.array(points)

def trimesh_to_open3d(trimesh_mesh):
    """
    Converts a Trimesh object to an Open3D object.

    Args:
        trimesh_mesh (trimesh.Trimesh): A Trimesh object.

    Returns:
        open3d.geometry.TriangleMesh: The corresponding Open3D mesh.
    """
    # Extract vertices and faces from Trimesh
    vertices = trimesh_mesh.vertices
    faces = trimesh_mesh.faces

    # Create Open3D TriangleMesh
    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(vertices)
    o3d_mesh.triangles = o3d.utility.Vector3iVector(faces)

    # Optionally compute normals
    o3d_mesh.compute_vertex_normals()

    return o3d_mesh 


# def pcl2mesh(pc):
#     pcd = o3d.geometry.PointCloud()
#     pcd.points = o3d.utility.Vector3dVector(pc)

#     # 2. 估计法线（表面重建需要）
#     pcd.estimate_normals()

#     # 3. Alpha Shape 重建
#     alpha = 0.05   # 半径参数，越小越贴合点云，越大越平滑
#     mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)
#     return mesh


def pointcloud_to_meshes(points: np.ndarray,
                         eps: float = 0.01,
                         min_points: int = 20,
                         method: str = "convex",
                         alpha: float = 0.05,
                         save_dir: str = None) -> list:
    """
    将 numpy 点云分割成多个物体，并计算每个物体的外包络 mesh

    Args:
        points (np.ndarray): 输入点云，形状 (N,3)
        eps (float): DBSCAN 半径阈值（点与点的最大聚类距离）
        min_points (int): DBSCAN 最小点数
        method (str): "convex"（凸包） 或 "alpha"（Alpha Shape）
        alpha (float): Alpha Shape 参数，越小越贴合点云

    Returns:
        meshes (List[o3d.geometry.TriangleMesh]): 每个物体的外包络网格
    """
    # 转换为 Open3D 点云
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # 聚类
    labels = np.array(
        pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=False)
    )
    n_clusters = labels.max() + 1

    meshes = []
    mesh_filenames = []
    for i in range(n_clusters):
        cluster = pcd.select_by_index(np.where(labels == i)[0])
        if len(cluster.points) == 0:
            continue

        if method == "convex":
            mesh, _ = cluster.compute_convex_hull()
            mesh.compute_vertex_normals()
            meshes.append(mesh)

        elif method == "alpha":
            mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
                cluster, alpha=alpha
            )
            mesh.compute_vertex_normals()
            meshes.append(mesh)
        else:
            raise ValueError("method 必须是 'convex' 或 'alpha'")
    
        # 保存 STL
        if save_dir is not None:
            filename = os.path.join(save_dir, f"object_{i}.stl")
            o3d.io.write_triangle_mesh(filename, mesh)
            mesh_filenames.append(filename)
    if save_dir is not None:
        return mesh, mesh_filenames

    return meshes