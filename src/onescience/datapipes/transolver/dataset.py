import torch
import vtk
import os
import itertools
import random
import numpy as np
from torch_geometric import nn as nng
from sklearn.neighbors import NearestNeighbors
from torch_geometric.data import Data, Dataset
from torch_geometric.utils import k_hop_subgraph, subgraph
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk
import pyvista as pv
import os.path as osp
from tqdm import tqdm
from onescience.utils.transolver.reorganize import reorganize

vtk.vtkRenderWindow.SetGlobalWarningDisplay(0)  # 关闭 VTK 警告
os.environ["DISPLAY"] = ":0"  # 欺骗 VTK 认为存在显示设备（无需实际 GUI）
os.environ["VTK_DISABLE_X_DISPLAY"] = "1"  # 彻底禁用 X Server
os.environ["MESA_NO_DEBUG"] = "1"
os.environ["LIBGL_DEBUG"] = "quiet"

import matplotlib

matplotlib.use("Agg")  # 纯CPU后端
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib import cm, colors


def _polydata_to_tris(poly_data):
    """将任意多边形面转为三角形，并返回 (points[N,3], triangles[M,3])"""
    tri_filter = vtk.vtkTriangleFilter()
    tri_filter.SetInputData(poly_data)
    tri_filter.Update()
    pd = tri_filter.GetOutput()

    pts = vtk_to_numpy(pd.GetPoints().GetData())  # (N, 3)
    polys = vtk_to_numpy(pd.GetPolys().GetData()).reshape(-1, 4)[
        :, 1:4
    ]  # (M, 3) 三角面索引
    return pts, polys, pd


def load_unstructured_grid_data(file_name):  # 加载VTK非结构化网格文件
    reader = vtk.vtkUnstructuredGridReader()
    reader.SetFileName(file_name)
    reader.Update()
    output = reader.GetOutput()
    return output


def unstructured_grid_data_to_poly_data(
    unstructured_grid_data,
):  # 将非结构化网格转为表面多边形数据
    filter = vtk.vtkDataSetSurfaceFilter()
    filter.SetInputData(unstructured_grid_data)
    filter.Update()
    poly_data = filter.GetOutput()
    return poly_data, filter


def get_speed_from_poly_data(poly_data):  # 更改为从poly_data获取速度
    # 获取速度分量
    velo = vtk_to_numpy(poly_data.GetPointData().GetVectors())
    if velo.size == 0:
        raise ValueError("速度向量数据不存在于poly_data的PointData中")
    # 计算速度模长
    speed = np.linalg.norm(velo, axis=1)
    return speed


def get_sdf(target, boundary):  # 计算符号距离函数(SDF)
    nbrs = NearestNeighbors(n_neighbors=1).fit(boundary)
    dists, indices = nbrs.kneighbors(target)
    neis = np.array([boundary[i[0]] for i in indices])
    dirs = (target - neis) / (dists + 1e-8)
    return dists.reshape(-1), dirs


def get_normal(unstructured_grid_data):  # 计算网格点法线
    poly_data, surface_filter = unstructured_grid_data_to_poly_data(
        unstructured_grid_data
    )
    # visualize_poly_data(poly_data, surface_filter)
    # poly_data.GetPointData().SetScalars(None)
    normal_filter = vtk.vtkPolyDataNormals()
    normal_filter.SetInputData(poly_data)
    normal_filter.SetAutoOrientNormals(1)
    normal_filter.SetConsistency(1)
    # normal_filter.SetSplitting(0)
    normal_filter.SetComputeCellNormals(1)
    normal_filter.SetComputePointNormals(0)
    normal_filter.Update()
    """
    normal_filter.SetComputeCellNormals(0)
    normal_filter.SetComputePointNormals(1)
    normal_filter.Update()
    #visualize_poly_data(poly_data, surface_filter, normal_filter)
    poly_data.GetPointData().SetNormals(normal_filter.GetOutput().GetPointData().GetNormals())
    p2c = vtk.vtkPointDataToCellData()
    p2c.ProcessAllArraysOn()
    p2c.SetInputData(poly_data)
    p2c.Update()
    unstructured_grid_data.GetCellData().SetNormals(p2c.GetOutput().GetCellData().GetNormals())
    #visualize_poly_data(poly_data, surface_filter, p2c)
    """

    unstructured_grid_data.GetCellData().SetNormals(
        normal_filter.GetOutput().GetCellData().GetNormals()
    )
    c2p = vtk.vtkCellDataToPointData()
    # c2p.ProcessAllArraysOn()
    c2p.SetInputData(unstructured_grid_data)
    c2p.Update()
    unstructured_grid_data = c2p.GetOutput()
    # return unstructured_grid_data
    normal = vtk_to_numpy(c2p.GetOutput().GetPointData().GetNormals()).astype(np.double)
    # print(np.max(np.max(np.abs(normal), axis=1)), np.min(np.max(np.abs(normal), axis=1)))
    normal /= np.max(np.abs(normal), axis=1, keepdims=True) + 1e-8
    normal /= np.linalg.norm(normal, axis=1, keepdims=True) + 1e-8
    if np.isnan(normal).sum() > 0:
        print(np.isnan(normal).sum())
        print("recalculate")
        return get_normal(unstructured_grid_data)  # re-calculate
    # print(normal)
    return normal


def visualize_poly_data(
    poly_data, surface_filter, scalar_data=None, colorbar_title="Data", save_path=None
):
    # 创建渲染器和窗口（不显示）
    renderer_window = vtk.vtkRenderWindow()
    renderer_window.SetOffScreenRendering(1)
    renderer_window.SetSize(1200, 1000)

    # 创建主网格的渲染器和执行器
    mapper = vtk.vtkDataSetMapper()
    mapper.SetInputData(poly_data)

    # 创建自定义颜色查找表（关键修改部分）
    lut = vtk.vtkLookupTable()
    lut.SetHueRange(0.667, 0.0)  # 从蓝色（0.667）到红色（0.0）
    lut.SetAlphaRange(1.0, 1.0)  # 不透明度固定
    lut.SetValueRange(1.0, 1.0)  # 保持颜色亮度
    lut.Build()

    if scalar_data is not None:
        scalar_array = vtk.vtkDoubleArray()
        scalar_array.SetName("Speed")
        scalar_array.SetNumberOfComponents(1)
        scalar_array.SetNumberOfTuples(len(scalar_data))

        for i in range(len(scalar_data)):
            scalar_array.SetTuple1(i, scalar_data[i])

        poly_data.GetPointData().AddArray(scalar_array)
        poly_data.GetPointData().SetActiveScalars("Speed")  # 显式激活标量数组

        mapper.SetLookupTable(lut)  # 设置颜色映射表
        mapper.SetScalarModeToUsePointData()
        mapper.SetScalarRange(
            np.min(scalar_data), np.max(scalar_data)
        )  # 使用传入数据的范围
    else:
        # 如果没有传入 scalar_data，尝试从 poly_data 中提取标量值
        if poly_data.GetPointData().GetScalars() is not None:
            mapper.SetLookupTable(lut)  # 设置颜色映射表
            mapper.SetScalarModeToUsePointData()
            scalar_range = (
                poly_data.GetPointData().GetScalars().GetRange()
            )  # 从数据中获取标量范围
            mapper.SetScalarRange(scalar_range)  # 设置标量范围为数据范围
        else:
            # 如果没有可用的标量，则设置为默认值
            mapper.SetScalarRange(0.0, 1.0)

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetOpacity(0.5)

    renderer = vtk.vtkRenderer()
    renderer.AddActor(actor)
    renderer.SetBackground(1, 1, 1)

    # 创建颜色条
    scalar_bar = vtk.vtkScalarBarActor()
    scalar_bar.SetLookupTable(lut)  # 设置关联的颜色表
    scalar_bar.SetTitle(colorbar_title)  # 使用传入的颜色条标题
    scalar_bar.SetNumberOfLabels(4)  # 标签数量
    scalar_bar.GetPositionCoordinate().SetCoordinateSystemToNormalizedDisplay()  # 设置为标准化显示坐标
    scalar_bar.SetPosition(0.95, 0.1)  # 调整颜色条位置
    scalar_bar.SetPosition2(0.05, 0.8)  # 宽度和高度
    scalar_bar.SetOrientationToVertical()  # 设置颜色条为竖直方向
    renderer.AddActor(scalar_bar)  # 将颜色条添加到渲染器

    # 获取相机对象
    camera = renderer.GetActiveCamera()
    renderer_window.SetAlphaBitPlanes(1)  # 透明背景设置
    bounds = poly_data.GetBounds()
    center = [
        (bounds[0] + bounds[1]) / 2,
        (bounds[2] + bounds[3]) / 2,
        (bounds[4] + bounds[5]) / 2,
    ]

    camera.SetPosition(center[0] + 5, center[1] + 2, center[2] - 10)  # 设置相机位置
    camera.SetFocalPoint(center)  # 设置焦点
    camera.SetViewUp(0, 1, 0)  # Z轴向上

    # 将渲染器添加到窗口并渲染
    renderer_window.AddRenderer(renderer)
    renderer_window.Render()
    # 保存图片到本地
    if save_path:
        window_to_image = vtk.vtkWindowToImageFilter()
        window_to_image.SetInput(renderer_window)
        window_to_image.SetScale(1)
        window_to_image.SetInputBufferTypeToRGB()
        window_to_image.ReadFrontBufferOff()  # 避免读取前端缓冲区
        window_to_image.Update()

        writer = vtk.vtkPNGWriter()
        writer.SetFileName(save_path)
        writer.SetInputConnection(window_to_image.GetOutputPort())
        writer.Write()
        # print(f"[可视化结果已保存] {save_path}")


def visualize_poly_data_cpu(
    poly_data, scalar_data=None, colorbar_title="Data", save_path=None
):
    pts, tris, pd_tri = _polydata_to_tris(poly_data)

    if scalar_data is None and pd_tri.GetPointData().GetScalars() is not None:
        scalar_data = vtk_to_numpy(pd_tri.GetPointData().GetScalars())

    fig = plt.figure(figsize=(10, 8), dpi=120)
    ax = fig.add_subplot(111, projection="3d")

    pts = pts[:, [0, 2, 1]]
    faces = pts[tris]
    coll = Poly3DCollection(faces, linewidths=0.05, edgecolors="none")
    if scalar_data is not None:
        scalar_data = np.asarray(scalar_data).reshape(-1)
        face_vals = scalar_data[tris].mean(axis=1)
        norm = colors.Normalize(
            vmin=float(np.nanmin(face_vals)), vmax=float(np.nanmax(face_vals))
        )
        cmap = cm.get_cmap("jet")
        coll.set_facecolor(cmap(norm(face_vals)))
        mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
        mappable.set_array(face_vals)
        cbar = fig.colorbar(mappable, ax=ax, shrink=0.7, pad=0.02)
        cbar.set_label(colorbar_title)
    else:
        coll.set_facecolor((0.7, 0.7, 0.8, 1.0))
    ax.add_collection3d(coll)

    # ====== 关键：按 VTK 相机 position=center+(5,2,-10) 固定视角 ======
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]

    # 以几何中心为对焦点（等价于 VTK 的 center）
    cx, cy, cz = (
        (x.max() + x.min()) * 0.5,
        (y.max() + y.min()) * 0.5,
        (z.max() + z.min()) * 0.5,
    )

    # 对称设置三轴范围，避免物体偏到一侧；略放大一点边界
    rx, ry, rz = (
        (x.max() - x.min()) * 0.5,
        (y.max() - y.min()) * 0.5,
        (z.max() - z.min()) * 0.5,
    )
    r = 1.15 * max(rx, ry, rz)  # 1.15 相当于一点点“缩小”视图，防止裁切
    ax.set_xlim(cx - r, cx + r)
    ax.set_ylim(cy - r, cy + r)
    ax.set_zlim(cz - r, cz + r)
    ax.set_box_aspect((1, 1, 1))

    azim, elev, roll = 210, 20, 0.0  # 单位：度

    ax.view_init(elev=elev, azim=azim, roll=roll)  # 只调用一次
    # 可选：如果想更像你原来 VTK 的半透明，可以开下 alpha
    coll.set_alpha(0.5)

    ax.set_axis_off()
    ax.set_title(colorbar_title)

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def get_cardatalist(
    root, samples, norm=False, coef_norm=None, savedir=None, preprocessed=False
):
    dataset = []
    mean_in, mean_out = 0, 0
    std_in, std_out = 0, 0
    for k, s in enumerate(samples):
        if preprocessed and savedir is not None:
            save_path = os.path.join(savedir, s)
            if not os.path.exists(save_path):
                continue
            init = np.load(os.path.join(save_path, "x.npy"))
            target = np.load(os.path.join(save_path, "y.npy"))
            pos = np.load(os.path.join(save_path, "pos.npy"))
            surf = np.load(os.path.join(save_path, "surf.npy"))
            edge_index = np.load(os.path.join(save_path, "edge_index.npy"))
        else:
            file_name_press = os.path.join(root, os.path.join(s, "quadpress_smpl.vtk"))
            file_name_velo = os.path.join(root, os.path.join(s, "hexvelo_smpl.vtk"))

            if not os.path.exists(file_name_press) or not os.path.exists(
                file_name_velo
            ):
                continue

            unstructured_grid_data_press = load_unstructured_grid_data(file_name_press)
            unstructured_grid_data_velo = load_unstructured_grid_data(file_name_velo)

            velo = vtk_to_numpy(unstructured_grid_data_velo.GetPointData().GetVectors())
            press = vtk_to_numpy(
                unstructured_grid_data_press.GetPointData().GetScalars()
            )
            points_velo = vtk_to_numpy(
                unstructured_grid_data_velo.GetPoints().GetData()
            )
            points_press = vtk_to_numpy(
                unstructured_grid_data_press.GetPoints().GetData()
            )
            # 计算边索引
            edges_press = get_edges(
                unstructured_grid_data_press, points_press, cell_size=4
            )
            edges_velo = get_edges(
                unstructured_grid_data_velo, points_velo, cell_size=8
            )
            # 生成SDF和法向量
            sdf_velo, normal_velo = get_sdf(points_velo, points_press)
            sdf_press = np.zeros(points_press.shape[0])
            normal_press = get_normal(unstructured_grid_data_press)

            surface = {tuple(p) for p in points_press}
            exterior_indices = [
                i for i, p in enumerate(points_velo) if tuple(p) not in surface
            ]
            velo_dict = {tuple(p): velo[i] for i, p in enumerate(points_velo)}

            pos_ext = points_velo[exterior_indices]
            pos_surf = points_press
            sdf_ext = sdf_velo[exterior_indices]
            sdf_surf = sdf_press
            normal_ext = normal_velo[exterior_indices]
            normal_surf = normal_press
            velo_ext = velo[exterior_indices]
            velo_surf = np.array(
                [
                    velo_dict[tuple(p)] if tuple(p) in velo_dict else np.zeros(3)
                    for p in pos_surf
                ]
            )
            press_ext = np.zeros([len(exterior_indices), 1])
            press_surf = press

            init_ext = np.c_[pos_ext, sdf_ext, normal_ext]
            init_surf = np.c_[pos_surf, sdf_surf, normal_surf]
            target_ext = np.c_[velo_ext, press_ext]
            target_surf = np.c_[velo_surf, press_surf]

            surf = np.concatenate([np.zeros(len(pos_ext)), np.ones(len(pos_surf))])
            pos = np.concatenate([pos_ext, pos_surf])
            init = np.concatenate([init_ext, init_surf])
            target = np.concatenate([target_ext, target_surf])
            edge_index = get_edge_index(pos, edges_press, edges_velo)

            if savedir is not None:
                save_path = os.path.join(savedir, s)
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                np.save(os.path.join(save_path, "x.npy"), init)
                np.save(os.path.join(save_path, "y.npy"), target)
                np.save(os.path.join(save_path, "pos.npy"), pos)
                np.save(os.path.join(save_path, "surf.npy"), surf)
                np.save(os.path.join(save_path, "edge_index.npy"), edge_index)

        surf = torch.tensor(surf)
        pos = torch.tensor(pos)
        x = torch.tensor(init)
        y = torch.tensor(target)
        edge_index = torch.tensor(edge_index)

        if norm and coef_norm is None:
            if k == 0:
                old_length = init.shape[0]
                mean_in = init.mean(axis=0)
                mean_out = target.mean(axis=0)
            else:
                new_length = old_length + init.shape[0]
                mean_in += (init.sum(axis=0) - init.shape[0] * mean_in) / new_length
                mean_out += (target.sum(axis=0) - init.shape[0] * mean_out) / new_length
                old_length = new_length
        data = Data(pos=pos, x=x, y=y, surf=surf.bool(), edge_index=edge_index)
        # data = Data(pos=pos, x=x, y=y, surf=surf.bool())
        dataset.append(data)

    if norm and coef_norm is None:
        for k, data in enumerate(dataset):
            if k == 0:
                old_length = data.x.numpy().shape[0]
                std_in = ((data.x.numpy() - mean_in) ** 2).sum(axis=0) / old_length
                std_out = ((data.y.numpy() - mean_out) ** 2).sum(axis=0) / old_length
            else:
                new_length = old_length + data.x.numpy().shape[0]
                std_in += (
                    ((data.x.numpy() - mean_in) ** 2).sum(axis=0)
                    - data.x.numpy().shape[0] * std_in
                ) / new_length
                std_out += (
                    ((data.y.numpy() - mean_out) ** 2).sum(axis=0)
                    - data.x.numpy().shape[0] * std_out
                ) / new_length
                old_length = new_length

        std_in = np.sqrt(std_in)
        std_out = np.sqrt(std_out)

        for data in dataset:
            data.x = ((data.x - mean_in) / (std_in + 1e-8)).float()
            data.y = ((data.y - mean_out) / (std_out + 1e-8)).float()

        coef_norm = (mean_in, std_in, mean_out, std_out)
        dataset = (dataset, coef_norm)

    elif coef_norm is not None:
        for data in dataset:
            data.x = ((data.x - coef_norm[0]) / (coef_norm[1] + 1e-8)).float()
            data.y = ((data.y - coef_norm[2]) / (coef_norm[3] + 1e-8)).float()
    # 节点坐标 pos、输入特征 x、目标值 y、表面标记 surf、边索引 edge_index，coef_norm 存储标准化参数
    return dataset


def save_prediction_to_vtk(
    out_denorm, targets, cfd_data, sample_name, output_dir, index, data_dir
):
    """将预测结果保存为VTK文件"""
    # 解析样本完整路径
    sample_fullpath = os.path.join(data_dir, sample_name)

    # 验证原始数据路径存在
    if not os.path.exists(sample_fullpath):
        raise FileNotFoundError(f"Sample directory not found: {sample_fullpath}")

    # 处理压力网格
    press_path = os.path.join(sample_fullpath, "quadpress_smpl.vtk")
    if not os.path.isfile(press_path):
        raise FileNotFoundError(f"Pressure file missing: {press_path}")

    press_reader = vtk.vtkUnstructuredGridReader()
    press_reader.SetFileName(press_path)
    press_reader.Update()
    press_grid = vtk.vtkUnstructuredGrid()
    press_grid.DeepCopy(press_reader.GetOutput())

    # 处理速度网格
    velo_path = os.path.join(sample_fullpath, "hexvelo_smpl.vtk")
    if not os.path.isfile(velo_path):
        raise FileNotFoundError(f"Velocity file missing: {velo_path}")

    velo_reader = vtk.vtkUnstructuredGridReader()
    velo_reader.SetFileName(velo_path)
    velo_reader.Update()
    velo_grid = vtk.vtkUnstructuredGrid()
    velo_grid.DeepCopy(velo_reader.GetOutput())

    # 获取预测数据
    pred_press = out_denorm[cfd_data.surf, -1].cpu().numpy().squeeze()
    pred_velo = out_denorm[~cfd_data.surf, :-1].cpu().numpy().reshape(-1, 3)

    # 更新压力数据
    press_array = numpy_to_vtk(pred_press.astype(np.float32))
    press_array.SetName("PredictedPressure")
    press_grid.GetPointData().SetScalars(press_array)

    # 更新速度数据
    points_velo = vtk_to_numpy(velo_grid.GetPoints().GetData())
    surface_points = set(
        tuple(p) for p in vtk_to_numpy(press_grid.GetPoints().GetData())
    )
    exterior_indices = [
        i for i, p in enumerate(points_velo) if tuple(p) not in surface_points
    ]

    original_velo = vtk_to_numpy(velo_grid.GetPointData().GetVectors())
    velo_array = original_velo.copy()
    velo_array[exterior_indices] = pred_velo.astype(np.float32)

    velo_vtk_array = numpy_to_vtk(velo_array, deep=True)
    velo_vtk_array.SetName("PredictedVelocity")
    velo_grid.GetPointData().SetVectors(velo_vtk_array)

    # 保存文件
    press_writer = vtk.vtkUnstructuredGridWriter()
    press_writer.SetFileName(os.path.join(output_dir, f"sample{index}_pred_press.vtk"))
    press_writer.SetInputData(press_grid)
    press_writer.Write()

    velo_writer = vtk.vtkUnstructuredGridWriter()
    velo_writer.SetFileName(os.path.join(output_dir, f"sample{index}_pred_velo.vtk"))
    velo_writer.SetInputData(velo_grid)
    velo_writer.Write()


def visualize_speed_cpu(poly_data, save_path):
    """从 poly_data 的点向量取速度模长并可视化（纯CPU）。"""
    vecs = vtk_to_numpy(poly_data.GetPointData().GetVectors())
    if vecs is None or vecs.size == 0:
        raise ValueError("poly_data 点数据中没有向量（速度）")
    speed = np.linalg.norm(vecs, axis=1)
    visualize_poly_data_cpu(
        poly_data, scalar_data=speed, colorbar_title="Speed", save_path=save_path
    )


def visualize_prediction(output_dir, vis_dir, index):
    """可视化预测的VTK结果"""
    try:
        # 构建预测文件路径
        press_path = os.path.join(output_dir, f"sample{index}_pred_press.vtk")
        velo_path = os.path.join(output_dir, f"sample{index}_pred_velo.vtk")

        # 加载预测数据
        pred_press = load_unstructured_grid_data(press_path)
        pred_velo = load_unstructured_grid_data(velo_path)

        # 可视化压力预测
        press_poly, _ = unstructured_grid_data_to_poly_data(pred_press)
        visualize_poly_data_cpu(
            press_poly,
            colorbar_title="Predicted Pressure",
            save_path=os.path.join(vis_dir, f"pred_pressure_{index}.png"),
        )

        velo_poly, _ = unstructured_grid_data_to_poly_data(pred_velo)
        visualize_speed_cpu(
            velo_poly,
            save_path=os.path.join(vis_dir, f"pred_speed_{index}.png"),
        )

    except Exception as e:
        print(f"可视化失败 index:{index} error:{str(e)}")


def get_edges(unstructured_grid_data, points, cell_size=4):  # 提取网格单元边信息
    edge_indeces = set()
    cells = vtk_to_numpy(unstructured_grid_data.GetCells().GetData()).reshape(
        -1, cell_size + 1
    )
    for i in range(len(cells)):
        for j, k in itertools.product(range(1, cell_size + 1), repeat=2):
            edge_indeces.add((cells[i][j], cells[i][k]))
            edge_indeces.add((cells[i][k], cells[i][j]))
    edges = [[], []]
    for u, v in edge_indeces:
        edges[0].append(tuple(points[u]))
        edges[1].append(tuple(points[v]))
    return edges


def get_edge_index(pos, edges_press, edges_velo):  # 合并压力/速度场边信息
    indices = {tuple(pos[i]): i for i in range(len(pos))}
    edges = set()
    for i in range(len(edges_press[0])):
        edges.add((indices[edges_press[0][i]], indices[edges_press[1][i]]))
    for i in range(len(edges_velo[0])):
        edges.add((indices[edges_velo[0][i]], indices[edges_velo[1][i]]))
    edge_index = np.array(list(edges)).T
    return edge_index


def get_induced_graph(data, idx, num_hops):  # 提取子图
    subset, sub_edge_index, _, _ = k_hop_subgraph(
        node_idx=idx, num_hops=num_hops, edge_index=data.edge_index, relabel_nodes=True
    )
    return Data(x=data.x[subset], y=data.y[idx], edge_index=sub_edge_index)


def pc_normalize(pc):  # 点云归一化
    centroid = torch.mean(pc, axis=0)
    pc = pc - centroid
    m = torch.max(torch.sqrt(torch.sum(pc**2, axis=1)))
    pc = pc / m
    return pc


def get_shape(
    data, max_n_point=8192, normalize=True, use_height=False
):  # 提取表面形状特征
    surf_indices = torch.where(data.surf)[0].tolist()

    if len(surf_indices) > max_n_point:
        surf_indices = np.array(random.sample(range(len(surf_indices)), max_n_point))

    shape_pc = data.pos[surf_indices].clone()

    if normalize:
        shape_pc = pc_normalize(shape_pc)

    if use_height:
        gravity_dim = 1
        height_array = (
            shape_pc[:, gravity_dim : gravity_dim + 1]
            - shape_pc[:, gravity_dim : gravity_dim + 1].min()
        )
        shape_pc = torch.cat((shape_pc, height_array), axis=1)

    return shape_pc


def create_edge_index_radius(data, r, max_neighbors=32):  # 基于半径构建邻域图
    data.edge_index = nng.radius_graph(
        x=data.pos, r=r, loop=True, max_num_neighbors=max_neighbors
    )
    # print(f'r = {r}, #edges = {data.edge_index.size(1)}')
    return data


class GraphDataset(Dataset):  # PyG数据集封装
    def __init__(self, datalist, use_height=False, use_cfd_mesh=True, r=None):
        super().__init__()
        self.datalist = datalist
        self.use_height = use_height
        if not use_cfd_mesh:
            assert r is not None
            for i in range(len(self.datalist)):
                self.datalist[i] = create_edge_index_radius(self.datalist[i], r)

    def len(self):
        return len(self.datalist)

    def get(self, idx):
        data = self.datalist[idx]
        shape = get_shape(data, use_height=self.use_height)
        return self.datalist[idx], shape


def cell_sampling_2d(cell_points, cell_attr=None):
    """
    Sample points in a two dimensional cell via parallelogram sampling and triangle interpolation via barycentric coordinates. The vertices have to be ordered in a certain way.

    Args:
        cell_points (array): Vertices of the 2 dimensional cells. Shape (N, 4) for N cells with 4 vertices.
        cell_attr (array, optional): Features of the vertices of the 2 dimensional cells. Shape (N, 4, k) for N cells with 4 edges and k features.
            If given shape (N, 4) it will resize it automatically in a (N, 4, 1) array. Default: ``None``
    """
    # Sampling via triangulation of the cell and parallelogram sampling
    v0, v1 = (
        cell_points[:, 1] - cell_points[:, 0],
        cell_points[:, 3] - cell_points[:, 0],
    )
    v2, v3 = (
        cell_points[:, 3] - cell_points[:, 2],
        cell_points[:, 1] - cell_points[:, 2],
    )
    a0, a1 = np.abs(
        np.linalg.det(np.hstack([v0[:, :2], v1[:, :2]]).reshape(-1, 2, 2))
    ), np.abs(np.linalg.det(np.hstack([v2[:, :2], v3[:, :2]]).reshape(-1, 2, 2)))
    p = a0 / (a0 + a1)
    index_triangle = np.random.binomial(1, p)[:, None]
    u = np.random.uniform(size=(len(p), 2))
    sampled_point = index_triangle * (u[:, 0:1] * v0 + u[:, 1:2] * v1) + (
        1 - index_triangle
    ) * (u[:, 0:1] * v2 + u[:, 1:2] * v3)
    sampled_point_mirror = index_triangle * (
        (1 - u[:, 0:1]) * v0 + (1 - u[:, 1:2]) * v1
    ) + (1 - index_triangle) * ((1 - u[:, 0:1]) * v2 + (1 - u[:, 1:2]) * v3)
    reflex = u.sum(axis=1) > 1
    sampled_point[reflex] = sampled_point_mirror[reflex]

    # Interpolation on a triangle via barycentric coordinates
    if cell_attr is not None:
        t0, t1, t2 = (
            np.zeros_like(v0),
            index_triangle * v0 + (1 - index_triangle) * v2,
            index_triangle * v1 + (1 - index_triangle) * v3,
        )
        w = (t1[:, 1] - t2[:, 1]) * (t0[:, 0] - t2[:, 0]) + (t2[:, 0] - t1[:, 0]) * (
            t0[:, 1] - t2[:, 1]
        )
        w0 = (t1[:, 1] - t2[:, 1]) * (sampled_point[:, 0] - t2[:, 0]) + (
            t2[:, 0] - t1[:, 0]
        ) * (sampled_point[:, 1] - t2[:, 1])
        w1 = (t2[:, 1] - t0[:, 1]) * (sampled_point[:, 0] - t2[:, 0]) + (
            t0[:, 0] - t2[:, 0]
        ) * (sampled_point[:, 1] - t2[:, 1])
        w0, w1 = w0 / w, w1 / w
        w2 = 1 - w0 - w1

        if len(cell_attr.shape) == 2:
            cell_attr = cell_attr[:, :, None]
        attr0 = (
            index_triangle * cell_attr[:, 0] + (1 - index_triangle) * cell_attr[:, 2]
        )
        attr1 = (
            index_triangle * cell_attr[:, 1] + (1 - index_triangle) * cell_attr[:, 1]
        )
        attr2 = (
            index_triangle * cell_attr[:, 3] + (1 - index_triangle) * cell_attr[:, 3]
        )
        sampled_attr = w0[:, None] * attr0 + w1[:, None] * attr1 + w2[:, None] * attr2

    sampled_point += (
        index_triangle * cell_points[:, 0] + (1 - index_triangle) * cell_points[:, 2]
    )

    return (
        np.hstack([sampled_point[:, :2], sampled_attr])
        if cell_attr is not None
        else sampled_point[:, :2]
    )


def cell_sampling_1d(line_points, line_attr=None):
    """
    Sample points in a one dimensional cell via linear sampling and interpolation.

    Args:
        line_points (array): Edges of the 1 dimensional cells. Shape (N, 2) for N cells with 2 edges.
        line_attr (array, optional): Features of the edges of the 1 dimensional cells. Shape (N, 2, k) for N cells with 2 edges and k features.
            If given shape (N, 2) it will resize it automatically in a (N, 2, 1) array. Default: ``None``
    """
    # Linear sampling
    u = np.random.uniform(size=(len(line_points), 1))
    sampled_point = u * line_points[:, 0] + (1 - u) * line_points[:, 1]

    # Linear interpolation
    if line_attr is not None:
        if len(line_attr.shape) == 2:
            line_attr = line_attr[:, :, None]
        sampled_attr = u * line_attr[:, 0] + (1 - u) * line_attr[:, 1]

    return (
        np.hstack([sampled_point[:, :2], sampled_attr])
        if line_attr is not None
        else sampled_point[:, :2]
    )


def get_airfoildatalist(
    set,
    norm=False,
    coef_norm=None,
    crop=None,
    sample=None,
    n_boot=int(5e5),
    surf_ratio=0.1,
    data_path="/data/path",
):
    """
    Create a list of simulation to input in a PyTorch Geometric DataLoader. Simulation are transformed by keeping vertices of the CFD mesh or
    by sampling (uniformly or via the mesh density) points in the simulation cells.

    Args:
        set (list): List of geometry names to include in the dataset.
        norm (bool, optional): If norm is set to ``True``, the mean and the standard deviation of the dataset will be computed and returned.
            Moreover, the dataset will be normalized by these quantities. Ignored when ``coef_norm`` is not None. Default: ``False``
        coef_norm (tuple, optional): This has to be a tuple of the form (mean input, std input, mean output, std ouput) if not None.
            The dataset generated will be normalized by those quantites. Default: ``None``
        crop (list, optional): List of the vertices of the rectangular [xmin, xmax, ymin, ymax] box to crop simulations. Default: ``None``
        sample (string, optional): Type of sampling. If ``None``, no sampling strategy is applied and the nodes of the CFD mesh are returned.
            If ``uniform`` or ``mesh`` is chosen, uniform or mesh density sampling is applied on the domain. Default: ``None``
        n_boot (int, optional): Used only if sample is not None, gives the size of the sampling for each simulation. Defaul: ``int(5e5)``
        surf_ratio (float, optional): Used only if sample is not None, gives the ratio of point over the airfoil to sample with respect to point
            in the volume. Default: ``0.1``
    """
    if norm and coef_norm is not None:
        raise ValueError(
            "If coef_norm is not None and norm is True, the normalization will be done via coef_norm"
        )

    dataset = []

    for k, s in enumerate(tqdm(set)):
        # Get the 3D mesh, add the signed distance function and slice it to return in 2D
        internal = pv.read(osp.join(data_path, s, s + "_internal.vtu"))
        aerofoil = pv.read(osp.join(data_path, s, s + "_aerofoil.vtp"))
        internal = internal.compute_cell_sizes(length=False, volume=False)

        # Cropping if needed, crinkle is True.
        if crop is not None:
            bounds = (crop[0], crop[1], crop[2], crop[3], 0, 1)
            internal = internal.clip_box(bounds=bounds, invert=False, crinkle=True)

        # If sampling strategy is chosen, it will sample points in the cells of the simulation instead of directly taking the nodes of the mesh.
        if sample is not None:
            # Sample on a new point cloud
            if sample == "uniform":  # Uniform sampling strategy
                p = internal.cell_data["Area"] / internal.cell_data["Area"].sum()
                sampled_cell_indices = np.random.choice(
                    internal.n_cells, size=n_boot, p=p
                )
                surf_p = (
                    aerofoil.cell_data["Length"] / aerofoil.cell_data["Length"].sum()
                )
                sampled_line_indices = np.random.choice(
                    aerofoil.n_cells, size=int(n_boot * surf_ratio), p=surf_p
                )
            elif sample == "mesh":  # Sample via mesh density
                sampled_cell_indices = np.random.choice(internal.n_cells, size=n_boot)
                sampled_line_indices = np.random.choice(
                    aerofoil.n_cells, size=int(n_boot * surf_ratio)
                )

            cell_dict = internal.cells.reshape(-1, 5)[sampled_cell_indices, 1:]
            cell_points = internal.points[cell_dict]
            line_dict = aerofoil.lines.reshape(-1, 3)[sampled_line_indices, 1:]
            line_points = aerofoil.points[line_dict]

            # Geometry information
            geom = -internal.point_data["implicit_distance"][
                cell_dict, None
            ]  # Signed distance function
            Uinf, alpha = float(s.split("_")[2]), float(s.split("_")[3]) * np.pi / 180
            # u = (np.array([np.cos(alpha), np.sin(alpha)])*Uinf).reshape(1, 2)*(internal.point_data['U'][cell_dict, :1] != 0)
            u = (np.array([np.cos(alpha), np.sin(alpha)]) * Uinf).reshape(
                1, 2
            ) * np.ones_like(internal.point_data["U"][cell_dict, :1])
            normal = np.zeros_like(u)

            surf_geom = np.zeros_like(aerofoil.point_data["U"][line_dict, :1])
            # surf_u = np.zeros_like(aerofoil.point_data['U'][line_dict, :2])
            surf_u = (np.array([np.cos(alpha), np.sin(alpha)]) * Uinf).reshape(
                1, 2
            ) * np.ones_like(aerofoil.point_data["U"][line_dict, :1])
            surf_normal = -aerofoil.point_data["Normals"][line_dict, :2]

            attr = np.concatenate(
                [
                    u,
                    geom,
                    normal,
                    internal.point_data["U"][cell_dict, :2],
                    internal.point_data["p"][cell_dict, None],
                    internal.point_data["nut"][cell_dict, None],
                ],
                axis=-1,
            )
            surf_attr = np.concatenate(
                [
                    surf_u,
                    surf_geom,
                    surf_normal,
                    aerofoil.point_data["U"][line_dict, :2],
                    aerofoil.point_data["p"][line_dict, None],
                    aerofoil.point_data["nut"][line_dict, None],
                ],
                axis=-1,
            )
            sampled_points = cell_sampling_2d(cell_points, attr)
            surf_sampled_points = cell_sampling_1d(line_points, surf_attr)

            # Define the inputs and the targets
            pos = sampled_points[:, :2]
            init = sampled_points[:, :7]
            target = sampled_points[:, 7:]
            surf_pos = surf_sampled_points[:, :2]
            surf_init = surf_sampled_points[:, :7]
            surf_target = surf_sampled_points[:, 7:]

            # Put everything in tensor
            surf = torch.cat([torch.zeros(len(pos)), torch.ones(len(surf_pos))], dim=0)
            pos = torch.cat(
                [
                    torch.tensor(pos, dtype=torch.float),
                    torch.tensor(surf_pos, dtype=torch.float),
                ],
                dim=0,
            )
            x = torch.cat(
                [
                    torch.tensor(init, dtype=torch.float),
                    torch.tensor(surf_init, dtype=torch.float),
                ],
                dim=0,
            )
            y = torch.cat(
                [
                    torch.tensor(target, dtype=torch.float),
                    torch.tensor(surf_target, dtype=torch.float),
                ],
                dim=0,
            )

        else:  # Keep the mesh nodes
            surf_bool = internal.point_data["U"][:, 0] == 0
            geom = -internal.point_data["implicit_distance"][
                :, None
            ]  # Signed distance function
            Uinf, alpha = float(s.split("_")[2]), float(s.split("_")[3]) * np.pi / 180
            u = (np.array([np.cos(alpha), np.sin(alpha)]) * Uinf).reshape(
                1, 2
            ) * np.ones_like(internal.point_data["U"][:, :1])
            normal = np.zeros_like(u)
            normal[surf_bool] = reorganize(
                aerofoil.points[:, :2],
                internal.points[surf_bool, :2],
                -aerofoil.point_data["Normals"][:, :2],
            )

            attr = np.concatenate(
                [
                    u,
                    geom,
                    normal,
                    internal.point_data["U"][:, :2],
                    internal.point_data["p"][:, None],
                    internal.point_data["nut"][:, None],
                ],
                axis=-1,
            )

            pos = internal.points[:, :2]
            init = np.concatenate([pos, attr[:, :5]], axis=1)
            target = attr[:, 5:]

            # Put everything in tensor
            surf = torch.tensor(surf_bool)
            pos = torch.tensor(pos, dtype=torch.float)
            x = torch.tensor(init, dtype=torch.float)
            y = torch.tensor(target, dtype=torch.float)

        if norm and coef_norm is None:
            if k == 0:
                old_length = init.shape[0]
                mean_in = init.mean(axis=0, dtype=np.double)
                mean_out = target.mean(axis=0, dtype=np.double)
            else:
                new_length = old_length + init.shape[0]
                mean_in += (
                    init.sum(axis=0, dtype=np.double) - init.shape[0] * mean_in
                ) / new_length
                mean_out += (
                    target.sum(axis=0, dtype=np.double) - init.shape[0] * mean_out
                ) / new_length
                old_length = new_length

        data = Data(pos=pos, x=x, y=y, surf=surf.bool())
        dataset.append(data)

    if norm and coef_norm is None:
        # Compute normalization
        mean_in = mean_in.astype(np.single)
        mean_out = mean_out.astype(np.single)
        for k, data in enumerate(dataset):

            if k == 0:
                old_length = data.x.numpy().shape[0]
                std_in = ((data.x.numpy() - mean_in) ** 2).sum(
                    axis=0, dtype=np.double
                ) / old_length
                std_out = ((data.y.numpy() - mean_out) ** 2).sum(
                    axis=0, dtype=np.double
                ) / old_length
            else:
                new_length = old_length + data.x.numpy().shape[0]
                std_in += (
                    ((data.x.numpy() - mean_in) ** 2).sum(axis=0, dtype=np.double)
                    - data.x.numpy().shape[0] * std_in
                ) / new_length
                std_out += (
                    ((data.y.numpy() - mean_out) ** 2).sum(axis=0, dtype=np.double)
                    - data.x.numpy().shape[0] * std_out
                ) / new_length
                old_length = new_length

        std_in = np.sqrt(std_in).astype(np.single)
        std_out = np.sqrt(std_out).astype(np.single)

        # Normalize
        for data in dataset:
            data.x = (data.x - mean_in) / (std_in + 1e-8)
            data.y = (data.y - mean_out) / (std_out + 1e-8)

        coef_norm = (mean_in, std_in, mean_out, std_out)
        dataset = (dataset, coef_norm)

    elif coef_norm is not None:
        # Normalize
        for data in dataset:
            data.x = (data.x - coef_norm[0]) / (coef_norm[1] + 1e-8)
            data.y = (data.y - coef_norm[2]) / (coef_norm[3] + 1e-8)

    return dataset


if __name__ == "__main__":
    import numpy as np

    file_name = "1a0bc9ab92c915167ae33d942430658c"

    root = (
        "../../../examples/cfd/Transolver-Car-Design/dataset/mlcfd_data/training_data"
    )
    save_path = (
        "../../../../examples/cfd/Transolver-Car-Design/dataset/mlcfd_data/preprocessed_data/param0/"
        + file_name
    )
    file_name_press = "param0/" + file_name + "/quadpress_smpl.vtk"
    file_name_velo = "param0/" + file_name + "/hexvelo_smpl.vtk"

    file_name_press = os.path.join(root, file_name_press)
    file_name_velo = os.path.join(root, file_name_velo)

    unstructured_grid_data_press = load_unstructured_grid_data(file_name_press)
    unstructured_grid_data_velo = load_unstructured_grid_data(file_name_velo)

    velo = vtk_to_numpy(unstructured_grid_data_velo.GetPointData().GetVectors())
    press = vtk_to_numpy(unstructured_grid_data_press.GetPointData().GetScalars())
    points_velo = vtk_to_numpy(unstructured_grid_data_velo.GetPoints().GetData())
    points_press = vtk_to_numpy(unstructured_grid_data_press.GetPoints().GetData())

    edges_press = get_edges(unstructured_grid_data_press, points_press, cell_size=4)
    edges_velo = get_edges(unstructured_grid_data_velo, points_velo, cell_size=8)

    sdf_velo, normal_velo = get_sdf(points_velo, points_press)
    sdf_press = np.zeros(points_press.shape[0])
    normal_press = get_normal(unstructured_grid_data_press)

    surface = {tuple(p) for p in points_press}
    exterior_indices = [i for i, p in enumerate(points_velo) if tuple(p) not in surface]
    velo_dict = {tuple(p): velo[i] for i, p in enumerate(points_velo)}

    pos_ext = points_velo[exterior_indices]
    pos_surf = points_press
    sdf_ext = sdf_velo[exterior_indices]
    sdf_surf = sdf_press
    normal_ext = normal_velo[exterior_indices]
    normal_surf = normal_press
    velo_ext = velo[exterior_indices]
    velo_surf = np.array(
        [
            velo_dict[tuple(p)] if tuple(p) in velo_dict else np.zeros(3)
            for p in pos_surf
        ]
    )
    press_ext = np.zeros([len(exterior_indices), 1])
    press_surf = press

    init_ext = np.c_[pos_ext, sdf_ext, normal_ext]
    init_surf = np.c_[pos_surf, sdf_surf, normal_surf]
    target_ext = np.c_[velo_ext, press_ext]
    target_surf = np.c_[velo_surf, press_surf]

    surf = np.concatenate([np.zeros(len(pos_ext)), np.ones(len(pos_surf))])
    pos = np.concatenate([pos_ext, pos_surf])
    init = np.concatenate([init_ext, init_surf])
    target = np.concatenate([target_ext, target_surf])

    edge_index = get_edge_index(pos, edges_press, edges_velo)

    data = Data(pos=torch.tensor(pos), edge_index=torch.tensor(edge_index))
    data = create_edge_index_radius(data, r=0.2)
    x, y = data.edge_index
    import torch_geometric

    print(max(torch_geometric.utils.degree(x)), max(torch_geometric.utils.degree(y)))

    print(points_velo.shape, points_press.shape)
    print(surf.shape, pos.shape, init.shape, target.shape, edge_index.shape)

    # 示例：可视化压力网格表面并保存
    poly_data, surface_filter = unstructured_grid_data_to_poly_data(
        unstructured_grid_data_press
    )
    # 可视化压力网格，并指定颜色条标题为"Pressure"
    visualize_poly_data(
        poly_data,
        surface_filter,
        colorbar_title="Pressure",
        save_path="pressure_surface_mesh.png",  # 指定保存路径
    )
    # 调用示例
    poly_data_velo, surface_filter_velo = unstructured_grid_data_to_poly_data(
        unstructured_grid_data_velo
    )
    # 从表面网格数据获取速度
    speed_data = get_speed_from_poly_data(poly_data_velo)  # 关键修改点
    # 可视化速度网格，并指定颜色条标题为"Speed"
    visualize_poly_data(
        poly_data_velo,
        surface_filter_velo,
        scalar_data=speed_data,
        colorbar_title="Speed",
        save_path="speed_surface_with_colors.png",
    )
