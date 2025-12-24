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
