# 医学数据加载器 - DICOM 加载器
# 加载和处理 DICOM 医学图像

import logging
from typing import Any, Dict, Optional, Union
import numpy as np

logger = logging.getLogger(__name__)


class DICOMLoader:
    """
    DICOM 医学图像加载器
    支持 CT、MRI、X-ray 等 DICOM 格式
    """

    def __init__(self):
        """初始化 DICOM 加载器"""
        self.pydicom_available = self._check_pydicom()

    def _check_pydicom(self) -> bool:
        """检查 pydicom 是否可用"""
        try:
            import pydicom
            return True
        except ImportError:
            logger.warning("pydicom not installed. DICOM loading will not be available.")
            return False

    def load_dicom(self, dicom_path: str) -> Dict[str, Any]:
        """
        加载 DICOM 文件

        Args:
            dicom_path: DICOM 文件路径

        Returns:
            包含图像数据和元数据的字典
        """
        if not self.pydicom_available:
            raise RuntimeError("pydicom is required for DICOM loading. Install with: pip install pydicom")

        import pydicom

        # 读取 DICOM 文件
        ds = pydicom.dcmread(dicom_path)

        # 提取图像数据
        image_data = ds.pixel_array

        # 提取元数据
        metadata = {
            "patient_id": getattr(ds, "PatientID", None),
            "patient_name": str(getattr(ds, "PatientName", "")),
            "modality": getattr(ds, "Modality", None),
            "study_description": getattr(ds, "StudyDescription", None),
            "series_description": getattr(ds, "SeriesDescription", None),
            "image_type": getattr(ds, "ImageType", None),
            "rows": getattr(ds, "Rows", None),
            "columns": getattr(ds, "Columns", None),
            "pixel_spacing": getattr(ds, "PixelSpacing", None),
            "slice_thickness": getattr(ds, "SliceThickness", None),
            "window_center": getattr(ds, "WindowCenter", None),
            "window_width": getattr(ds, "WindowWidth", None),
        }

        return {
            "image": image_data,
            "metadata": metadata,
            "dicom_dataset": ds,
        }

    def load_dicom_series(self, dicom_dir: str) -> Dict[str, Any]:
        """
        加载 DICOM 序列（多个文件）

        Args:
            dicom_dir: 包含 DICOM 文件的目录

        Returns:
            包含 3D 图像数据和元数据的字典
        """
        if not self.pydicom_available:
            raise RuntimeError("pydicom is required")

        import pydicom
        import os
        from pathlib import Path

        # 获取所有 DICOM 文件
        dicom_files = []
        for root, dirs, files in os.walk(dicom_dir):
            for file in files:
                if file.endswith('.dcm') or file.endswith('.DCM'):
                    dicom_files.append(os.path.join(root, file))

        if not dicom_files:
            raise ValueError(f"No DICOM files found in {dicom_dir}")

        # 读取所有切片
        slices = []
        for filepath in dicom_files:
            ds = pydicom.dcmread(filepath)
            slices.append(ds)

        # 按切片位置排序
        slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))

        # 堆叠为 3D 数组
        image_3d = np.stack([s.pixel_array for s in slices], axis=0)

        # 提取元数据（从第一个切片）
        first_slice = slices[0]
        metadata = {
            "patient_id": getattr(first_slice, "PatientID", None),
            "modality": getattr(first_slice, "Modality", None),
            "num_slices": len(slices),
            "image_shape": image_3d.shape,
            "pixel_spacing": getattr(first_slice, "PixelSpacing", None),
            "slice_thickness": getattr(first_slice, "SliceThickness", None),
        }

        return {
            "image_3d": image_3d,
            "metadata": metadata,
            "slices": slices,
        }

    def apply_windowing(
        self,
        image: np.ndarray,
        window_center: float,
        window_width: float
    ) -> np.ndarray:
        """
        应用窗位窗宽

        Args:
            image: 原始图像数据
            window_center: 窗位
            window_width: 窗宽

        Returns:
            应用窗位窗宽后的图像
        """
        img_min = window_center - window_width // 2
        img_max = window_center + window_width // 2

        windowed = np.clip(image, img_min, img_max)
        windowed = (windowed - img_min) / (img_max - img_min) * 255

        return windowed.astype(np.uint8)

    def normalize_hounsfield(self, image: np.ndarray) -> np.ndarray:
        """
        归一化 CT Hounsfield 单位

        Args:
            image: CT 图像（HU 单位）

        Returns:
            归一化的图像
        """
        # CT 典型范围：-1000 (空气) 到 +3000 (骨骼)
        # 常用范围：-1000 到 +400
        hu_min = -1000
        hu_max = 400

        normalized = np.clip(image, hu_min, hu_max)
        normalized = (normalized - hu_min) / (hu_max - hu_min)

        return normalized
