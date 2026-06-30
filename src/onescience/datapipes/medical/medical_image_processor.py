# 医学图像处理器
# 预处理 CT、CXR、WSI 等医学图像

import logging
from typing import Any, Dict, Optional, Tuple, Union
import numpy as np

logger = logging.getLogger(__name__)


class MedicalImageProcessor:
    """
    医学图像预处理器
    支持 CT、胸部 X 光（CXR）、全切片图像（WSI）等
    """

    def __init__(
        self,
        target_size: Tuple[int, int] = (224, 224),
        normalize: bool = True,
    ):
        """
        初始化图像处理器

        Args:
            target_size: 目标图像尺寸 (height, width)
            normalize: 是否归一化到 [0, 1]
        """
        self.target_size = target_size
        self.normalize = normalize

        # 检查 opencv 和 PIL
        self.cv2_available = self._check_cv2()
        self.pil_available = self._check_pil()

    def _check_cv2(self) -> bool:
        """检查 OpenCV 是否可用"""
        try:
            import cv2
            return True
        except ImportError:
            logger.warning("opencv-python not installed")
            return False

    def _check_pil(self) -> bool:
        """检查 PIL 是否可用"""
        try:
            from PIL import Image
            return True
        except ImportError:
            logger.warning("Pillow not installed")
            return False

    def resize_image(
        self,
        image: np.ndarray,
        target_size: Optional[Tuple[int, int]] = None
    ) -> np.ndarray:
        """
        调整图像大小

        Args:
            image: 输入图像
            target_size: 目标尺寸 (height, width)

        Returns:
            调整后的图像
        """
        if target_size is None:
            target_size = self.target_size

        if self.cv2_available:
            import cv2
            # OpenCV 使用 (width, height) 顺序
            resized = cv2.resize(image, (target_size[1], target_size[0]))
        elif self.pil_available:
            from PIL import Image
            pil_image = Image.fromarray(image)
            pil_image = pil_image.resize((target_size[1], target_size[0]))
            resized = np.array(pil_image)
        else:
            raise RuntimeError("Neither opencv-python nor Pillow is available")

        return resized

    def normalize_image(self, image: np.ndarray) -> np.ndarray:
        """
        归一化图像到 [0, 1]

        Args:
            image: 输入图像

        Returns:
            归一化的图像
        """
        image = image.astype(np.float32)
        if image.max() > 1.0:
            image = image / 255.0
        return image

    def process_ct_image(
        self,
        image: np.ndarray,
        window_center: float = 40,
        window_width: float = 400,
    ) -> np.ndarray:
        """
        处理 CT 图像

        Args:
            image: CT 图像（Hounsfield 单位）
            window_center: 窗位
            window_width: 窗宽

        Returns:
            处理后的图像
        """
        # 应用窗位窗宽
        img_min = window_center - window_width // 2
        img_max = window_center + window_width // 2

        image = np.clip(image, img_min, img_max)
        image = (image - img_min) / (img_max - img_min)

        # 调整大小
        image = self.resize_image((image * 255).astype(np.uint8))

        # 归一化
        if self.normalize:
            image = self.normalize_image(image)

        return image

    def process_cxr_image(
        self,
        image: np.ndarray,
        clahe: bool = True
    ) -> np.ndarray:
        """
        处理胸部 X 光图像

        Args:
            image: CXR 图像
            clahe: 是否应用 CLAHE（对比度受限的自适应直方图均衡化）

        Returns:
            处理后的图像
        """
        # 转换为 uint8
        if image.dtype != np.uint8:
            image = ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)

        # 应用 CLAHE
        if clahe and self.cv2_available:
            import cv2
            clahe_obj = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            if len(image.shape) == 2:
                image = clahe_obj.apply(image)
            elif len(image.shape) == 3:
                # 对每个通道应用
                for i in range(image.shape[2]):
                    image[:, :, i] = clahe_obj.apply(image[:, :, i])

        # 调整大小
        image = self.resize_image(image)

        # 归一化
        if self.normalize:
            image = self.normalize_image(image)

        return image

    def process_wsi_patch(
        self,
        image: np.ndarray,
        color_normalization: bool = True
    ) -> np.ndarray:
        """
        处理全切片图像（WSI）的 patch

        Args:
            image: WSI patch
            color_normalization: 是否进行颜色归一化

        Returns:
            处理后的图像
        """
        # 调整大小
        image = self.resize_image(image)

        # 颜色归一化（可选）
        if color_normalization:
            image = self._normalize_stain(image)

        # 归一化
        if self.normalize:
            image = self.normalize_image(image)

        return image

    def _normalize_stain(self, image: np.ndarray) -> np.ndarray:
        """
        染色归一化（简化版）

        Args:
            image: RGB 图像

        Returns:
            归一化的图像
        """
        # 简单的颜色归一化
        # 实际应用中可以使用更复杂的方法（如 Macenko、Reinhard 等）
        image = image.astype(np.float32)

        # 对每个通道进行归一化
        for i in range(3):
            channel = image[:, :, i]
            mean = channel.mean()
            std = channel.std()
            if std > 0:
                image[:, :, i] = (channel - mean) / std * 50 + 128

        image = np.clip(image, 0, 255).astype(np.uint8)
        return image

    def convert_to_rgb(self, image: np.ndarray) -> np.ndarray:
        """
        转换图像为 RGB 格式

        Args:
            image: 输入图像

        Returns:
            RGB 图像
        """
        if len(image.shape) == 2:
            # 灰度图转 RGB
            image = np.stack([image] * 3, axis=-1)
        elif len(image.shape) == 3 and image.shape[2] == 1:
            # 单通道转 RGB
            image = np.repeat(image, 3, axis=-1)

        return image

    def preprocess_for_model(
        self,
        image: np.ndarray,
        modality: str = "CT"
    ) -> np.ndarray:
        """
        根据影像模态预处理图像

        Args:
            image: 输入图像
            modality: 影像模态（CT、CXR、MRI、WSI）

        Returns:
            预处理后的图像
        """
        if modality.upper() == "CT":
            return self.process_ct_image(image)
        elif modality.upper() in ["CXR", "XRAY", "X-RAY"]:
            return self.process_cxr_image(image)
        elif modality.upper() == "WSI":
            return self.process_wsi_patch(image)
        elif modality.upper() == "MRI":
            # MRI 处理类似 CT
            return self.process_ct_image(image, window_center=500, window_width=1000)
        else:
            # 默认处理
            image = self.resize_image(image)
            if self.normalize:
                image = self.normalize_image(image)
            return image
