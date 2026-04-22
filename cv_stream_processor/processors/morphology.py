"""
형태학적 연산 프로세서 (processors/morphology.py)
침식, 팽창, 열기, 닫기, 그라디언트, 탑햇, 블랙햇 등
"""

import cv2
import numpy as np
from ..core.base_processor import BaseProcessor
from ..core.frame import Frame

# OpenCV 형태학 연산 매핑
MORPH_OPS = {
    "erode"    : cv2.MORPH_ERODE,
    "dilate"   : cv2.MORPH_DILATE,
    "open"     : cv2.MORPH_OPEN,
    "close"    : cv2.MORPH_CLOSE,
    "gradient" : cv2.MORPH_GRADIENT,
    "tophat"   : cv2.MORPH_TOPHAT,
    "blackhat" : cv2.MORPH_BLACKHAT,
}

KERNEL_SHAPES = {
    "rect"    : cv2.MORPH_RECT,
    "ellipse" : cv2.MORPH_ELLIPSE,
    "cross"   : cv2.MORPH_CROSS,
}


class MorphologyProcessor(BaseProcessor):
    """
    형태학적 연산 프로세서.
    params:
        operation  : "erode"|"dilate"|"open"|"close"|"gradient"|"tophat"|"blackhat"
        kernel_size: 커널 크기 (홀수, 기본 5)
        kernel_shape: "rect"|"ellipse"|"cross" (기본 "rect")
        iterations : 반복 횟수 (기본 1)
    """

    def __init__(self) -> None:
        super().__init__(
            name="morphology",
            description="형태학적 연산 (팽창/침식 등)",
            params={
                "operation"  : "dilate",
                "kernel_size": 5,
                "kernel_shape": "rect",
                "iterations" : 1,
            },
        )

    def _get_kernel(self) -> np.ndarray:
        k = self.get_param("kernel_size", 5)
        k = k if k % 2 == 1 else k + 1
        shape = KERNEL_SHAPES.get(self.get_param("kernel_shape", "rect"), cv2.MORPH_RECT)
        return cv2.getStructuringElement(shape, (k, k))

    def process(self, frame: Frame) -> Frame:
        op_name = self.get_param("operation", "dilate")
        op = MORPH_OPS.get(op_name, cv2.MORPH_DILATE)
        kernel = self._get_kernel()
        iters = self.get_param("iterations", 1)

        frame.processed = cv2.morphologyEx(frame.output, op, kernel, iterations=iters)
        return frame


# ── 이진화 + 형태학 콤보 ─────────────────────────────────────────────

class ThresholdMorphProcessor(BaseProcessor):
    """
    그레이스케일 이진화 후 형태학 연산을 결합합니다.
    params:
        thresh_val : 이진화 임계값 (0이면 Otsu 자동 계산)
        morph_op   : "dilate" | "erode" | "close" | "open"
        ksize      : 형태학 커널 크기 (기본 3)
    """

    def __init__(self) -> None:
        super().__init__(
            name="thresh_morph",
            description="이진화 + 형태학 연산",
            params={"thresh_val": 0, "morph_op": "close", "ksize": 3},
        )

    def process(self, frame: Frame) -> Frame:
        gray = cv2.cvtColor(frame.output, cv2.COLOR_BGR2GRAY)
        tv = self.get_param("thresh_val", 0)

        if tv == 0:
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        else:
            _, binary = cv2.threshold(gray, tv, 255, cv2.THRESH_BINARY)

        k = self.get_param("ksize", 3)
        k = k if k % 2 == 1 else k + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
        op = MORPH_OPS.get(self.get_param("morph_op", "close"), cv2.MORPH_CLOSE)
        result = cv2.morphologyEx(binary, op, kernel)

        frame.processed = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
        return frame
