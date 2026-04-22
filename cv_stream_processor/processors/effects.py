"""
특수 효과 프로세서 (processors/effects.py)
카툰 효과, 스케치, 엠보싱, 픽셀화, 색 양자화, 열화상 등
"""

import cv2
import numpy as np
from ..core.base_processor import BaseProcessor
from ..core.frame import Frame


# ── 1. 카툰 효과 ─────────────────────────────────────────────────────

class CartoonProcessor(BaseProcessor):
    """
    엣지 검출 + 바이래터럴 필터로 카툰 스타일 효과.
    params:
        num_bilateral: 바이래터럴 반복 횟수 (기본 7)
        block_size   : 엣지 블록 크기 (기본 9)
        c            : 적응형 임계값 상수 (기본 9)
    """

    def __init__(self) -> None:
        super().__init__(
            name="cartoon",
            description="카툰 효과",
            params={"num_bilateral": 7, "block_size": 9, "c": 9},
        )

    def process(self, frame: Frame) -> Frame:
        img = frame.output
        n = self.get_param("num_bilateral", 7)

        # 색상 평탄화 (바이래터럴 반복)
        color = img.copy()
        for _ in range(n):
            color = cv2.bilateralFilter(color, 9, 300, 300)

        # 엣지 마스크
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 7)
        bs = self.get_param("block_size", 9)
        bs = bs if bs % 2 == 1 else bs + 1
        edges = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY,
            bs,
            self.get_param("c", 9),
        )
        edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        frame.processed = cv2.bitwise_and(color, edges)
        return frame


# ── 2. 연필 스케치 효과 ──────────────────────────────────────────────

class SketchProcessor(BaseProcessor):
    """
    cv2.pencilSketch 기반 스케치 효과.
    params:
        sigma_s  : 공간 범위 (기본 60)
        sigma_r  : 색 범위 (기본 0.07)
        shade_fac: 음영 강도 (기본 0.05)
        color    : True이면 컬러 스케치 (기본 False)
    """

    def __init__(self) -> None:
        super().__init__(
            name="sketch",
            description="연필 스케치 효과",
            params={"sigma_s": 60, "sigma_r": 0.07, "shade_fac": 0.05, "color": False},
        )

    def process(self, frame: Frame) -> Frame:
        gray_sketch, color_sketch = cv2.pencilSketch(
            frame.output,
            sigma_s=self.get_param("sigma_s", 60),
            sigma_r=self.get_param("sigma_r", 0.07),
            shade_factor=self.get_param("shade_fac", 0.05),
        )
        if self.get_param("color", False):
            frame.processed = color_sketch
        else:
            frame.processed = cv2.cvtColor(gray_sketch, cv2.COLOR_GRAY2BGR)
        return frame


# ── 3. 엠보싱 효과 ───────────────────────────────────────────────────

class EmbossProcessor(BaseProcessor):
    """
    엠보싱 커널을 이용한 부조(浮彫) 효과.
    params:
        direction: "top_left" | "top_right" | "bottom_left" | "bottom_right"
    """

    KERNELS = {
        "top_left"    : np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]], dtype=np.float32),
        "top_right"   : np.array([[0, -1, -2], [1, 1, -1], [2, 1, 0]], dtype=np.float32),
        "bottom_left" : np.array([[0, 1, 2], [-1, 1, 1], [-2, -1, 0]], dtype=np.float32),
        "bottom_right": np.array([[2, 1, 0], [1, 1, -1], [0, -1, -2]], dtype=np.float32),
    }

    def __init__(self) -> None:
        super().__init__(
            name="emboss",
            description="엠보싱 효과",
            params={"direction": "top_left"},
        )

    def process(self, frame: Frame) -> Frame:
        kernel = self.KERNELS.get(
            self.get_param("direction", "top_left"),
            self.KERNELS["top_left"],
        )
        embossed = cv2.filter2D(frame.output, -1, kernel) + 128
        frame.processed = np.clip(embossed, 0, 255).astype(np.uint8)
        return frame


# ── 4. 픽셀화 효과 ───────────────────────────────────────────────────

class PixelateProcessor(BaseProcessor):
    """
    이미지를 픽셀 블록으로 축소 후 확대하여 픽셀화 효과.
    params:
        block_size: 픽셀 블록 크기 (기본 16)
    """

    def __init__(self) -> None:
        super().__init__(
            name="pixelate",
            description="픽셀화 효과",
            params={"block_size": 16},
        )

    def process(self, frame: Frame) -> Frame:
        h, w = frame.output.shape[:2]
        bs = max(1, self.get_param("block_size", 16))
        small = cv2.resize(frame.output, (w // bs, h // bs), interpolation=cv2.INTER_LINEAR)
        frame.processed = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
        return frame


# ── 5. 열화상 (False Color) ──────────────────────────────────────────

class ThermalProcessor(BaseProcessor):
    """
    그레이스케일을 열화상 컬러맵으로 변환합니다.
    params:
        colormap: OpenCV 컬러맵 인덱스 (기본 11 = COLORMAP_JET)
                  옵션: 2(AUTUMN), 3(BONE), 11(JET), 17(INFERNO)
    """

    COLORMAP_NAMES = {
        2: "AUTUMN", 3: "BONE", 4: "COOL", 6: "HOT",
        7: "HSV", 9: "OCEAN", 10: "PINK", 11: "JET",
        17: "INFERNO", 20: "VIRIDIS",
    }

    def __init__(self) -> None:
        super().__init__(
            name="thermal",
            description="열화상 컬러맵 (Jet)",
            params={"colormap": 11},
        )

    def process(self, frame: Frame) -> Frame:
        gray = cv2.cvtColor(frame.output, cv2.COLOR_BGR2GRAY)
        cm = self.get_param("colormap", cv2.COLORMAP_JET)
        frame.processed = cv2.applyColorMap(gray, cm)
        return frame


# ── 6. 오일 페인팅 효과 ──────────────────────────────────────────────

class OilPaintProcessor(BaseProcessor):
    """
    xphoto.oilPainting 기반 유화(油畫) 효과.
    params:
        size      : 브러시 크기 (기본 7)
        dyn_ratio : 동적 범위 비율 (기본 1)
    """

    def __init__(self) -> None:
        super().__init__(
            name="oil_paint",
            description="오일 페인팅 효과",
            params={"size": 7, "dyn_ratio": 1},
        )

    def process(self, frame: Frame) -> Frame:
        try:
            frame.processed = cv2.xphoto.oilPainting(
                frame.output,
                self.get_param("size", 7),
                self.get_param("dyn_ratio", 1),
            )
        except AttributeError:
            # opencv-contrib 없을 경우 fallback
            self._logger.warning("cv2.xphoto not available; using bilateral fallback")
            frame.processed = cv2.bilateralFilter(frame.output, 15, 80, 80)
        return frame
