"""
엣지 검출 프로세서 모음 (processors/edge_detection.py)
Canny, Sobel, Laplacian, Scharr 엣지 검출 알고리즘
"""

import cv2
import numpy as np
from ..core.base_processor import BaseProcessor
from ..core.frame import Frame


# ── 1. Canny 엣지 검출 ──────────────────────────────────────────────

class CannyProcessor(BaseProcessor):
    """
    Canny 엣지 검출기.
    params:
        threshold1 : 하이퍼레시스 낮은 임계값 (기본 50)
        threshold2 : 하이퍼레시스 높은 임계값 (기본 150)
        aperture   : Sobel 커널 크기 (기본 3, 홀수만)
        color      : True이면 BGR 컬러 엣지 출력 (기본 False)
    """

    def __init__(self) -> None:
        super().__init__(
            name="canny",
            description="Canny 엣지 검출",
            params={"threshold1": 50, "threshold2": 150, "aperture": 3, "color": False},
        )

    def process(self, frame: Frame) -> Frame:
        img = frame.output
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        edges = cv2.Canny(
            blurred,
            self.get_param("threshold1", 50),
            self.get_param("threshold2", 150),
            apertureSize=self.get_param("aperture", 3),
        )

        if self.get_param("color", False):
            # 컬러 엣지: 원본과 마스크 합성
            mask = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
            frame.processed = cv2.bitwise_and(img, mask)
        else:
            frame.processed = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        return frame


# ── 2. Sobel 엣지 검출 ──────────────────────────────────────────────

class SobelProcessor(BaseProcessor):
    """
    Sobel 미분 연산자를 사용한 엣지 검출.
    params:
        ksize    : 커널 크기 (1, 3, 5, 7, 기본 3)
        direction: "x" | "y" | "xy" (기본 "xy")
    """

    def __init__(self) -> None:
        super().__init__(
            name="sobel",
            description="Sobel 엣지 검출",
            params={"ksize": 3, "direction": "xy"},
        )

    def process(self, frame: Frame) -> Frame:
        gray = cv2.cvtColor(frame.output, cv2.COLOR_BGR2GRAY)
        ksize = self.get_param("ksize", 3)
        direction = self.get_param("direction", "xy")

        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize)

        if direction == "x":
            result = np.abs(sobelx)
        elif direction == "y":
            result = np.abs(sobely)
        else:
            result = np.sqrt(sobelx**2 + sobely**2)

        result = np.clip(result / result.max() * 255, 0, 255).astype(np.uint8)
        frame.processed = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
        return frame


# ── 3. Laplacian 엣지 검출 ──────────────────────────────────────────

class LaplacianProcessor(BaseProcessor):
    """
    라플라시안(2차 미분) 기반 엣지 검출.
    params:
        ksize   : 커널 크기 (홀수, 기본 3)
        blur_ksize: 전처리 블러 커널 크기 (기본 5)
    """

    def __init__(self) -> None:
        super().__init__(
            name="laplacian",
            description="Laplacian 엣지 검출",
            params={"ksize": 3, "blur_ksize": 5},
        )

    def process(self, frame: Frame) -> Frame:
        gray = cv2.cvtColor(frame.output, cv2.COLOR_BGR2GRAY)
        bk = self.get_param("blur_ksize", 5)
        blurred = cv2.GaussianBlur(gray, (bk, bk), 0)

        lap = cv2.Laplacian(blurred, cv2.CV_64F, ksize=self.get_param("ksize", 3))
        lap = np.abs(lap)
        lap = np.clip(lap / (lap.max() + 1e-6) * 255, 0, 255).astype(np.uint8)

        frame.processed = cv2.cvtColor(lap, cv2.COLOR_GRAY2BGR)
        return frame


# ── 4. Scharr 엣지 검출 ─────────────────────────────────────────────

class ScharrProcessor(BaseProcessor):
    """
    Scharr 연산자 – 작은 커널에서 Sobel보다 정확한 엣지 검출.
    params:
        direction: "x" | "y" | "xy" (기본 "xy")
    """

    def __init__(self) -> None:
        super().__init__(
            name="scharr",
            description="Scharr 엣지 검출",
            params={"direction": "xy"},
        )

    def process(self, frame: Frame) -> Frame:
        gray = cv2.cvtColor(frame.output, cv2.COLOR_BGR2GRAY)
        direction = self.get_param("direction", "xy")

        sx = cv2.Scharr(gray, cv2.CV_64F, 1, 0)
        sy = cv2.Scharr(gray, cv2.CV_64F, 0, 1)

        if direction == "x":
            result = np.abs(sx)
        elif direction == "y":
            result = np.abs(sy)
        else:
            result = np.sqrt(sx**2 + sy**2)

        result = np.clip(result / (result.max() + 1e-6) * 255, 0, 255).astype(np.uint8)
        frame.processed = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
        return frame
