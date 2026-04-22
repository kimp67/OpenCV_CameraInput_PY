"""
블러 & 샤프닝 프로세서 (processors/blur_sharpen.py)
가우시안, 미디안, 모션 블러, 언샤프 마스킹, 디테일 향상 등
"""

import cv2
import numpy as np
from ..core.base_processor import BaseProcessor
from ..core.frame import Frame


# ── 1. 가우시안 블러 ─────────────────────────────────────────────────

class GaussianBlurProcessor(BaseProcessor):
    """
    가우시안 블러.
    params:
        ksize : 커널 크기 (홀수, 기본 15)
        sigma : 표준편차 (0이면 ksize에서 자동 계산)
    """

    def __init__(self) -> None:
        super().__init__(
            name="gaussian_blur",
            description="가우시안 블러",
            params={"ksize": 15, "sigma": 0},
        )

    def process(self, frame: Frame) -> Frame:
        k = self.get_param("ksize", 15)
        k = k if k % 2 == 1 else k + 1  # 홀수 보장
        frame.processed = cv2.GaussianBlur(
            frame.output, (k, k), self.get_param("sigma", 0)
        )
        return frame


# ── 2. 미디안 블러 ───────────────────────────────────────────────────

class MedianBlurProcessor(BaseProcessor):
    """
    미디안 블러 – 소금&후추 노이즈 제거에 효과적.
    params:
        ksize: 커널 크기 (홀수, 기본 9)
    """

    def __init__(self) -> None:
        super().__init__(
            name="median_blur",
            description="미디안 블러 (노이즈 제거)",
            params={"ksize": 9},
        )

    def process(self, frame: Frame) -> Frame:
        k = self.get_param("ksize", 9)
        k = k if k % 2 == 1 else k + 1
        frame.processed = cv2.medianBlur(frame.output, k)
        return frame


# ── 3. 바이래터럴 필터 ───────────────────────────────────────────────

class BilateralProcessor(BaseProcessor):
    """
    바이래터럴 필터 – 엣지를 보존하며 평활화.
    params:
        d         : 픽셀 이웃 지름 (기본 9)
        sigma_color: 색 공간 표준편차 (기본 75)
        sigma_space: 좌표 공간 표준편차 (기본 75)
    """

    def __init__(self) -> None:
        super().__init__(
            name="bilateral",
            description="바이래터럴 필터 (엣지 보존 평활화)",
            params={"d": 9, "sigma_color": 75, "sigma_space": 75},
        )

    def process(self, frame: Frame) -> Frame:
        frame.processed = cv2.bilateralFilter(
            frame.output,
            self.get_param("d", 9),
            self.get_param("sigma_color", 75),
            self.get_param("sigma_space", 75),
        )
        return frame


# ── 4. 모션 블러 ─────────────────────────────────────────────────────

class MotionBlurProcessor(BaseProcessor):
    """
    수평/수직 모션 블러.
    params:
        ksize    : 커널 크기 (기본 21)
        direction: "horizontal" | "vertical" | "diagonal" (기본 "horizontal")
    """

    def __init__(self) -> None:
        super().__init__(
            name="motion_blur",
            description="모션 블러",
            params={"ksize": 21, "direction": "horizontal"},
        )

    def process(self, frame: Frame) -> Frame:
        k = self.get_param("ksize", 21)
        direction = self.get_param("direction", "horizontal")

        kernel = np.zeros((k, k), dtype=np.float32)
        if direction == "horizontal":
            kernel[k // 2, :] = 1.0 / k
        elif direction == "vertical":
            kernel[:, k // 2] = 1.0 / k
        else:
            np.fill_diagonal(kernel, 1.0 / k)

        frame.processed = cv2.filter2D(frame.output, -1, kernel)
        return frame


# ── 5. 언샤프 마스킹 (선명도 향상) ──────────────────────────────────

class UnsharpMaskProcessor(BaseProcessor):
    """
    언샤프 마스킹으로 이미지 선명도를 높입니다.
    params:
        strength: 강도 (0.5 ~ 3.0, 기본 1.5)
        radius  : 블러 반경 (기본 5)
    """

    def __init__(self) -> None:
        super().__init__(
            name="unsharp_mask",
            description="언샤프 마스킹 (선명도 향상)",
            params={"strength": 1.5, "radius": 5},
        )

    def process(self, frame: Frame) -> Frame:
        r = self.get_param("radius", 5)
        r = r if r % 2 == 1 else r + 1
        s = self.get_param("strength", 1.5)

        blurred = cv2.GaussianBlur(frame.output, (r, r), 0)
        sharpened = cv2.addWeighted(frame.output, 1.0 + s, blurred, -s, 0)
        frame.processed = np.clip(sharpened, 0, 255).astype(np.uint8)
        return frame


# ── 6. 디테일 향상 (detail enhance) ─────────────────────────────────

class DetailEnhanceProcessor(BaseProcessor):
    """
    cv2.detailEnhance – 비포토리얼 렌더링 기반 디테일 강화.
    params:
        sigma_s: 공간 범위 (기본 10)
        sigma_r: 색 범위 (기본 0.15)
    """

    def __init__(self) -> None:
        super().__init__(
            name="detail_enhance",
            description="디테일 향상 (NPR)",
            params={"sigma_s": 10, "sigma_r": 0.15},
        )

    def process(self, frame: Frame) -> Frame:
        frame.processed = cv2.detailEnhance(
            frame.output,
            sigma_s=self.get_param("sigma_s", 10),
            sigma_r=self.get_param("sigma_r", 0.15),
        )
        return frame
