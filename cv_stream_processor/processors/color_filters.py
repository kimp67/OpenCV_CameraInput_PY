"""
색상 필터 프로세서 모음 (processors/color_filters.py)
그레이스케일, HSV 변환, 색상 반전, 채널 분리, 히스토그램 평활화 등
"""

import cv2
import numpy as np
from ..core.base_processor import BaseProcessor
from ..core.frame import Frame


# ── 1. 그레이스케일 ─────────────────────────────────────────────────

class GrayscaleProcessor(BaseProcessor):
    """BGR → 그레이스케일 변환 (3채널 BGR로 재변환하여 출력)."""

    def __init__(self) -> None:
        super().__init__(
            name="grayscale",
            description="그레이스케일 변환",
        )

    def process(self, frame: Frame) -> Frame:
        gray = cv2.cvtColor(frame.output, cv2.COLOR_BGR2GRAY)
        frame.processed = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        return frame


# ── 2. 색상 반전 ─────────────────────────────────────────────────────

class InvertProcessor(BaseProcessor):
    """픽셀 값을 반전(255 - pixel)합니다."""

    def __init__(self) -> None:
        super().__init__(
            name="invert",
            description="색상 반전 (Negative)",
        )

    def process(self, frame: Frame) -> Frame:
        frame.processed = cv2.bitwise_not(frame.output)
        return frame


# ── 3. HSV 채널 시각화 ───────────────────────────────────────────────

class HSVProcessor(BaseProcessor):
    """
    BGR → HSV 변환 후 지정 채널을 시각화합니다.
    params:
        channel: "h" | "s" | "v" | "hsv" (기본 "hsv")
    """

    def __init__(self) -> None:
        super().__init__(
            name="hsv",
            description="HSV 채널 시각화",
            params={"channel": "hsv"},
        )

    def process(self, frame: Frame) -> Frame:
        hsv = cv2.cvtColor(frame.output, cv2.COLOR_BGR2HSV)
        ch = self.get_param("channel", "hsv").lower()

        if ch == "h":
            img = cv2.cvtColor(hsv[:, :, 0], cv2.COLOR_GRAY2BGR)
        elif ch == "s":
            img = cv2.cvtColor(hsv[:, :, 1], cv2.COLOR_GRAY2BGR)
        elif ch == "v":
            img = cv2.cvtColor(hsv[:, :, 2], cv2.COLOR_GRAY2BGR)
        else:
            img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)  # 원본 HSV 표현

        frame.processed = img
        return frame


# ── 4. 히스토그램 평활화 ─────────────────────────────────────────────

class HistEqualProcessor(BaseProcessor):
    """
    명도 채널(V)에 히스토그램 평활화 또는 CLAHE를 적용합니다.
    params:
        method: "global" | "clahe" (기본 "clahe")
        clip_limit: CLAHE clip limit (기본 2.0)
        tile_size: CLAHE tile grid size (기본 8)
    """

    def __init__(self) -> None:
        super().__init__(
            name="hist_equal",
            description="히스토그램 평활화 (CLAHE)",
            params={"method": "clahe", "clip_limit": 2.0, "tile_size": 8},
        )
        self._clahe: cv2.CLAHE = None  # type: ignore

    def initialize(self) -> None:
        self._build_clahe()

    def _build_clahe(self) -> None:
        self._clahe = cv2.createCLAHE(
            clipLimit=self.get_param("clip_limit", 2.0),
            tileGridSize=(self.get_param("tile_size", 8),) * 2,
        )

    def process(self, frame: Frame) -> Frame:
        img = frame.output
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        method = self.get_param("method", "clahe")
        if method == "clahe":
            if self._clahe is None:
                self._build_clahe()
            v = self._clahe.apply(v)
        else:
            v = cv2.equalizeHist(v)

        frame.processed = cv2.cvtColor(cv2.merge([h, s, v]), cv2.COLOR_HSV2BGR)
        return frame


# ── 5. 색상 공간 채널 분리 시각화 ───────────────────────────────────

class ChannelSplitProcessor(BaseProcessor):
    """
    BGR 채널을 분리하여 나란히 표시합니다.
    params:
        layout: "horizontal" | "vertical" (기본 "horizontal")
    """

    def __init__(self) -> None:
        super().__init__(
            name="channel_split",
            description="BGR 채널 분리 시각화",
            params={"layout": "horizontal"},
        )

    def process(self, frame: Frame) -> Frame:
        img = frame.output
        b, g, r = cv2.split(img)
        zeros = np.zeros_like(b)

        b_img = cv2.merge([b, zeros, zeros])
        g_img = cv2.merge([zeros, g, zeros])
        r_img = cv2.merge([zeros, zeros, r])

        layout = self.get_param("layout", "horizontal")
        if layout == "vertical":
            combined = np.vstack([b_img, g_img, r_img])
            # 원본 높이로 리사이즈
            h, w = img.shape[:2]
            combined = cv2.resize(combined, (w, h))
        else:
            combined = np.hstack([b_img, g_img, r_img])
            # 원본 너비로 리사이즈
            h, w = img.shape[:2]
            combined = cv2.resize(combined, (w, h))

        frame.processed = combined
        return frame


# ── 6. 색온도 조정 (화이트 밸런스) ──────────────────────────────────

class WhiteBalanceProcessor(BaseProcessor):
    """
    간단한 그레이월드 화이트 밸런스 알고리즘을 적용합니다.
    """

    def __init__(self) -> None:
        super().__init__(
            name="white_balance",
            description="그레이월드 화이트 밸런스",
        )

    def process(self, frame: Frame) -> Frame:
        img = frame.output.astype(np.float32)
        b, g, r = cv2.split(img)

        b_mean, g_mean, r_mean = b.mean(), g.mean(), r.mean()
        total_mean = (b_mean + g_mean + r_mean) / 3.0

        b = np.clip(b * (total_mean / (b_mean + 1e-6)), 0, 255)
        g = np.clip(g * (total_mean / (g_mean + 1e-6)), 0, 255)
        r = np.clip(r * (total_mean / (r_mean + 1e-6)), 0, 255)

        frame.processed = cv2.merge([b, g, r]).astype(np.uint8)
        return frame
