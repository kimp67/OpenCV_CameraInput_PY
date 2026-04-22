"""
옵티컬 플로우 프로세서 (processors/optical_flow.py)
Lucas-Kanade 희소 광류 및 Farneback 밀집 광류 시각화
"""

import cv2
import numpy as np
from ..core.base_processor import BaseProcessor
from ..core.frame import Frame


# ── 1. Lucas-Kanade 희소 광류 ────────────────────────────────────────

class LKFlowProcessor(BaseProcessor):
    """
    Lucas-Kanade 피라미드 방식 희소 옵티컬 플로우.
    추적점을 자동 갱신하며 이동 궤적을 시각화합니다.
    params:
        max_points  : 추적할 최대 특징점 수 (기본 200)
        win_size    : LK 윈도우 크기 (기본 15)
        max_level   : 피라미드 레벨 (기본 2)
        trail_length: 궤적 길이 (기본 20)
        refresh_interval: 포인트 갱신 간격(프레임 수, 기본 30)
    """

    def __init__(self) -> None:
        super().__init__(
            name="lk_flow",
            description="Lucas-Kanade 희소 옵티컬 플로우",
            params={
                "max_points"      : 200,
                "win_size"        : 15,
                "max_level"       : 2,
                "trail_length"    : 20,
                "refresh_interval": 30,
            },
        )
        self._prev_gray: np.ndarray = None
        self._points: np.ndarray = None
        self._trails: list = []
        self._frame_count: int = 0
        self._mask: np.ndarray = None

    def initialize(self) -> None:
        self._prev_gray = None
        self._points = None
        self._trails = []
        self._frame_count = 0

    def release(self) -> None:
        self.initialize()

    def _detect_points(self, gray: np.ndarray) -> np.ndarray:
        return cv2.goodFeaturesToTrack(
            gray,
            maxCorners=self.get_param("max_points", 200),
            qualityLevel=0.01,
            minDistance=10,
            blockSize=7,
        )

    def process(self, frame: Frame) -> Frame:
        img = frame.output.copy()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        refresh = self.get_param("refresh_interval", 30)

        if self._prev_gray is None or self._frame_count % refresh == 0:
            self._points = self._detect_points(gray)
            self._trails = [[] for _ in range(
                len(self._points) if self._points is not None else 0
            )]
            self._mask = np.zeros_like(img)

        if self._prev_gray is not None and self._points is not None and len(self._points) > 0:
            lk_params = dict(
                winSize=(self.get_param("win_size", 15),) * 2,
                maxLevel=self.get_param("max_level", 2),
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
            )
            next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
                self._prev_gray, gray, self._points, None, **lk_params
            )
            if next_pts is not None and status is not None:
                good_new = next_pts[status == 1]
                good_old = self._points[status == 1]
                trail_len = self.get_param("trail_length", 20)

                new_trails = []
                for i, (new, old) in enumerate(zip(good_new, good_old)):
                    trail = self._trails[i] if i < len(self._trails) else []
                    trail.append(tuple(new.ravel().astype(int)))
                    if len(trail) > trail_len:
                        trail.pop(0)
                    new_trails.append(trail)

                    # 궤적 그리기
                    for j in range(1, len(trail)):
                        alpha = int(255 * j / len(trail))
                        color = (0, alpha, 255 - alpha)
                        cv2.line(self._mask, trail[j - 1], trail[j], color, 2)

                    cv2.circle(img, tuple(new.ravel().astype(int)), 3, (0, 255, 0), -1)

                self._trails = new_trails
                self._points = good_new.reshape(-1, 1, 2)

        combined = cv2.add(img, self._mask)
        self._prev_gray = gray
        self._frame_count += 1
        frame.processed = combined
        return frame


# ── 2. Farneback 밀집 광류 ───────────────────────────────────────────

class DenseFlowProcessor(BaseProcessor):
    """
    Farneback 알고리즘 기반 밀집 옵티컬 플로우.
    HSV 색상 표현으로 전체 화소의 이동 방향/크기를 시각화.
    params:
        pyr_scale   : 피라미드 스케일 (기본 0.5)
        levels      : 피라미드 레벨 수 (기본 3)
        winsize     : 평균화 윈도우 (기본 15)
        iterations  : 반복 횟수 (기본 3)
        poly_n      : 다항식 확장 이웃 크기 (기본 5)
        poly_sigma  : 다항식 가우시안 편차 (기본 1.2)
    """

    def __init__(self) -> None:
        super().__init__(
            name="dense_flow",
            description="Farneback 밀집 옵티컬 플로우",
            params={
                "pyr_scale": 0.5,
                "levels"   : 3,
                "winsize"  : 15,
                "iterations": 3,
                "poly_n"   : 5,
                "poly_sigma": 1.2,
            },
        )
        self._prev_gray: np.ndarray = None

    def initialize(self) -> None:
        self._prev_gray = None

    def release(self) -> None:
        self._prev_gray = None

    def process(self, frame: Frame) -> Frame:
        gray = cv2.cvtColor(frame.output, cv2.COLOR_BGR2GRAY)

        if self._prev_gray is None:
            self._prev_gray = gray
            frame.processed = frame.output.copy()
            return frame

        flow = cv2.calcOpticalFlowFarneback(
            self._prev_gray, gray, None,
            pyr_scale =self.get_param("pyr_scale", 0.5),
            levels    =self.get_param("levels", 3),
            winsize   =self.get_param("winsize", 15),
            iterations=self.get_param("iterations", 3),
            poly_n    =self.get_param("poly_n", 5),
            poly_sigma=self.get_param("poly_sigma", 1.2),
            flags     =0,
        )

        # 크기 및 방향 계산
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

        hsv = np.zeros_like(frame.output)
        hsv[..., 1] = 255
        hsv[..., 0] = (ang * 180 / np.pi / 2).astype(np.uint8)
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        self._prev_gray = gray
        frame.processed = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        return frame
