"""
객체/특징 검출 프로세서 (processors/detection.py)
얼굴 검출, 윤곽선 검출, 직선 검출(허프), 코너 검출(해리스/시토마시)
"""

import cv2
import numpy as np
from ..core.base_processor import BaseProcessor
from ..core.frame import Frame


# ── 1. 얼굴 검출 (Haar Cascade) ─────────────────────────────────────

class FaceDetectorProcessor(BaseProcessor):
    """
    OpenCV Haar Cascade 얼굴 검출기.
    params:
        scale_factor   : 이미지 피라미드 축소 비율 (기본 1.1)
        min_neighbors  : 최소 이웃 사각형 수 (기본 5)
        min_size       : 최소 얼굴 크기 px (기본 30)
        draw_color     : 박스 색상 BGR (기본 [0, 255, 0])
    """

    def __init__(self) -> None:
        super().__init__(
            name="face_detect",
            description="얼굴 검출 (Haar Cascade)",
            params={
                "scale_factor" : 1.1,
                "min_neighbors": 5,
                "min_size"     : 30,
                "draw_color"   : [0, 255, 0],
            },
        )
        self._cascade: cv2.CascadeClassifier = None  # type: ignore

    def initialize(self) -> None:
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self._cascade = cv2.CascadeClassifier(cascade_path)
        if self._cascade.empty():
            self._logger.error("Failed to load Haar cascade: %s", cascade_path)

    def process(self, frame: Frame) -> Frame:
        if self._cascade is None or self._cascade.empty():
            self.initialize()

        img = frame.output.copy()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        faces = self._cascade.detectMultiScale(
            gray,
            scaleFactor=self.get_param("scale_factor", 1.1),
            minNeighbors=self.get_param("min_neighbors", 5),
            minSize=(self.get_param("min_size", 30),) * 2,
        )

        color = tuple(self.get_param("draw_color", [0, 255, 0]))
        for (x, y, w, h) in faces if len(faces) > 0 else []:
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, "Face", (x, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        frame.metadata["face_count"] = len(faces) if len(faces) else 0
        frame.processed = img
        return frame


# ── 2. 윤곽선 검출 ───────────────────────────────────────────────────

class ContourDetectorProcessor(BaseProcessor):
    """
    이진화 후 윤곽선을 검출하고 그립니다.
    params:
        min_area  : 최소 윤곽선 면적 (기본 500)
        draw_color: 선 색상 BGR (기본 [0, 255, 255])
        thickness : 선 두께 (기본 2)
    """

    def __init__(self) -> None:
        super().__init__(
            name="contour_detect",
            description="윤곽선 검출",
            params={"min_area": 500, "draw_color": [0, 255, 255], "thickness": 2},
        )

    def process(self, frame: Frame) -> Frame:
        img = frame.output.copy()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        min_area = self.get_param("min_area", 500)
        color = tuple(self.get_param("draw_color", [0, 255, 255]))
        thickness = self.get_param("thickness", 2)

        filtered = [c for c in contours if cv2.contourArea(c) >= min_area]
        cv2.drawContours(img, filtered, -1, color, thickness)

        frame.metadata["contour_count"] = len(filtered)
        frame.processed = img
        return frame


# ── 3. 허프 직선 검출 ────────────────────────────────────────────────

class HoughLineProcessor(BaseProcessor):
    """
    허프 변환 직선 검출 (확률적 Hough).
    params:
        rho          : 거리 해상도 (기본 1)
        theta_deg    : 각도 해상도(도) (기본 1)
        threshold    : 최소 교차점 수 (기본 50)
        min_line_len : 최소 직선 길이 (기본 50)
        max_line_gap : 최대 직선 간격 (기본 10)
        draw_color   : BGR (기본 [255, 0, 0])
    """

    def __init__(self) -> None:
        super().__init__(
            name="hough_line",
            description="허프 직선 검출",
            params={
                "rho"         : 1,
                "theta_deg"   : 1,
                "threshold"   : 50,
                "min_line_len": 50,
                "max_line_gap": 10,
                "draw_color"  : [255, 0, 0],
            },
        )

    def process(self, frame: Frame) -> Frame:
        img = frame.output.copy()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)

        theta = np.deg2rad(self.get_param("theta_deg", 1))
        lines = cv2.HoughLinesP(
            edges,
            rho=self.get_param("rho", 1),
            theta=theta,
            threshold=self.get_param("threshold", 50),
            minLineLength=self.get_param("min_line_len", 50),
            maxLineGap=self.get_param("max_line_gap", 10),
        )

        color = tuple(self.get_param("draw_color", [255, 0, 0]))
        count = 0
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(img, (x1, y1), (x2, y2), color, 2)
                count += 1

        frame.metadata["line_count"] = count
        frame.processed = img
        return frame


# ── 4. 코너 검출 (해리스 / Shi-Tomasi) ──────────────────────────────

class CornerDetectorProcessor(BaseProcessor):
    """
    Shi-Tomasi 코너 검출기 (goodFeaturesToTrack).
    params:
        max_corners  : 최대 코너 수 (기본 200)
        quality      : 최소 품질 수준 (기본 0.01)
        min_distance : 코너 간 최소 거리 (기본 10)
        draw_color   : BGR (기본 [0, 0, 255])
        radius       : 점 반경 (기본 4)
    """

    def __init__(self) -> None:
        super().__init__(
            name="corner_detect",
            description="코너 검출 (Shi-Tomasi)",
            params={
                "max_corners" : 200,
                "quality"     : 0.01,
                "min_distance": 10,
                "draw_color"  : [0, 0, 255],
                "radius"      : 4,
            },
        )

    def process(self, frame: Frame) -> Frame:
        img = frame.output.copy()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)

        corners = cv2.goodFeaturesToTrack(
            gray,
            maxCorners=self.get_param("max_corners", 200),
            qualityLevel=self.get_param("quality", 0.01),
            minDistance=self.get_param("min_distance", 10),
        )

        color = tuple(self.get_param("draw_color", [0, 0, 255]))
        radius = self.get_param("radius", 4)
        count = 0

        if corners is not None:
            corners = np.int32(corners)
            for pt in corners:
                x, y = pt.ravel()
                cv2.circle(img, (x, y), radius, color, -1)
                count += 1

        frame.metadata["corner_count"] = count
        frame.processed = img
        return frame
