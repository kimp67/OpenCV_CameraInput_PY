"""
오버레이 렌더러 (ui/overlay.py)
프레임 위에 FPS, 파이프라인 정보, 도움말 등을 렌더링
"""

from __future__ import annotations

import cv2
import numpy as np
from typing import TYPE_CHECKING, List

from .pil_text import BgrPilTextLayer, text_size, truncate_to_pixel_width

if TYPE_CHECKING:
    from ..core.pipeline_registry import PipelineRegistry
    from ..utils.fps_counter import FPSCounter
    from ..config import DisplayConfig


# ── 색상 팔레트 ─────────────────────────────────────────────────────
COLOR_BG = (20, 20, 20)
COLOR_ACTIVE = (0, 220, 100)
COLOR_TEXT = (220, 220, 220)
COLOR_HEADER = (100, 200, 255)
COLOR_WARN = (0, 165, 255)
COLOR_DIM = (120, 120, 120)


def _draw_rounded_rect(
    img: np.ndarray,
    pt1: tuple,
    pt2: tuple,
    color: tuple,
    alpha: float = 0.6,
    radius: int = 8,
) -> None:
    """알파 블렌딩된 반투명 둥근 사각형을 그립니다."""
    overlay = img.copy()
    cv2.rectangle(overlay, pt1, pt2, color, -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)


def draw_overlay(
    frame: np.ndarray,
    registry: "PipelineRegistry",
    fps_counter: "FPSCounter",
    config: "DisplayConfig",
    recording: bool = False,
    paused: bool = False,
    show_help: bool = True,
    frame_width: int = 0,
    frame_height: int = 0,
) -> np.ndarray:
    """
    프레임 위에 정보 패널과 도움말을 렌더링합니다.

    Args:
        frame        : 출력 이미지 (BGR ndarray)
        registry     : PipelineRegistry 인스턴스
        fps_counter  : FPSCounter 인스턴스
        config       : DisplayConfig 인스턴스
        recording    : 현재 녹화 중인지 여부
        paused       : 일시정지 여부
        show_help    : 도움말 표시 여부
        frame_width  : 원본 프레임 너비 (해상도 표시용, 0이면 img 크기 사용)
        frame_height : 원본 프레임 높이 (해상도 표시용, 0이면 img 크기 사용)

    Returns:
        오버레이가 합성된 이미지
    """
    img = frame.copy()
    h, w = img.shape[:2]
    fs = config.font_scale
    ft = config.font_thickness
    panel_w = config.info_panel_width
    pad = 8
    line_h = int(22 * fs / 0.6)

    # 해상도: 인수로 받은 값 우선, 없으면 현재 프레임 크기 사용
    res_w = frame_width  if frame_width  > 0 else w
    res_h = frame_height if frame_height > 0 else h

    # ── 좌측 정보 패널 배경 ─────────────────────────────────────────
    names: List[str] = []
    if config.show_pipeline_info:
        names = registry.pipeline_names
        panel_h = (len(names) + 4) * line_h + pad * 3
        _draw_rounded_rect(
            img,
            (pad, pad),
            (panel_w, min(panel_h, h - pad)),
            COLOR_BG,
            alpha=0.70,
        )

    # ── FPS / 해상도 / 통계 패널 배경 (4행으로 확장) ─────────────────
    # 행 구성: [해상도]  [FPS]  [Frames]  [Time]
    x_r = w - 185
    y_t = pad + line_h
    if config.show_fps:
        _draw_rounded_rect(
            img,
            (x_r - 8, pad),
            (w - pad, y_t + line_h * 3 + 8),   # 기존 2행 → 3행(+해상도)
            COLOR_BG,
            alpha=0.65,
        )

    # ── 활성 파이프라인 배지 배경 ───────────────────────────────────
    badge = f"[{registry.active_name.upper()}]"
    bw, bh = text_size(badge, fs * 1.1)
    bx = (w - bw) // 2
    by = h - pad - bh - 4
    _draw_rounded_rect(
        img,
        (bx - 8, by - bh - 4),
        (bx + bw + 8, by + 6),
        COLOR_BG,
        alpha=0.70,
    )

    # ── 녹화 표시 (도형만) ───────────────────────────────────────────
    if recording:
        cv2.circle(img, (w - 28, 28), 10, (0, 0, 220), -1)

    # ── 일시정지 배경 ────────────────────────────────────────────────
    if paused:
        _draw_rounded_rect(
            img,
            (w // 2 - 60, h // 2 - 24),
            (w // 2 + 60, h // 2 + 24),
            (0, 0, 0),
            alpha=0.7,
        )

    # ── 도움말 패널 배경 ─────────────────────────────────────────────
    help_lines: List[str] = []
    help_y0 = 0
    help_h_pad = 6
    help_line_h = int(18 * fs / 0.6)
    if show_help and config.show_help:
        help_lines = [
            "Q/ESC: 종료   SPACE: 일시정지   S: 스크린샷",
            "N/P: 다음/이전 파이프라인   R: 녹화 시작/중지",
            "H: 도움말 토글   F: 전체화면   1-9: 파이프라인 선택",
        ]
        help_panel_h = len(help_lines) * help_line_h + help_h_pad * 2
        help_y0 = h - help_panel_h - help_h_pad
        overlay = img.copy()
        cv2.rectangle(overlay, (0, help_y0), (w, h), COLOR_BG, -1)
        cv2.addWeighted(overlay, 0.60, img, 0.40, 0, img)

    # ── Pillow 텍스트 (한글 포함) 한 번에 합성 ─────────────────────
    txt = BgrPilTextLayer(img)

    if config.show_pipeline_info:
        y = pad + line_h
        txt.put("[ PIPELINES ]", (pad + 6, y), fs * 0.85, COLOR_HEADER, ft)
        y += line_h
        max_px = panel_w - pad * 2 - 10
        for name in names:
            desc = registry.get_description(name)
            is_active = name == registry.active_name
            color = COLOR_ACTIVE if is_active else COLOR_TEXT
            prefix = ">> " if is_active else "   "
            label = f"{prefix}{desc}"
            label = truncate_to_pixel_width(label, fs * 0.75, max_px)
            txt.put(label, (pad + 6, y), fs * 0.75, color, ft)
            y += line_h
            if y > h - pad:
                break

    if config.show_fps:
        res_txt = f"{res_w} x {res_h}"
        fps_txt = f"FPS: {fps_counter.fps:5.1f}"
        frm_txt = f"Frames: {fps_counter.total_frames}"
        ela_txt = f"Time: {fps_counter.elapsed_seconds:6.1f}s"
        txt.put(res_txt, (x_r, y_t),              fs,        COLOR_HEADER, ft)  # 해상도 (강조)
        txt.put(fps_txt, (x_r, y_t + line_h),     fs,        COLOR_ACTIVE, ft)  # FPS
        txt.put(frm_txt, (x_r, y_t + line_h * 2), fs * 0.75, COLOR_TEXT,  ft)  # Frames
        txt.put(ela_txt, (x_r, y_t + line_h * 3), fs * 0.75, COLOR_DIM,   ft)  # Time

    txt.put(badge, (bx, by), fs * 1.1, COLOR_ACTIVE, ft + 1)

    if recording:
        txt.put("REC", (w - 68, 34), fs * 0.8, (0, 0, 220), ft)

    if paused:
        txt.put("PAUSED", (w // 2 - 50, h // 2 + 8), fs * 1.2, COLOR_WARN, ft + 1)

    if show_help and config.show_help:
        for i, line in enumerate(help_lines):
            y = help_y0 + help_h_pad + help_line_h * (i + 1)
            txt.put(line, (help_h_pad + 4, y), fs * 0.72, COLOR_DIM, ft)

    txt.commit()
    return img
