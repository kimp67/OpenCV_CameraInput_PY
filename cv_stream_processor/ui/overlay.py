"""
오버레이 렌더러 (ui/overlay.py)
프레임 위에 FPS, 파이프라인 정보, 도움말 등을 렌더링
"""

from __future__ import annotations

import cv2
import numpy as np
from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from ..core.pipeline_registry import PipelineRegistry
    from ..utils.fps_counter import FPSCounter
    from ..config import DisplayConfig


# ── 색상 팔레트 ─────────────────────────────────────────────────────
COLOR_BG     = (20, 20, 20)
COLOR_ACTIVE = (0, 220, 100)
COLOR_TEXT   = (220, 220, 220)
COLOR_HEADER = (100, 200, 255)
COLOR_WARN   = (0, 165, 255)
COLOR_DIM    = (120, 120, 120)


def _draw_rounded_rect(
    img: np.ndarray,
    pt1: tuple, pt2: tuple,
    color: tuple, alpha: float = 0.6, radius: int = 8
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
) -> np.ndarray:
    """
    프레임 위에 정보 패널과 도움말을 렌더링합니다.

    Args:
        frame      : 출력 이미지 (BGR ndarray)
        registry   : PipelineRegistry 인스턴스
        fps_counter: FPSCounter 인스턴스
        config     : DisplayConfig 인스턴스
        recording  : 현재 녹화 중인지 여부
        paused     : 일시정지 여부
        show_help  : 도움말 표시 여부

    Returns:
        오버레이가 합성된 이미지
    """
    img = frame.copy()
    h, w = img.shape[:2]
    fs = config.font_scale
    ft = config.font_thickness
    font = cv2.FONT_HERSHEY_SIMPLEX
    panel_w = config.info_panel_width

    # ── 좌측 정보 패널 ──────────────────────────────────────────────
    if config.show_pipeline_info:
        pad = 8
        line_h = int(22 * fs / 0.6)
        names = registry.pipeline_names
        panel_h = (len(names) + 4) * line_h + pad * 3

        _draw_rounded_rect(
            img,
            (pad, pad),
            (panel_w, min(panel_h, h - pad)),
            COLOR_BG, alpha=0.70,
        )

        y = pad + line_h
        # 헤더
        cv2.putText(img, "[ PIPELINES ]", (pad + 6, y),
                    font, fs * 0.85, COLOR_HEADER, ft)
        y += line_h

        for name in names:
            desc = registry.get_description(name)
            is_active = (name == registry.active_name)
            color = COLOR_ACTIVE if is_active else COLOR_TEXT
            prefix = ">> " if is_active else "   "
            label = f"{prefix}{desc}"

            # 너비 초과 시 자르기
            max_chars = (panel_w - pad * 2 - 10) // max(1, int(10 * fs))
            if len(label) > max_chars:
                label = label[:max_chars - 1] + "~"

            cv2.putText(img, label, (pad + 6, y), font, fs * 0.75, color, ft)
            y += line_h
            if y > h - pad:
                break

    # ── FPS / 통계 ───────────────────────────────────────────────────
    if config.show_fps:
        fps_txt = f"FPS: {fps_counter.fps:5.1f}"
        frm_txt = f"Frames: {fps_counter.total_frames}"
        ela_txt = f"Time: {fps_counter.elapsed_seconds:6.1f}s"

        x_r = w - 160
        y_t = pad + line_h
        _draw_rounded_rect(img, (x_r - 8, pad), (w - pad, y_t + line_h * 2 + 8),
                            COLOR_BG, alpha=0.65)
        cv2.putText(img, fps_txt, (x_r, y_t),          font, fs, COLOR_ACTIVE, ft)
        cv2.putText(img, frm_txt, (x_r, y_t + line_h), font, fs * 0.75, COLOR_TEXT, ft)
        cv2.putText(img, ela_txt, (x_r, y_t + line_h * 2), font, fs * 0.75, COLOR_DIM, ft)

    # ── 활성 파이프라인 배지 ─────────────────────────────────────────
    badge = f"[{registry.active_name.upper()}]"
    (bw, bh), _ = cv2.getTextSize(badge, font, fs * 1.1, ft + 1)
    bx = (w - bw) // 2
    by = h - pad - bh - 4
    _draw_rounded_rect(img, (bx - 8, by - bh - 4), (bx + bw + 8, by + 6),
                        COLOR_BG, alpha=0.70)
    cv2.putText(img, badge, (bx, by), font, fs * 1.1, COLOR_ACTIVE, ft + 1)

    # ── 녹화 표시 ────────────────────────────────────────────────────
    if recording:
        cv2.circle(img, (w - 28, 28), 10, (0, 0, 220), -1)
        cv2.putText(img, "REC", (w - 68, 34), font, fs * 0.8, (0, 0, 220), ft)

    # ── 일시정지 표시 ────────────────────────────────────────────────
    if paused:
        _draw_rounded_rect(img, (w // 2 - 60, h // 2 - 24),
                            (w // 2 + 60, h // 2 + 24), (0, 0, 0), alpha=0.7)
        cv2.putText(img, "PAUSED", (w // 2 - 50, h // 2 + 8),
                    font, fs * 1.2, COLOR_WARN, ft + 1)

    # ── 도움말 패널 ──────────────────────────────────────────────────
    if show_help and config.show_help:
        _draw_help(img, fs, ft, font, h, w)

    return img


def _draw_help(img, fs, ft, font, h, w) -> None:
    """하단 도움말 패널을 렌더링합니다."""
    help_lines = [
        "Q/ESC: 종료   SPACE: 일시정지   S: 스크린샷",
        "N/P: 다음/이전 파이프라인   R: 녹화 시작/중지",
        "H: 도움말 토글   F: 전체화면   1-9: 파이프라인 선택",
    ]
    pad = 6
    line_h = int(18 * fs / 0.6)
    panel_h = len(help_lines) * line_h + pad * 2
    y0 = h - panel_h - pad

    overlay = img.copy()
    cv2.rectangle(overlay, (0, y0), (w, h), COLOR_BG, -1)
    cv2.addWeighted(overlay, 0.60, img, 0.40, 0, img)

    for i, line in enumerate(help_lines):
        y = y0 + pad + line_h * (i + 1)
        cv2.putText(img, line, (pad + 4, y), font, fs * 0.72, COLOR_DIM, ft)
