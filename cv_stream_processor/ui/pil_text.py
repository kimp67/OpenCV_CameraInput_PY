"""
Pillow 기반 텍스트 렌더링 — OpenCV putText는 한글을 지원하지 않아 BGR 프레임에 합성합니다.
"""

from __future__ import annotations

import os
import sys
from functools import lru_cache
from typing import List, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def _candidate_font_paths() -> List[str]:
    paths: List[str] = []
    if sys.platform == "win32":
        windir = os.environ.get("WINDIR", r"C:\Windows")
        paths.extend(
            [
                os.path.join(windir, "Fonts", "malgun.ttf"),
                os.path.join(windir, "Fonts", "malgunbd.ttf"),
                os.path.join(windir, "Fonts", "gulim.ttc"),
                os.path.join(windir, "Fonts", "batang.ttc"),
            ]
        )
    elif sys.platform == "darwin":
        paths.extend(
            [
                "/System/Library/Fonts/Supplemental/AppleGothic.ttf",
                "/Library/Fonts/AppleGothic.ttf",
                "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
            ]
        )
    else:
        paths.extend(
            [
                "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
                "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.otf",
                "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            ]
        )
    return paths


def _truetype(path: str, size: int) -> ImageFont.FreeTypeFont:
    if path.lower().endswith(".ttc"):
        return ImageFont.truetype(path, size=size, index=0)
    return ImageFont.truetype(path, size=size)


@lru_cache(maxsize=32)
def _load_font(size: int) -> ImageFont.FreeTypeFont:
    for path in _candidate_font_paths():
        if path and os.path.isfile(path):
            try:
                return _truetype(path, size)
            except OSError:
                continue
    return ImageFont.load_default()


def cv_scale_to_px(cv_font_scale: float) -> int:
    """DisplayConfig.font_scale(기본 0.6) 기준으로 Hershey Simplex와 비슷한 크기의 픽셀 크기."""
    return max(11, int(20 * cv_font_scale / 0.6 + 0.5))


def text_size(text: str, cv_font_scale: float) -> Tuple[int, int]:
    """(너비, 높이) — 배치용. OpenCV getTextSize 대체."""
    font = _load_font(cv_scale_to_px(cv_font_scale))
    bbox = font.getbbox(text)
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    return w, h


def truncate_to_pixel_width(
    text: str, cv_font_scale: float, max_width: int, ellipsis: str = "…"
) -> str:
    """표시 너비가 max_width를 넘지 않도록 잘라냅니다."""
    if max_width <= 0:
        return ""
    font = _load_font(cv_scale_to_px(cv_font_scale))
    if _pixel_width(font, text) <= max_width:
        return text
    ell = ellipsis if _pixel_width(font, ellipsis) <= max_width else ""
    lo, hi = 0, len(text)
    best = ell
    while lo <= hi:
        mid = (lo + hi) // 2
        cand = text[:mid] + ell
        if _pixel_width(font, cand) <= max_width:
            best = cand
            lo = mid + 1
        else:
            hi = mid - 1
    return best


def _pixel_width(font: ImageFont.FreeTypeFont, text: str) -> int:
    bbox = font.getbbox(text)
    return bbox[2] - bbox[0]


class BgrPilTextLayer:
    """
    BGR 이미지 위에 Pillow로만 문자를 올립니다.
    배경(cv2 도형)을 모두 그린 뒤 인스턴스를 만들고 put → commit 한 번으로 합성합니다.
    """

    __slots__ = ("_bgr", "_pil", "_draw")

    def __init__(self, bgr: np.ndarray) -> None:
        self._bgr = bgr
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        self._pil = Image.fromarray(rgb)
        self._draw = ImageDraw.Draw(self._pil)

    def put(
        self,
        text: str,
        org: Tuple[int, int],
        cv_font_scale: float,
        color_bgr: Tuple[int, int, int],
        thickness: int = 1,
    ) -> None:
        """org = (x, y) 좌하단 기준선 (OpenCV putText와 동일)."""
        if not text:
            return
        px = cv_scale_to_px(cv_font_scale)
        font = _load_font(px)
        rgb = (int(color_bgr[2]), int(color_bgr[1]), int(color_bgr[0]))
        stroke_w = max(0, int(thickness) - 1)
        kw: dict = {"font": font, "fill": rgb, "anchor": "ls"}
        if stroke_w > 0:
            kw["stroke_width"] = stroke_w
            kw["stroke_fill"] = (0, 0, 0)
        self._draw.text(org, text, **kw)

    def commit(self) -> None:
        out = cv2.cvtColor(np.asarray(self._pil), cv2.COLOR_RGB2BGR)
        np.copyto(self._bgr, out)
