"""
한글(유니코드) 텍스트 렌더러 (utils/korean_text.py)

cv2.putText()는 ASCII만 지원하므로 한글이 깨집니다.
PIL(Pillow)을 이용해 한글을 포함한 유니코드 텍스트를 OpenCV 이미지에 직접 그립니다.

사용 예:
    from cv_stream_processor.utils.korean_text import put_text_kr

    put_text_kr(img, "안녕하세요", (10, 50), font_size=20, color=(220, 220, 220))
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Tuple, Optional

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


# ── 한글 폰트 우선순위 탐색 ─────────────────────────────────────────
_FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
    "/usr/share/fonts/truetype/nanum/NanumBarunGothic.ttf",
    "/usr/share/fonts/truetype/nanum/NanumGothicLight.ttf",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSerifCJK-Bold.ttc",
    "/usr/share/fonts/truetype/nanum/NanumSquareRoundR.ttf",
]


@lru_cache(maxsize=32)
def _load_font(font_size: int) -> ImageFont.FreeTypeFont:
    """지정한 크기의 한글 폰트를 로드합니다 (LRU 캐시)."""
    for path in _FONT_CANDIDATES:
        if os.path.isfile(path):
            return ImageFont.truetype(path, font_size)
    # 폴백: PIL 기본 폰트 (한글 미지원이지만 크래시 방지)
    return ImageFont.load_default()


def put_text_kr(
    img: np.ndarray,
    text: str,
    org: Tuple[int, int],
    font_size: int = 18,
    color: Tuple[int, int, int] = (220, 220, 220),
    alpha: float = 1.0,
) -> np.ndarray:
    """
    OpenCV 이미지(BGR ndarray)에 한글을 포함한 유니코드 텍스트를 그립니다.

    Args:
        img      : BGR ndarray (in-place 수정됨)
        text     : 출력할 텍스트 (한글 포함 가능)
        org      : 텍스트 좌측 하단 좌표 (x, y)  ← cv2.putText 와 동일 기준
        font_size: 폰트 크기 (픽셀)
        color    : BGR 색상 튜플
        alpha    : 투명도 (1.0 = 불투명)

    Returns:
        수정된 img (동일 객체)
    """
    font = _load_font(font_size)

    # PIL은 RGB 순서이므로 BGR → RGB 변환
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(pil_img)

    # org는 cv2 기준 (좌측 하단), PIL은 좌측 상단 기준으로 그리므로 y 조정
    x, y = org
    # getbbox로 텍스트 높이 계산
    bbox = font.getbbox(text)          # (left, top, right, bottom)
    text_h = bbox[3] - bbox[1]
    pil_y = y - text_h - bbox[1]      # cv2 baseline → PIL top-left 변환

    # PIL color는 RGB
    r, g, b = color[2], color[1], color[0]  # BGR → RGB

    if alpha < 1.0:
        # 투명도 지원: 별도 레이어에 그린 뒤 합성
        text_layer = Image.new("RGBA", pil_img.size, (0, 0, 0, 0))
        layer_draw = ImageDraw.Draw(text_layer)
        layer_draw.text((x, pil_y), text, font=font, fill=(r, g, b, int(255 * alpha)))
        pil_img = pil_img.convert("RGBA")
        pil_img = Image.alpha_composite(pil_img, text_layer)
        pil_img = pil_img.convert("RGB")
    else:
        draw.text((x, pil_y), text, font=font, fill=(r, g, b))

    # RGB → BGR 변환 후 원본 img에 반영
    result = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    np.copyto(img, result)
    return img


def get_text_size_kr(
    text: str,
    font_size: int = 18,
) -> Tuple[int, int]:
    """
    한글 텍스트의 (width, height)를 반환합니다.
    cv2.getTextSize()의 대체 함수입니다.

    Returns:
        (width, height) 픽셀 크기
    """
    font = _load_font(font_size)
    bbox = font.getbbox(text)          # (left, top, right, bottom)
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    return w, h
