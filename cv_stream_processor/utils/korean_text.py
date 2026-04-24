"""
한글(유니코드) 텍스트 렌더러 (utils/korean_text.py)

cv2.putText()는 ASCII 문자만 지원하여 한글이 깨집니다.
PIL(Pillow)을 이용해 한글을 포함한 유니코드 텍스트를
OpenCV BGR ndarray 이미지에 직접 렌더링합니다.

공개 API:
    put_text_kr(img, text, org, font_size, color, alpha)
    get_text_size_kr(text, font_size) -> (width, height)
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ── 시스템 한글 폰트 우선순위 ────────────────────────────────────────
_FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
    "/usr/share/fonts/truetype/nanum/NanumBarunGothic.ttf",
    "/usr/share/fonts/truetype/nanum/NanumGothicLight.ttf",
    "/usr/share/fonts/truetype/nanum/NanumSquareR.ttf",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSerifCJK-Bold.ttc",
]


@lru_cache(maxsize=32)
def _load_font(font_size: int) -> ImageFont.FreeTypeFont:
    """지정 크기의 한글 트루타입 폰트를 로드합니다 (LRU 캐시 적용)."""
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
    OpenCV BGR ndarray에 한글을 포함한 유니코드 텍스트를 그립니다.

    cv2.putText()와 동일한 인터페이스를 유지합니다.
      - org : 텍스트 좌측 하단(baseline) 좌표 (x, y)
      - color : BGR 튜플

    Args:
        img       : BGR ndarray (in-place 수정 후 반환)
        text      : 출력할 문자열 (한글 포함 가능)
        org       : (x, y) — cv2.putText 와 동일하게 좌측 하단 기준
        font_size : PIL 폰트 크기(px)
        color     : BGR 색상 튜플
        alpha     : 불투명도 (1.0 = 완전 불투명)

    Returns:
        수정된 img (동일 객체)
    """
    if not text:
        return img

    font = _load_font(font_size)

    # BGR → RGB (PIL 은 RGB 순서)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    draw    = ImageDraw.Draw(pil_img)

    # org 는 cv2 기준 좌측 하단(baseline) → PIL 좌측 상단으로 변환
    x, y = org
    bbox   = font.getbbox(text)          # (left, top, right, bottom)
    text_h = bbox[3] - bbox[1]
    pil_y  = y - text_h - bbox[1]

    # BGR → RGB 색상 변환
    r, g, b = int(color[2]), int(color[1]), int(color[0])

    if alpha < 1.0:
        # 투명도가 있으면 별도 RGBA 레이어에 그린 뒤 합성
        layer      = Image.new("RGBA", pil_img.size, (0, 0, 0, 0))
        layer_draw = ImageDraw.Draw(layer)
        layer_draw.text((x, pil_y), text, font=font,
                        fill=(r, g, b, int(255 * alpha)))
        pil_img = pil_img.convert("RGBA")
        pil_img = Image.alpha_composite(pil_img, layer)
        pil_img = pil_img.convert("RGB")
    else:
        draw.text((x, pil_y), text, font=font, fill=(r, g, b))

    # RGB → BGR 후 원본 배열에 복사
    result = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    np.copyto(img, result)
    return img


def get_text_size_kr(
    text: str,
    font_size: int = 18,
) -> Tuple[int, int]:
    """
    한글 텍스트의 픽셀 크기를 반환합니다.

    cv2.getTextSize() 의 대체 함수입니다.

    Returns:
        (width, height) 픽셀 단위
    """
    if not text:
        return 0, 0
    font = _load_font(font_size)
    bbox = font.getbbox(text)    # (left, top, right, bottom)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]
