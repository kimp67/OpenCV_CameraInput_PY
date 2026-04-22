"""
프레임 데이터 클래스 (core/frame.py)
파이프라인 전체에서 사용하는 프레임 컨테이너
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np


@dataclass
class Frame:
    """
    카메라에서 캡처된 단일 프레임과 메타데이터를 담는 컨테이너.

    Attributes:
        image       : BGR numpy 배열 (원본 이미지)
        processed   : 처리 후 결과 이미지 (처리 전에는 None)
        frame_id    : 누적 프레임 번호
        timestamp   : 캡처 시각 (perf_counter 기준, 초)
        source_id   : 카메라 소스 식별자
        metadata    : 파이프라인 단계별 추가 정보 저장용 딕셔너리
    """

    image: np.ndarray
    frame_id: int = 0
    timestamp: float = field(default_factory=time.perf_counter)
    source_id: Any = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    processed: Optional[np.ndarray] = None

    # ── 편의 프로퍼티 ──────────────────────────────────────────────

    @property
    def height(self) -> int:
        return self.image.shape[0]

    @property
    def width(self) -> int:
        return self.image.shape[1]

    @property
    def channels(self) -> int:
        return self.image.shape[2] if self.image.ndim == 3 else 1

    @property
    def shape(self):
        return self.image.shape

    @property
    def output(self) -> np.ndarray:
        """처리 결과 이미지, 없으면 원본 반환."""
        return self.processed if self.processed is not None else self.image

    def clone(self) -> "Frame":
        """프레임의 깊은 복사본을 반환합니다."""
        return Frame(
            image=self.image.copy(),
            frame_id=self.frame_id,
            timestamp=self.timestamp,
            source_id=self.source_id,
            metadata=dict(self.metadata),
            processed=self.processed.copy() if self.processed is not None else None,
        )

    def __repr__(self) -> str:
        return (
            f"Frame(id={self.frame_id}, "
            f"size={self.width}x{self.height}, "
            f"ch={self.channels}, "
            f"ts={self.timestamp:.3f})"
        )
