"""
패스스루 프로세서 (processors/passthrough.py)
아무 처리 없이 원본 프레임을 그대로 통과시킵니다. (기본/기준 파이프라인)
"""

import numpy as np
from ..core.base_processor import BaseProcessor
from ..core.frame import Frame


class PassthroughProcessor(BaseProcessor):
    """원본 이미지를 processed에 복사만 합니다."""

    def __init__(self) -> None:
        super().__init__(
            name="passthrough",
            description="원본 영상 그대로 출력 (No Processing)",
        )

    def process(self, frame: Frame) -> Frame:
        frame.processed = frame.image.copy()
        return frame
