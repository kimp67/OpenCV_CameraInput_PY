"""
파이프라인 관리자 (core/pipeline.py)
여러 프로세서를 체인으로 연결하고 프레임별 실행을 관리
"""

from __future__ import annotations

import logging
import time
from typing import Dict, List, Optional

from .base_processor import BaseProcessor
from .frame import Frame


class Pipeline:
    """
    복수의 BaseProcessor를 순서대로 실행하는 파이프라인.

    프로세서는 list 순서대로 체인으로 연결되며,
    각 프로세서의 출력(frame.processed)이 다음 입력의
    frame.image 로 넘어가지 않고 frame 객체 자체를 전달합니다.
    (각 프로세서가 frame.output을 참조하여 처리하면 됩니다.)

    Attributes:
        name        : 파이프라인 이름
        description : 설명
        processors  : 순서가 있는 프로세서 목록
    """

    def __init__(self, name: str, description: str = "") -> None:
        self.name = name
        self.description = description
        self.processors: List[BaseProcessor] = []
        self._logger = logging.getLogger(f"cv_stream.pipeline.{name}")
        self._process_times: Dict[str, float] = {}   # 각 단계 처리 시간 기록

    # ── 프로세서 관리 ───────────────────────────────────────────────

    def add(self, processor: BaseProcessor) -> "Pipeline":
        """프로세서를 파이프라인 끝에 추가합니다. (메서드 체이닝 지원)"""
        self.processors.append(processor)
        self._logger.debug("Added processor: %s", processor.name)
        return self

    def remove(self, name: str) -> bool:
        """이름으로 프로세서를 제거합니다."""
        before = len(self.processors)
        self.processors = [p for p in self.processors if p.name != name]
        removed = len(self.processors) < before
        if removed:
            self._logger.info("Removed processor: %s", name)
        return removed

    def get(self, name: str) -> Optional[BaseProcessor]:
        """이름으로 프로세서를 조회합니다."""
        for p in self.processors:
            if p.name == name:
                return p
        return None

    def clear(self) -> None:
        """모든 프로세서를 제거합니다."""
        self.processors.clear()

    # ── 파이프라인 실행 ─────────────────────────────────────────────

    def run(self, frame: Frame) -> Frame:
        """
        프레임을 파이프라인의 모든 프로세서에 순서대로 통과시킵니다.

        Args:
            frame: 입력 Frame

        Returns:
            최종 처리 Frame
        """
        current = frame
        self._process_times.clear()

        for processor in self.processors:
            t_start = time.perf_counter()
            current = processor(current)
            self._process_times[processor.name] = (time.perf_counter() - t_start) * 1000  # ms

        return current

    # ── 초기화 / 해제 ───────────────────────────────────────────────

    def initialize(self) -> None:
        """파이프라인 내 모든 프로세서를 초기화합니다."""
        for p in self.processors:
            try:
                p.initialize()
                self._logger.info("Initialized: %s", p.name)
            except Exception as exc:
                self._logger.error("Failed to initialize '%s': %s", p.name, exc)

    def release(self) -> None:
        """파이프라인 내 모든 프로세서의 리소스를 해제합니다."""
        for p in self.processors:
            try:
                p.release()
            except Exception as exc:
                self._logger.warning("Error releasing '%s': %s", p.name, exc)

    # ── 정보 조회 ───────────────────────────────────────────────────

    @property
    def process_times(self) -> Dict[str, float]:
        """최근 run()에서 각 프로세서의 처리 시간(ms)."""
        return dict(self._process_times)

    @property
    def total_time_ms(self) -> float:
        """최근 run()의 전체 처리 시간(ms)."""
        return sum(self._process_times.values())

    def summary(self) -> str:
        """파이프라인 구성 요약 문자열."""
        lines = [f"Pipeline: {self.name}"]
        for i, p in enumerate(self.processors):
            status = "ON " if p.enabled else "OFF"
            lines.append(f"  [{i+1}] [{status}] {p.name} — {p.description}")
        return "\n".join(lines)

    def __len__(self) -> int:
        return len(self.processors)

    def __repr__(self) -> str:
        return f"Pipeline(name={self.name!r}, processors={len(self.processors)})"
