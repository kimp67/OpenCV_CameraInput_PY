"""
베이스 프로세서 추상 클래스 (core/base_processor.py)
모든 이미지 처리 모듈이 상속해야 하는 인터페이스 정의
"""

from __future__ import annotations

import abc
import logging
from typing import Any, Dict, Optional

from .frame import Frame


class BaseProcessor(abc.ABC):
    """
    이미지 처리 프로세서의 추상 기반 클래스.

    모든 프로세서는 이 클래스를 상속하고
    `process()` 메서드를 구현해야 합니다.

    Attributes:
        name        : 프로세서 고유 이름 (파이프라인 식별용)
        description : 사람이 읽을 수 있는 설명
        enabled     : 활성화/비활성화 플래그
        params      : 동적으로 변경 가능한 파라미터 딕셔너리
    """

    def __init__(
        self,
        name: str,
        description: str = "",
        enabled: bool = True,
        params: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.name = name
        self.description = description
        self.enabled = enabled
        self.params: Dict[str, Any] = params or {}
        self._logger = logging.getLogger(f"cv_stream.processor.{name}")

    # ── 구현 필수 메서드 ────────────────────────────────────────────

    @abc.abstractmethod
    def process(self, frame: Frame) -> Frame:
        """
        프레임을 처리하고 결과를 반환합니다.

        Args:
            frame: 입력 Frame 객체

        Returns:
            처리 결과가 담긴 Frame 객체
            (frame.processed 에 결과 이미지를 저장하세요)
        """

    # ── 선택적 오버라이드 메서드 ────────────────────────────────────

    def initialize(self) -> None:
        """프로세서 초기화 (파이프라인 시작 시 한 번 호출)."""

    def release(self) -> None:
        """리소스 해제 (파이프라인 종료 시 호출)."""

    def get_param(self, key: str, default: Any = None) -> Any:
        """파라미터 값을 안전하게 가져옵니다."""
        return self.params.get(key, default)

    def set_param(self, key: str, value: Any) -> None:
        """파라미터를 동적으로 설정합니다."""
        old = self.params.get(key)
        self.params[key] = value
        self._logger.debug("Param [%s]: %s → %s", key, old, value)

    def toggle(self) -> bool:
        """활성화/비활성화를 전환하고 현재 상태를 반환합니다."""
        self.enabled = not self.enabled
        self._logger.info("Processor '%s' %s", self.name, "enabled" if self.enabled else "disabled")
        return self.enabled

    # ── 파이프라인 내부 호출 메서드 ─────────────────────────────────

    def __call__(self, frame: Frame) -> Frame:
        """
        파이프라인에서 직접 호출.
        enabled=False 이면 원본을 그대로 통과시킵니다.
        """
        if not self.enabled:
            return frame
        try:
            return self.process(frame)
        except Exception as exc:
            self._logger.error("Error in processor '%s': %s", self.name, exc, exc_info=True)
            return frame  # 오류 시 원본 프레임 반환

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"name={self.name!r}, "
            f"enabled={self.enabled}, "
            f"params={self.params})"
        )
