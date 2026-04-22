"""
파이프라인 레지스트리 (core/pipeline_registry.py)
사용 가능한 파이프라인을 등록하고 선택/전환하는 관리자
"""

from __future__ import annotations

import logging
from typing import Callable, Dict, List, Optional

from .pipeline import Pipeline


# 파이프라인 팩토리 타입: 호출 시 Pipeline 인스턴스를 반환하는 callable
PipelineFactory = Callable[[], Pipeline]


class PipelineRegistry:
    """
    파이프라인을 이름으로 등록하고, 런타임에 전환하는 레지스트리.

    Usage:
        registry = PipelineRegistry()
        registry.register("edge", lambda: build_edge_pipeline())
        registry.select("edge")
        frame = registry.run(frame)
    """

    def __init__(self, default_name: str = "passthrough") -> None:
        self._factories: Dict[str, PipelineFactory] = {}
        self._descriptions: Dict[str, str] = {}
        self._active_pipeline: Optional[Pipeline] = None
        self._active_name: str = ""
        self._default_name = default_name
        self._logger = logging.getLogger("cv_stream.registry")

    # ── 등록 ────────────────────────────────────────────────────────

    def register(
        self,
        name: str,
        factory: PipelineFactory,
        description: str = "",
    ) -> None:
        """
        파이프라인 팩토리를 등록합니다.

        Args:
            name       : 파이프라인 고유 이름
            factory    : Pipeline 인스턴스를 반환하는 callable
            description: 설명 (UI 표시용)
        """
        self._factories[name] = factory
        self._descriptions[name] = description
        self._logger.debug("Registered pipeline: %s", name)

    def unregister(self, name: str) -> bool:
        """등록된 파이프라인을 제거합니다."""
        if name in self._factories:
            del self._factories[name]
            self._descriptions.pop(name, None)
            return True
        return False

    # ── 선택 / 전환 ─────────────────────────────────────────────────

    def select(self, name: str) -> bool:
        """
        사용할 파이프라인을 이름으로 선택합니다.
        이전 파이프라인의 리소스를 해제하고 새 파이프라인을 초기화합니다.

        Returns:
            성공 여부
        """
        if name not in self._factories:
            self._logger.error("Pipeline not found: %s", name)
            return False

        # 현재 파이프라인 해제
        if self._active_pipeline is not None:
            self._active_pipeline.release()

        # 새 파이프라인 생성 및 초기화
        try:
            pipeline = self._factories[name]()
            pipeline.initialize()
            self._active_pipeline = pipeline
            self._active_name = name
            self._logger.info("Switched to pipeline: %s", name)
            return True
        except Exception as exc:
            self._logger.error("Failed to create pipeline '%s': %s", name, exc)
            return False

    def next_pipeline(self) -> str:
        """등록된 파이프라인을 순환하며 다음 파이프라인으로 전환합니다."""
        names = self.pipeline_names
        if not names:
            return self._active_name
        if self._active_name not in names:
            self.select(names[0])
            return names[0]
        idx = (names.index(self._active_name) + 1) % len(names)
        next_name = names[idx]
        self.select(next_name)
        return next_name

    def prev_pipeline(self) -> str:
        """등록된 파이프라인을 순환하며 이전 파이프라인으로 전환합니다."""
        names = self.pipeline_names
        if not names:
            return self._active_name
        if self._active_name not in names:
            self.select(names[0])
            return names[0]
        idx = (names.index(self._active_name) - 1) % len(names)
        prev_name = names[idx]
        self.select(prev_name)
        return prev_name

    # ── 실행 ────────────────────────────────────────────────────────

    def run(self, frame):
        """현재 선택된 파이프라인으로 프레임을 처리합니다."""
        if self._active_pipeline is None:
            # 파이프라인이 선택되지 않았으면 기본값 시도
            self.select(self._default_name)
        if self._active_pipeline is None:
            return frame
        return self._active_pipeline.run(frame)

    # ── 정보 조회 ───────────────────────────────────────────────────

    @property
    def pipeline_names(self) -> List[str]:
        """등록된 파이프라인 이름 목록."""
        return list(self._factories.keys())

    @property
    def active_name(self) -> str:
        """현재 활성 파이프라인 이름."""
        return self._active_name

    @property
    def active_pipeline(self) -> Optional[Pipeline]:
        """현재 활성 Pipeline 인스턴스."""
        return self._active_pipeline

    def get_description(self, name: str) -> str:
        """파이프라인 설명을 반환합니다."""
        return self._descriptions.get(name, "")

    def release_all(self) -> None:
        """활성 파이프라인의 모든 리소스를 해제합니다."""
        if self._active_pipeline:
            self._active_pipeline.release()
            self._active_pipeline = None

    def __repr__(self) -> str:
        return (
            f"PipelineRegistry("
            f"pipelines={self.pipeline_names}, "
            f"active={self._active_name!r})"
        )
