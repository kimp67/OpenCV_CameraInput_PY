"""
FPS 측정 유틸리티 (utils/fps_counter.py)
슬라이딩 윈도우 방식으로 실시간 FPS 측정
"""

import time
from collections import deque


class FPSCounter:
    """
    슬라이딩 윈도우(deque) 기반 FPS 카운터.

    Args:
        window_size: FPS 평균을 계산할 샘플 개수 (기본 30프레임)
    """

    def __init__(self, window_size: int = 30) -> None:
        self._timestamps: deque = deque(maxlen=window_size)
        self._start_time: float = time.perf_counter()
        self._frame_count: int = 0

    def tick(self) -> None:
        """프레임 완료 시 호출 – 타임스탬프를 기록합니다."""
        self._timestamps.append(time.perf_counter())
        self._frame_count += 1

    @property
    def fps(self) -> float:
        """현재 추정 FPS (슬라이딩 윈도우 평균)."""
        if len(self._timestamps) < 2:
            return 0.0
        elapsed = self._timestamps[-1] - self._timestamps[0]
        return (len(self._timestamps) - 1) / elapsed if elapsed > 0 else 0.0

    @property
    def total_frames(self) -> int:
        """애플리케이션 시작 후 누적 프레임 수."""
        return self._frame_count

    @property
    def elapsed_seconds(self) -> float:
        """애플리케이션 시작 후 경과 시간(초)."""
        return time.perf_counter() - self._start_time

    def reset(self) -> None:
        """카운터를 초기화합니다."""
        self._timestamps.clear()
        self._start_time = time.perf_counter()
        self._frame_count = 0

    def __repr__(self) -> str:
        return (
            f"FPSCounter(fps={self.fps:.1f}, "
            f"frames={self.total_frames}, "
            f"elapsed={self.elapsed_seconds:.1f}s)"
        )
