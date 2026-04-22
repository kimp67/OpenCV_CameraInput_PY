"""
스트림 캡처 모듈 (core/stream_capture.py)
카메라/스트림 URL에서 프레임을 안정적으로 읽어오는 스레드 기반 캡처기
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Optional, Tuple

import cv2
import numpy as np

from ..config import CameraConfig, DEFAULT_CONFIG


class StreamCapture:
    """
    별도 스레드에서 카메라/스트림을 지속적으로 읽어
    최신 프레임을 유지하는 비차단(non-blocking) 캡처기.

    - 카메라 인덱스(int) 또는 스트림 URL(str) 지원
    - RTSP / HTTP MJPEG / 파일 경로 모두 사용 가능
    - 연결 실패 시 자동 재연결
    - thread-safe 최신 프레임 제공

    Args:
        config: CameraConfig 인스턴스 (None이면 기본값 사용)
    """

    def __init__(self, config: Optional[CameraConfig] = None) -> None:
        self._cfg = config or DEFAULT_CONFIG.camera
        self._logger = logging.getLogger("cv_stream.capture")

        self._cap: Optional[cv2.VideoCapture] = None
        self._frame: Optional[np.ndarray] = None
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._frame_count: int = 0
        self._connected: bool = False
        self._reconnect_count: int = 0

    # ── 공개 API ────────────────────────────────────────────────────

    def start(self) -> bool:
        """
        캡처 스레드를 시작합니다.

        Returns:
            초기 연결 성공 여부
        """
        if self._thread and self._thread.is_alive():
            self._logger.warning("Capture already running.")
            return True

        self._stop_event.clear()
        connected = self._connect()

        self._thread = threading.Thread(
            target=self._capture_loop,
            name="StreamCaptureThread",
            daemon=True,
        )
        self._thread.start()
        self._logger.info(
            "Capture started — source=%s, resolution=%dx%d",
            self._cfg.source,
            self._cfg.width,
            self._cfg.height,
        )
        return connected

    def stop(self) -> None:
        """캡처 스레드를 안전하게 종료합니다."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=3.0)
        self._release()
        self._logger.info("Capture stopped. Total frames captured: %d", self._frame_count)

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        최신 프레임을 반환합니다.

        Returns:
            (success: bool, frame: ndarray | None)
        """
        with self._lock:
            if self._frame is None:
                return False, None
            return True, self._frame.copy()

    @property
    def is_connected(self) -> bool:
        """현재 연결 상태."""
        return self._connected

    @property
    def frame_count(self) -> int:
        """누적 캡처 프레임 수."""
        return self._frame_count

    @property
    def source(self) -> Any:
        return self._cfg.source

    # ── 내부 메서드 ─────────────────────────────────────────────────

    def _connect(self) -> bool:
        """VideoCapture를 초기화하고 해상도/FPS를 설정합니다."""
        source = self._cfg.source

        # 정수면 로컬 카메라, 문자열이면 URL/파일
        if isinstance(source, str) and source.isdigit():
            source = int(source)

        self._logger.info("Connecting to source: %s", source)

        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            self._logger.error("Failed to open source: %s", source)
            self._connected = False
            return False

        # 버퍼 크기 최소화 (지연 감소)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, self._cfg.buffer_size)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._cfg.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._cfg.height)
        cap.set(cv2.CAP_PROP_FPS, self._cfg.fps)

        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = cap.get(cv2.CAP_PROP_FPS)
        self._logger.info(
            "Camera opened: %dx%d @ %.1f fps", actual_w, actual_h, actual_fps
        )

        self._cap = cap
        self._connected = True
        self._reconnect_count = 0
        return True

    def _release(self) -> None:
        """VideoCapture 리소스를 해제합니다."""
        if self._cap and self._cap.isOpened():
            self._cap.release()
        self._cap = None
        self._connected = False

    def _capture_loop(self) -> None:
        """백그라운드 스레드에서 지속적으로 프레임을 읽는 루프."""
        while not self._stop_event.is_set():
            if self._cap is None or not self._cap.isOpened():
                # 재연결 시도
                if self._reconnect_count >= self._cfg.max_reconnect:
                    self._logger.error(
                        "Max reconnect attempts (%d) reached. Stopping capture.",
                        self._cfg.max_reconnect,
                    )
                    break
                self._logger.warning(
                    "Connection lost. Reconnecting in %.1fs... (attempt %d/%d)",
                    self._cfg.reconnect_delay,
                    self._reconnect_count + 1,
                    self._cfg.max_reconnect,
                )
                time.sleep(self._cfg.reconnect_delay)
                self._reconnect_count += 1
                self._connect()
                continue

            ret, frame = self._cap.read()
            if not ret or frame is None:
                self._logger.warning("Frame read failed — stream may have ended.")
                self._connected = False
                self._release()
                continue

            with self._lock:
                self._frame = frame
                self._frame_count += 1

        self._logger.debug("Capture loop exited.")

    # ── 컨텍스트 매니저 지원 ────────────────────────────────────────

    def __enter__(self) -> "StreamCapture":
        self.start()
        return self

    def __exit__(self, *args) -> None:
        self.stop()

    def __repr__(self) -> str:
        return (
            f"StreamCapture(source={self._cfg.source!r}, "
            f"connected={self._connected}, "
            f"frames={self._frame_count})"
        )
