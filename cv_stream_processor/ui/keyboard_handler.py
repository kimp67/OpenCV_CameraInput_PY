"""
키보드 입력 처리기 (ui/keyboard_handler.py)
OpenCV waitKey 결과를 해석하여 애플리케이션 명령으로 변환
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING, Optional

import cv2
import numpy as np

if TYPE_CHECKING:
    from ..core.pipeline_registry import PipelineRegistry
    from ..config import AppConfig

logger = logging.getLogger("cv_stream.keyboard")


class AppCommand(Enum):
    """키 입력으로 생성되는 애플리케이션 명령."""
    NONE        = auto()
    QUIT        = auto()
    PAUSE       = auto()
    NEXT_PIPE   = auto()
    PREV_PIPE   = auto()
    SCREENSHOT  = auto()
    RECORD      = auto()
    TOGGLE_HELP = auto()
    FULLSCREEN  = auto()
    SELECT_PIPE = auto()   # 숫자키로 직접 선택


@dataclass
class KeyEvent:
    command: AppCommand = AppCommand.NONE
    pipe_index: Optional[int] = None   # SELECT_PIPE 시 파이프라인 인덱스


# ── 키 매핑 ─────────────────────────────────────────────────────────
_KEY_MAP = {
    ord("q"): AppCommand.QUIT,
    27       : AppCommand.QUIT,        # ESC
    ord(" "): AppCommand.PAUSE,
    ord("n"): AppCommand.NEXT_PIPE,
    ord("p"): AppCommand.PREV_PIPE,
    ord("s"): AppCommand.SCREENSHOT,
    ord("r"): AppCommand.RECORD,
    ord("h"): AppCommand.TOGGLE_HELP,
    ord("f"): AppCommand.FULLSCREEN,
}


def process_key(key: int) -> KeyEvent:
    """
    waitKey(1) 반환값을 KeyEvent로 변환합니다.

    Args:
        key: cv2.waitKey() 반환값

    Returns:
        KeyEvent 인스턴스
    """
    if key == -1:
        return KeyEvent()

    key = key & 0xFF  # 상위 비트 제거

    if key in _KEY_MAP:
        return KeyEvent(command=_KEY_MAP[key])

    # 숫자 키 (0 ~ 9) → 파이프라인 직접 선택
    if ord("0") <= key <= ord("9"):
        return KeyEvent(command=AppCommand.SELECT_PIPE, pipe_index=key - ord("0"))

    return KeyEvent()


class ActionHandler:
    """
    KeyEvent를 받아 실제 동작을 수행하는 핸들러.

    Args:
        registry: PipelineRegistry
        config  : AppConfig
    """

    def __init__(self, registry: "PipelineRegistry", config: "AppConfig") -> None:
        self._registry = registry
        self._config = config
        self.paused = False
        self.show_help = True
        self.recording = False
        self._writer: Optional[cv2.VideoWriter] = None
        self._fullscreen = False

    def handle(self, event: KeyEvent, frame: Optional[np.ndarray] = None) -> bool:
        """
        이벤트를 처리합니다.

        Returns:
            False이면 애플리케이션 종료 신호
        """
        cmd = event.command

        if cmd == AppCommand.QUIT:
            logger.info("Quit command received.")
            self._stop_recording()
            return False

        elif cmd == AppCommand.PAUSE:
            self.paused = not self.paused
            logger.info("Paused: %s", self.paused)

        elif cmd == AppCommand.NEXT_PIPE:
            name = self._registry.next_pipeline()
            logger.info("Switched to pipeline: %s", name)

        elif cmd == AppCommand.PREV_PIPE:
            name = self._registry.prev_pipeline()
            logger.info("Switched to pipeline: %s", name)

        elif cmd == AppCommand.SELECT_PIPE:
            names = self._registry.pipeline_names
            idx = event.pipe_index or 0
            if idx < len(names):
                self._registry.select(names[idx])
                logger.info("Selected pipeline [%d]: %s", idx, names[idx])

        elif cmd == AppCommand.SCREENSHOT:
            self._save_screenshot(frame)

        elif cmd == AppCommand.RECORD:
            if self.recording:
                self._stop_recording()
            else:
                self._start_recording(frame)

        elif cmd == AppCommand.TOGGLE_HELP:
            self.show_help = not self.show_help

        elif cmd == AppCommand.FULLSCREEN:
            self._toggle_fullscreen()

        return True

    # ── 스크린샷 ─────────────────────────────────────────────────────

    def _save_screenshot(self, frame: Optional[np.ndarray]) -> None:
        if frame is None:
            return
        os.makedirs(self._config.save_dir, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        pipe = self._registry.active_name
        path = os.path.join(self._config.save_dir, f"screenshot_{pipe}_{ts}.png")
        cv2.imwrite(path, frame)
        logger.info("Screenshot saved: %s", path)

    # ── 녹화 ─────────────────────────────────────────────────────────

    def _start_recording(self, frame: Optional[np.ndarray]) -> None:
        if frame is None:
            return
        os.makedirs(self._config.save_dir, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        pipe = self._registry.active_name
        path = os.path.join(self._config.save_dir, f"record_{pipe}_{ts}.avi")
        h, w = frame.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*self._config.record_fourcc)
        self._writer = cv2.VideoWriter(path, fourcc, self._config.record_fps, (w, h))
        self.recording = True
        logger.info("Recording started: %s", path)

    def _stop_recording(self) -> None:
        if self._writer:
            self._writer.release()
            self._writer = None
        self.recording = False
        logger.info("Recording stopped.")

    def write_frame(self, frame: np.ndarray) -> None:
        """녹화 중이면 프레임을 파일에 씁니다."""
        if self.recording and self._writer:
            self._writer.write(frame)

    # ── 전체화면 토글 ────────────────────────────────────────────────

    def _toggle_fullscreen(self) -> None:
        win = self._config.display.window_name
        if self._fullscreen:
            cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
        else:
            cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        self._fullscreen = not self._fullscreen
