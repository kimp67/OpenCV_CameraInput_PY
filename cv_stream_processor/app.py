"""
메인 애플리케이션 (app.py)
스트림 캡처 → 파이프라인 처리 → 화면 출력을 통합 제어하는 루프
"""

from __future__ import annotations

import logging
import os
import sys
import time
from typing import Optional

import cv2

from .config import AppConfig, DEFAULT_CONFIG
from .core import Frame, PipelineRegistry, StreamCapture
from .core.pipeline_factory import register_all
from .ui import draw_overlay, ActionHandler, process_key
from .ui.pil_text import BgrPilTextLayer, text_size
from .utils import FPSCounter, app_logger


class CVStreamApp:
    """
    메인 애플리케이션 클래스.

    구조:
        StreamCapture  →  PipelineRegistry  →  draw_overlay  →  cv2.imshow
             ↑                    ↑                                  ↓
        백그라운드 스레드       파이프라인 선택 (N/P/숫자키)       키보드 입력 처리

    Args:
        config: AppConfig 인스턴스 (None이면 기본값 사용)
    """

    def __init__(self, config: Optional[AppConfig] = None) -> None:
        self._cfg = config or DEFAULT_CONFIG
        self._logger = logging.getLogger("cv_stream.app")

        # ── 출력 디렉터리 ────────────────────────────────────────────
        os.makedirs(self._cfg.save_dir, exist_ok=True)

        # ── 파이프라인 레지스트리 ────────────────────────────────────
        self._registry = PipelineRegistry(default_name=self._cfg.default_pipeline)
        register_all(self._registry)
        self._registry.select(self._cfg.default_pipeline)
        self._logger.info("Registered %d pipelines.", len(self._registry.pipeline_names))

        # ── 스트림 캡처 ──────────────────────────────────────────────
        self._capture = StreamCapture(self._cfg.camera)

        # ── UI 컴포넌트 ──────────────────────────────────────────────
        self._fps = FPSCounter(window_size=30)
        self._action = ActionHandler(self._registry, self._cfg)

        # ── 상태 변수 ────────────────────────────────────────────────
        self._frame_id: int = 0
        self._skip_count: int = 0

    # ── 공개 API ────────────────────────────────────────────────────

    def run(self) -> None:
        """애플리케이션 메인 루프를 실행합니다."""
        self._logger.info("=" * 60)
        self._logger.info("CV Stream Processor Starting...")
        self._logger.info("  Source  : %s", self._cfg.camera.source)
        self._logger.info("  Pipeline: %s", self._cfg.default_pipeline)
        self._logger.info("  Save dir: %s", self._cfg.save_dir)
        self._logger.info("=" * 60)

        # 캡처 시작
        if not self._capture.start():
            self._logger.error("Failed to open camera/stream source: %s", self._cfg.camera.source)
            self._show_no_signal()
            return

        # OpenCV 윈도우 초기화
        win_name = self._cfg.display.window_name
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win_name, self._cfg.camera.width, self._cfg.camera.height)

        try:
            self._main_loop(win_name)
        except KeyboardInterrupt:
            self._logger.info("Interrupted by user (Ctrl+C).")
        finally:
            self._cleanup()

        self._logger.info("Application exited. Total frames: %d", self._fps.total_frames)

    # ── 내부 루프 ───────────────────────────────────────────────────

    def _main_loop(self, win_name: str) -> None:
        """메인 처리 루프."""
        skip = self._cfg.pipeline.skip_frames

        while True:
            # ── 1. 프레임 읽기 ──────────────────────────────────────
            ret, raw = self._capture.read()

            if not ret or raw is None:
                # 신호 없음 화면 표시 후 재시도
                no_signal = self._make_no_signal(
                    self._cfg.camera.width, self._cfg.camera.height
                )
                cv2.imshow(win_name, no_signal)
                key = cv2.waitKey(100)
                event = process_key(key)
                if not self._action.handle(event):
                    break
                continue

            # ── 2. 프레임 건너뛰기 (CPU 절약) ───────────────────────
            if skip > 0:
                self._skip_count += 1
                if self._skip_count % (skip + 1) != 0:
                    continue

            # ── 3. 일시정지 ─────────────────────────────────────────
            if self._action.paused:
                # 일시정지 중에도 오버레이 표시
                frame_obj = Frame(image=raw, frame_id=self._frame_id)
                frame_obj.processed = raw.copy()
                display = self._render(frame_obj)
                cv2.imshow(win_name, display)
                key = cv2.waitKey(50)
                event = process_key(key)
                if not self._action.handle(event, display):
                    break
                continue

            # ── 4. Frame 객체 생성 ──────────────────────────────────
            self._frame_id += 1
            frame_obj = Frame(
                image=raw,
                frame_id=self._frame_id,
                source_id=self._cfg.camera.source,
            )

            # ── 5. 파이프라인 처리 ──────────────────────────────────
            try:
                processed_frame = self._registry.run(frame_obj)
            except Exception as exc:
                self._logger.error("Pipeline error: %s", exc, exc_info=True)
                processed_frame = frame_obj
                processed_frame.processed = raw.copy()

            # ── 6. FPS 업데이트 ─────────────────────────────────────
            self._fps.tick()

            # ── 7. 오버레이 렌더링 ───────────────────────────────────
            display = self._render(processed_frame)

            # ── 8. 화면 출력 ─────────────────────────────────────────
            cv2.imshow(win_name, display)

            # ── 9. 녹화 ─────────────────────────────────────────────
            self._action.write_frame(display)

            # ── 10. 키 입력 처리 ────────────────────────────────────
            key = cv2.waitKey(1)
            event = process_key(key)
            if not self._action.handle(event, display):
                break

    # ── 렌더링 ──────────────────────────────────────────────────────

    def _render(self, frame: Frame) -> "cv2.Mat":
        """오버레이를 합성한 최종 출력 이미지를 반환합니다."""
        output = frame.output  # processed or original
        # 원본(캡처) 해상도를 오버레이에 전달
        src_h, src_w = frame.image.shape[:2]
        return draw_overlay(
            frame=output,
            registry=self._registry,
            fps_counter=self._fps,
            config=self._cfg.display,
            recording=self._action.recording,
            paused=self._action.paused,
            show_help=self._action.show_help,
            frame_width=src_w,
            frame_height=src_h,
        )

    # ── 유틸리티 ────────────────────────────────────────────────────

    def _cleanup(self) -> None:
        """리소스를 해제합니다."""
        self._logger.info("Releasing resources...")
        self._capture.stop()
        self._registry.release_all()
        cv2.destroyAllWindows()

    @staticmethod
    def _make_no_signal(w: int = 1280, h: int = 720) -> "cv2.Mat":
        """신호 없음 이미지를 생성합니다."""
        import numpy as np

        img = np.zeros((h, w, 3), dtype="uint8")
        msg = "신호 없음 — 카메라·스트림 연결 대기 중"
        cv_scale = 1.0
        tw, th = text_size(msg, cv_scale)
        y = (h + th) // 2
        txt = BgrPilTextLayer(img)
        txt.put(msg, ((w - tw) // 2, y), cv_scale, (0, 180, 255), 2)
        txt.commit()
        return img

    def _show_no_signal(self) -> None:
        """카메라 연결 실패 시 메시지를 표시합니다."""
        win = self._cfg.display.window_name
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        img = self._make_no_signal(self._cfg.camera.width, self._cfg.camera.height)
        cv2.imshow(win, img)
        self._logger.error(
            "Cannot open source '%s'. Press any key to exit.", self._cfg.camera.source
        )
        cv2.waitKey(0)
        cv2.destroyAllWindows()
