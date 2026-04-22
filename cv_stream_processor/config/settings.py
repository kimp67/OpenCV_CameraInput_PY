"""
설정 파일 (config/settings.py)
애플리케이션 전역 설정 관리
"""

import os
from dataclasses import dataclass, field
from typing import Tuple, Optional, Dict, Any


@dataclass
class CameraConfig:
    """카메라 설정"""
    source: Any = 0                          # 카메라 인덱스(int) 또는 스트림 URL(str)
    width: int = 1280
    height: int = 720
    fps: int = 30
    buffer_size: int = 1                     # 버퍼 크기 (낮을수록 지연 감소)
    reconnect_delay: float = 2.0             # 연결 실패 시 재시도 대기 시간(초)
    max_reconnect: int = 5                   # 최대 재연결 시도 횟수


@dataclass
class DisplayConfig:
    """화면 출력 설정"""
    window_name: str = "CV Stream Processor"
    show_fps: bool = True
    show_pipeline_info: bool = True
    show_help: bool = True
    font_scale: float = 0.6
    font_thickness: int = 1
    overlay_alpha: float = 0.6              # 오버레이 투명도
    info_panel_width: int = 280            # 정보 패널 너비


@dataclass
class PipelineConfig:
    """파이프라인 설정"""
    max_queue_size: int = 5                 # 프레임 큐 최대 크기
    processing_threads: int = 2            # 처리 스레드 수
    skip_frames: int = 0                   # 처리 건너뛸 프레임 수 (0=모두 처리)


@dataclass
class LogConfig:
    """로깅 설정"""
    log_dir: str = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")
    log_level: str = "INFO"                 # DEBUG, INFO, WARNING, ERROR, CRITICAL
    log_to_file: bool = True
    log_filename: str = "stream_processor.log"
    max_bytes: int = 10 * 1024 * 1024      # 10MB
    backup_count: int = 5


@dataclass
class AppConfig:
    """애플리케이션 전체 설정"""
    camera: CameraConfig = field(default_factory=CameraConfig)
    display: DisplayConfig = field(default_factory=DisplayConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    log: LogConfig = field(default_factory=LogConfig)

    # 기본 활성화할 파이프라인 이름 (None이면 'passthrough')
    default_pipeline: str = "passthrough"

    # 저장 경로
    save_dir: str = os.path.join(os.path.dirname(os.path.dirname(__file__)), "output")

    # 녹화 설정
    record_fourcc: str = "XVID"
    record_fps: float = 20.0


# ── 전역 기본 설정 인스턴스 ──────────────────────────────────────────
DEFAULT_CONFIG = AppConfig()
