"""
CV Stream Processor
====================
OpenCV 기반 실시간 카메라 스트림 이미지 처리 프레임워크.

Python 3.11 + OpenJDK 17 환경 전용
"""

__version__ = "1.0.0"
__author__ = "CV Stream Processor"

from .app import CVStreamApp
from .config import AppConfig, CameraConfig, DisplayConfig, PipelineConfig

__all__ = [
    "CVStreamApp",
    "AppConfig",
    "CameraConfig",
    "DisplayConfig",
    "PipelineConfig",
]
