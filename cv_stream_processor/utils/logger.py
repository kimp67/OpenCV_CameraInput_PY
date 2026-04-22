"""
로거 유틸리티 (utils/logger.py)
애플리케이션 전역 로거 설정
"""

import logging
import os
from logging.handlers import RotatingFileHandler
from typing import Optional

from ..config import LogConfig, DEFAULT_CONFIG


def setup_logger(
    name: str = "cv_stream",
    config: Optional[LogConfig] = None,
) -> logging.Logger:
    """
    구조화된 로거를 생성하고 반환합니다.

    Args:
        name : 로거 이름
        config: 로그 설정 (None이면 기본값 사용)

    Returns:
        설정된 Logger 인스턴스
    """
    cfg = config or DEFAULT_CONFIG.log
    logger = logging.getLogger(name)

    # 이미 핸들러가 설정되어 있으면 재설정 방지
    if logger.handlers:
        return logger

    level = getattr(logging, cfg.log_level.upper(), logging.INFO)
    logger.setLevel(level)

    fmt = logging.Formatter(
        "[%(asctime)s] %(levelname)-8s [%(name)s:%(lineno)d] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # 콘솔 핸들러
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # 파일 핸들러
    if cfg.log_to_file:
        os.makedirs(cfg.log_dir, exist_ok=True)
        log_path = os.path.join(cfg.log_dir, cfg.log_filename)
        fh = RotatingFileHandler(
            log_path,
            maxBytes=cfg.max_bytes,
            backupCount=cfg.backup_count,
            encoding="utf-8",
        )
        fh.setLevel(level)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


# 기본 애플리케이션 로거
app_logger = setup_logger("cv_stream")
