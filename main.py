#!/usr/bin/env python3
"""
메인 진입점 (main.py)
커맨드라인 인수를 파싱하여 CVStreamApp을 실행합니다.

사용 예:
    python main.py                         # 기본 웹캠(인덱스 0) 사용
    python main.py --source 1              # 웹캠 인덱스 1
    python main.py --source rtsp://...     # RTSP 스트림
    python main.py --source video.mp4      # 동영상 파일
    python main.py --pipeline canny        # 초기 파이프라인 선택
    python main.py --width 1920 --height 1080 --fps 30
    python main.py --list-pipelines        # 사용 가능한 파이프라인 목록
"""

import argparse
import sys
import os

# ── 패키지 경로 설정 ─────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cv_stream_processor import CVStreamApp
from cv_stream_processor.config import (
    AppConfig, CameraConfig, DisplayConfig, PipelineConfig
)
from cv_stream_processor.core.pipeline_factory import _PIPELINE_DEFS
from cv_stream_processor.utils import setup_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="cv_stream_processor",
        description="OpenCV 기반 실시간 카메라 스트림 이미지 처리 프레임워크",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
키보드 단축키:
  Q / ESC     : 종료
  SPACE       : 일시정지 / 재개
  N           : 다음 파이프라인
  P           : 이전 파이프라인
  0~9         : 파이프라인 직접 선택 (등록 순서)
  S           : 스크린샷 저장
  R           : 녹화 시작/중지
  H           : 도움말 토글
  F           : 전체화면 전환
""",
    )

    # 소스 설정
    parser.add_argument(
        "--source", "-s",
        default="0",
        help="카메라 인덱스(숫자) 또는 스트림/파일 URL (기본: 0)"
    )

    # 해상도/FPS
    parser.add_argument("--width",  type=int, default=1280, help="캡처 너비 (기본: 1280)")
    parser.add_argument("--height", type=int, default=720,  help="캡처 높이 (기본: 720)")
    parser.add_argument("--fps",    type=int, default=30,   help="목표 FPS (기본: 30)")

    # 파이프라인
    pipeline_names = [name for name, _, _ in _PIPELINE_DEFS]
    parser.add_argument(
        "--pipeline", "-p",
        default="passthrough",
        choices=pipeline_names,
        metavar="PIPELINE",
        help=f"초기 파이프라인 (기본: passthrough). 선택 가능: {', '.join(pipeline_names)}"
    )
    parser.add_argument(
        "--list-pipelines", "-l",
        action="store_true",
        help="사용 가능한 파이프라인 목록 출력 후 종료"
    )

    # 처리 설정
    parser.add_argument(
        "--skip-frames", type=int, default=0,
        help="처리를 건너뛸 프레임 수 (0=모두 처리, 기본: 0)"
    )

    # 디스플레이
    parser.add_argument("--no-fps",    action="store_true", help="FPS 오버레이 숨김")
    parser.add_argument("--no-info",   action="store_true", help="파이프라인 정보 패널 숨김")
    parser.add_argument("--no-help",   action="store_true", help="도움말 오버레이 숨김")

    # 로그
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="로그 레벨 (기본: INFO)"
    )
    parser.add_argument(
        "--no-log-file",
        action="store_true",
        help="파일 로그 비활성화"
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    # ── 파이프라인 목록 출력 ─────────────────────────────────────────
    if args.list_pipelines:
        print("\n사용 가능한 파이프라인 목록:")
        print("=" * 55)
        for i, (name, _, desc) in enumerate(_PIPELINE_DEFS):
            print(f"  [{i:2d}] {name:<20s}  {desc}")
        print("=" * 55)
        return 0

    # ── 로거 초기화 ──────────────────────────────────────────────────
    from cv_stream_processor.config import LogConfig
    log_cfg = LogConfig(
        log_level=args.log_level,
        log_to_file=not args.no_log_file,
    )
    setup_logger("cv_stream", log_cfg)

    # ── 소스 결정 (정수면 int로 변환) ────────────────────────────────
    source = args.source
    if source.isdigit():
        source = int(source)

    # ── 설정 객체 구성 ───────────────────────────────────────────────
    config = AppConfig(
        camera=CameraConfig(
            source=source,
            width=args.width,
            height=args.height,
            fps=args.fps,
        ),
        display=DisplayConfig(
            show_fps=not args.no_fps,
            show_pipeline_info=not args.no_info,
            show_help=not args.no_help,
        ),
        pipeline=PipelineConfig(
            skip_frames=args.skip_frames,
        ),
        log=log_cfg,
        default_pipeline=args.pipeline,
    )

    # ── 애플리케이션 실행 ────────────────────────────────────────────
    print(f"""
╔══════════════════════════════════════════════════╗
║      CV Stream Processor  v1.0.0                ║
║      Python 3.11  +  OpenJDK 17                 ║
╠══════════════════════════════════════════════════╣
║  Source   : {str(source):<37s}║
║  Pipeline : {args.pipeline:<37s}║
║  Size     : {args.width}x{args.height:<29s}║
╚══════════════════════════════════════════════════╝
    """)

    app = CVStreamApp(config=config)
    app.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
