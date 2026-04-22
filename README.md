# CV Stream Processor

**OpenCV 기반 실시간 카메라 스트림 이미지 처리 프레임워크**

Python 3.11 + OpenJDK 17 환경에서 카메라/스트림 영상을 프레임 단위로 분리하여
다양한 이미지 처리 파이프라인을 런타임에 선택·전환할 수 있는 구조로 설계되었습니다.

---

## 📁 프로젝트 구조

```
cv_stream_processor/
├── __init__.py                # 패키지 진입점
├── app.py                     # 메인 애플리케이션 클래스 (CVStreamApp)
│
├── config/
│   ├── __init__.py
│   └── settings.py            # 전역 설정 (Camera / Display / Pipeline / Log)
│
├── core/
│   ├── __init__.py
│   ├── frame.py               # Frame 데이터 컨테이너
│   ├── base_processor.py      # BaseProcessor 추상 클래스
│   ├── pipeline.py            # Pipeline (프로세서 체인)
│   ├── pipeline_registry.py   # PipelineRegistry (파이프라인 관리자)
│   ├── pipeline_factory.py    # 사전 정의 파이프라인 팩토리 & 자동 등록
│   └── stream_capture.py      # StreamCapture (스레드 기반 비차단 캡처)
│
├── processors/
│   ├── __init__.py
│   ├── passthrough.py         # 원본 통과
│   ├── color_filters.py       # 그레이스케일, 반전, HSV, CLAHE, 채널분리, WB
│   ├── blur_sharpen.py        # 가우시안/미디안/바이래터럴/모션 블러, 언샤프, 디테일
│   ├── edge_detection.py      # Canny, Sobel, Laplacian, Scharr
│   ├── morphology.py          # 팽창/침식/열기/닫기/그라디언트/탑햇/블랙햇, 이진화
│   ├── detection.py           # 얼굴/윤곽선/허프직선/코너 검출
│   ├── effects.py             # 카툰/스케치/엠보싱/픽셀화/열화상/오일페인팅
│   └── optical_flow.py        # Lucas-Kanade 희소, Farneback 밀집 광류
│
├── ui/
│   ├── __init__.py
│   ├── overlay.py             # FPS·파이프라인·도움말 오버레이 렌더러
│   └── keyboard_handler.py    # 키 입력 파싱 & 액션 핸들러
│
├── utils/
│   ├── __init__.py
│   ├── logger.py              # 구조화된 로거 (파일 + 콘솔)
│   └── fps_counter.py         # 슬라이딩 윈도우 FPS 측정
│
├── tests/
│   ├── __init__.py
│   └── test_processors.py     # pytest 단위 테스트 (GUI 불필요)
│
└── logs/                      # 자동 생성되는 로그 파일 디렉터리

main.py                        # CLI 진입점
requirements.txt               # 의존성
```

---

## ⚙️ 설치

```bash
# 1. 저장소 클론 / 압축 해제 후 디렉터리 이동
cd cv_stream_processor_project

# 2. 가상환경 생성 (Python 3.11)
python3.11 -m venv .venv
source .venv/bin/activate        # Linux / macOS
# .venv\Scripts\activate         # Windows

# 3. 의존성 설치
pip install -r requirements.txt

# 4. (선택) contrib 기능 사용 시
pip install opencv-contrib-python>=4.8.0
```

---

## 🚀 실행

```bash
# 기본 웹캠 (인덱스 0), passthrough 파이프라인
python main.py

# 웹캠 인덱스 1, Canny 엣지 파이프라인으로 시작
python main.py --source 1 --pipeline canny

# RTSP 스트림
python main.py --source "rtsp://192.168.1.100:554/stream"

# HTTP MJPEG 스트림
python main.py --source "http://192.168.1.100:8080/video"

# 동영상 파일
python main.py --source ./sample.mp4 --pipeline cartoon

# 1080p, 열화상 효과로 시작
python main.py --width 1920 --height 1080 --pipeline thermal

# 사용 가능한 파이프라인 전체 목록 출력
python main.py --list-pipelines
```

---

## ⌨️ 키보드 단축키

| 키      | 기능                                                       |
|--------|---------------------------------------------------|
| `Q` / `ESC` | 애플리케이션 종료                      |
| `SPACE`       | 일시정지 / 재개                          |
| `N`               | 다음 파이프라인으로 전환           |
| `P`               | 이전 파이프라인으로 전환            |
| `0` ~ `9`     | 파이프라인 직접 선택 (등록 순번) |
| `S`               | 스크린샷 저장 (`output/` 폴더)  |
| `R`               | 영상 녹화 시작 / 중지 (AVI)          |
| `H`               | 도움말 오버레이 토글                  |
| `F`                | 전체화면 전환                            |
---------------------------------------------------------------

## 🔄 파이프라인 목록

| # | 이름 | 설명 |
|---|------|------|
| 0 | `passthrough` | 원본 영상 (No Processing) |
| 1 | `grayscale` | 그레이스케일 변환 |
| 2 | `invert` | 색상 반전 |
| 3 | `hsv` | HSV 채널 시각화 |
| 4 | `hist_equal` | 히스토그램 평활화 (CLAHE) |
| 5 | `channel_split` | BGR 채널 분리 |
| 6 | `white_balance` | 그레이월드 화이트밸런스 |
| 7 | `gaussian_blur` | 가우시안 블러 |
| 8 | `median_blur` | 미디안 블러 |
| 9 | `bilateral` | 바이래터럴 필터 |
| 10 | `motion_blur` | 모션 블러 |
| 11 | `unsharp_mask` | 언샤프 마스킹 (선명화) |
| 12 | `detail_enhance` | 디테일 향상 |
| 13 | `canny` | Canny 엣지 검출 |
| 14 | `sobel` | Sobel 엣지 검출 |
| 15 | `laplacian` | Laplacian 엣지 검출 |
| 16 | `scharr` | Scharr 엣지 검출 |
| 17 | `morphology` | 형태학 연산 (팽창/침식) |
| 18 | `thresh_morph` | 이진화 + 형태학 |
| 19 | `face_detect` | 얼굴 검출 (Haar Cascade) |
| 20 | `contour_detect` | 윤곽선 검출 |
| 21 | `hough_line` | 허프 직선 검출 |
| 22 | `corner_detect` | 코너 검출 (Shi-Tomasi) |
| 23 | `cartoon` | 카툰 효과 |
| 24 | `sketch` | 연필 스케치 |
| 25 | `emboss` | 엠보싱 효과 |
| 26 | `pixelate` | 픽셀화 효과 |
| 27 | `thermal` | 열화상 컬러맵 |
| 28 | `oil_paint` | 오일 페인팅 |
| 29 | `lk_flow` | LK 희소 옵티컬 플로우 |
| 30 | `dense_flow` | Farneback 밀집 옵티컬 플로우 |
| 31 | `edge_enhance` | 엣지 강조 (언샤프+Canny 복합) |
| 32 | `face_cartoon` | 얼굴 검출 + 카툰 (복합) |

---

## 🧩 커스텀 프로세서 추가 방법

### 1. `BaseProcessor` 상속

```python
# processors/my_processor.py
import cv2
from cv_stream_processor.core.base_processor import BaseProcessor
from cv_stream_processor.core.frame import Frame

class MyProcessor(BaseProcessor):
    def __init__(self):
        super().__init__(
            name="my_processor",
            description="나만의 커스텀 프로세서",
            params={"intensity": 1.0},   # 동적 파라미터
        )

    def initialize(self):
        # 모델 로드, 버퍼 초기화 등 (선택)
        pass

    def process(self, frame: Frame) -> Frame:
        img = frame.output  # 이전 프로세서의 결과 또는 원본
        # ─── 여기에 처리 로직 작성 ───
        intensity = self.get_param("intensity", 1.0)
        result = cv2.convertScaleAbs(img, alpha=intensity)
        # ──────────────────────────────
        frame.processed = result
        return frame

    def release(self):
        pass  # 리소스 해제 (선택)
```

### 2. 파이프라인 팩토리 등록

```python
# core/pipeline_factory.py 하단 _PIPELINE_DEFS에 추가
from ..processors.my_processor import MyProcessor

def build_my_processor():
    return Pipeline("my_processor", "나만의 프로세서").add(MyProcessor())

_PIPELINE_DEFS.append(
    ("my_processor", build_my_processor, "나만의 커스텀 프로세서")
)
```

### 3. 복합 파이프라인 구성

```python
def build_my_chain():
    return (
        Pipeline("my_chain", "내 복합 파이프라인")
        .add(GrayscaleProcessor())   # Step 1: 그레이스케일
        .add(CannyProcessor())        # Step 2: 엣지 검출
        .add(MyProcessor())           # Step 3: 커스텀 처리
    )
```

---

## 🧪 테스트 실행

```bash
# pytest 설치
pip install pytest

# 전체 테스트 실행 (GUI 불필요)
pytest cv_stream_processor/tests/test_processors.py -v

# 특정 테스트 클래스만
pytest cv_stream_processor/tests/test_processors.py::TestCanny -v
```

---

## 🏗️ 아키텍처 다이어그램

```
┌─────────────────────────────────────────────────────────┐
│                    CVStreamApp                          │
│                                                         │
│  ┌──────────────┐    ┌──────────────────────────────┐   │
│  │ StreamCapture│    │      PipelineRegistry         │   │
│  │  (Thread)    │    │                              │   │
│  │              │    │  ┌──────────┐ ┌──────────┐   │   │
│  │  Camera/URL  │───▶│  │Pipeline A│ │Pipeline B│   │   │
│  │  → Frame     │    │  │          │ │          │   │   │
│  │  (non-block) │    │  │ Proc 1   │ │ Proc 1   │   │   │
│  └──────────────┘    │  │ Proc 2   │ │ Proc 2   │   │   │
│                       │  │ Proc 3   │ │   ...    │   │   │
│                       │  └──────────┘ └──────────┘   │   │
│                       │       ▲  N/P/숫자키로 전환    │   │
│                       └───────┼──────────────────────┘   │
│                               │                          │
│  ┌──────────────────────────────────────────────────┐   │
│  │               UI Layer                           │   │
│  │  overlay.py   ──  FPS / 파이프라인 정보 렌더링   │   │
│  │  keyboard.py  ──  N, P, S, R, H, F, Q ...        │   │
│  └──────────────────────────────────────────────────┘   │
│                               │                          │
│                        cv2.imshow()                      │
└─────────────────────────────────────────────────────────┘

Frame 데이터 흐름:
  raw ndarray  →  Frame(image=raw)  →  Pipeline.run()
  →  Processor1(frame) → Processor2(frame) → ...
  →  frame.processed  →  draw_overlay()  →  화면 출력
```

---

## 📋 환경 요구사항

| 항목 | 버전 |
|------|------|
| Python | 3.11 |
| OpenJDK | 17 (시스템 JVM) |
| opencv-python | ≥ 4.8.0 |
| numpy | ≥ 1.24.0 |
| OS | Linux / macOS / Windows |

---

## 📝 라이선스

MIT License
