"""
파이프라인 팩토리 (core/pipeline_factory.py)
사전 정의된 파이프라인을 생성하는 팩토리 함수 모음
PipelineRegistry에 모두 등록됩니다.
"""

from __future__ import annotations

from .pipeline import Pipeline
from .pipeline_registry import PipelineRegistry
from ..processors import (
    PassthroughProcessor,
    # Color
    GrayscaleProcessor, InvertProcessor, HSVProcessor,
    HistEqualProcessor, ChannelSplitProcessor, WhiteBalanceProcessor,
    # Blur
    GaussianBlurProcessor, MedianBlurProcessor, BilateralProcessor,
    MotionBlurProcessor, UnsharpMaskProcessor, DetailEnhanceProcessor,
    # Edge
    CannyProcessor, SobelProcessor, LaplacianProcessor, ScharrProcessor,
    # Morphology
    MorphologyProcessor, ThresholdMorphProcessor,
    # Detection
    FaceDetectorProcessor, ContourDetectorProcessor,
    HoughLineProcessor, CornerDetectorProcessor,
    # Effects
    CartoonProcessor, SketchProcessor, EmbossProcessor,
    PixelateProcessor, ThermalProcessor, OilPaintProcessor,
    # Flow
    LKFlowProcessor, DenseFlowProcessor,
)


# ══════════════════════════════════════════════════════════════════════
# 개별 파이프라인 팩토리 함수
# ══════════════════════════════════════════════════════════════════════

def build_passthrough() -> Pipeline:
    return Pipeline("passthrough", "원본 영상").add(PassthroughProcessor())


# ── 색상 처리 ────────────────────────────────────────────────────────

def build_grayscale() -> Pipeline:
    return Pipeline("grayscale", "그레이스케일").add(GrayscaleProcessor())

def build_invert() -> Pipeline:
    return Pipeline("invert", "색상 반전").add(InvertProcessor())

def build_hsv() -> Pipeline:
    return Pipeline("hsv", "HSV 채널 시각화").add(HSVProcessor())

def build_hist_equal() -> Pipeline:
    return Pipeline("hist_equal", "CLAHE 히스토그램 평활화").add(HistEqualProcessor())

def build_channel_split() -> Pipeline:
    return Pipeline("channel_split", "BGR 채널 분리").add(ChannelSplitProcessor())

def build_white_balance() -> Pipeline:
    return Pipeline("white_balance", "그레이월드 화이트밸런스").add(WhiteBalanceProcessor())


# ── 블러 / 샤프닝 ───────────────────────────────────────────────────

def build_gaussian_blur() -> Pipeline:
    return Pipeline("gaussian_blur", "가우시안 블러").add(GaussianBlurProcessor())

def build_median_blur() -> Pipeline:
    return Pipeline("median_blur", "미디안 블러").add(MedianBlurProcessor())

def build_bilateral() -> Pipeline:
    return Pipeline("bilateral", "바이래터럴 필터").add(BilateralProcessor())

def build_motion_blur() -> Pipeline:
    return Pipeline("motion_blur", "모션 블러").add(MotionBlurProcessor())

def build_unsharp() -> Pipeline:
    return Pipeline("unsharp_mask", "언샤프 마스킹").add(UnsharpMaskProcessor())

def build_detail_enhance() -> Pipeline:
    return Pipeline("detail_enhance", "디테일 향상").add(DetailEnhanceProcessor())


# ── 엣지 검출 ────────────────────────────────────────────────────────

def build_canny() -> Pipeline:
    return Pipeline("canny", "Canny 엣지").add(CannyProcessor())

def build_sobel() -> Pipeline:
    return Pipeline("sobel", "Sobel 엣지").add(SobelProcessor())

def build_laplacian() -> Pipeline:
    return Pipeline("laplacian", "Laplacian 엣지").add(LaplacianProcessor())

def build_scharr() -> Pipeline:
    return Pipeline("scharr", "Scharr 엣지").add(ScharrProcessor())


# ── 형태학 ───────────────────────────────────────────────────────────

def build_morphology() -> Pipeline:
    return Pipeline("morphology", "형태학 연산").add(MorphologyProcessor())

def build_thresh_morph() -> Pipeline:
    return Pipeline("thresh_morph", "이진화+형태학").add(ThresholdMorphProcessor())


# ── 검출 ─────────────────────────────────────────────────────────────

def build_face_detect() -> Pipeline:
    return Pipeline("face_detect", "얼굴 검출").add(FaceDetectorProcessor())

def build_contour_detect() -> Pipeline:
    return Pipeline("contour_detect", "윤곽선 검출").add(ContourDetectorProcessor())

def build_hough_line() -> Pipeline:
    return Pipeline("hough_line", "허프 직선 검출").add(HoughLineProcessor())

def build_corner_detect() -> Pipeline:
    return Pipeline("corner_detect", "코너 검출").add(CornerDetectorProcessor())


# ── 특수 효과 ────────────────────────────────────────────────────────

def build_cartoon() -> Pipeline:
    return Pipeline("cartoon", "카툰 효과").add(CartoonProcessor())

def build_sketch() -> Pipeline:
    return Pipeline("sketch", "연필 스케치").add(SketchProcessor())

def build_emboss() -> Pipeline:
    return Pipeline("emboss", "엠보싱 효과").add(EmbossProcessor())

def build_pixelate() -> Pipeline:
    return Pipeline("pixelate", "픽셀화 효과").add(PixelateProcessor())

def build_thermal() -> Pipeline:
    return Pipeline("thermal", "열화상 컬러맵").add(ThermalProcessor())

def build_oil_paint() -> Pipeline:
    return Pipeline("oil_paint", "오일 페인팅").add(OilPaintProcessor())


# ── 옵티컬 플로우 ────────────────────────────────────────────────────

def build_lk_flow() -> Pipeline:
    return Pipeline("lk_flow", "LK 희소 광류").add(LKFlowProcessor())

def build_dense_flow() -> Pipeline:
    return Pipeline("dense_flow", "Farneback 밀집 광류").add(DenseFlowProcessor())


# ── 복합 파이프라인 ──────────────────────────────────────────────────

def build_edge_enhance() -> Pipeline:
    """언샤프 마스킹 → Canny 엣지 오버레이 복합 파이프라인."""
    p = Pipeline("edge_enhance", "엣지 강조 (언샤프+Canny 오버레이)")

    class EdgeOverlayProcessor(PassthroughProcessor):
        """원본 + Canny 엣지 오버레이."""
        def __init__(self):
            super().__init__()
            self.name = "edge_overlay"
            self.description = "Canny 엣지 오버레이"

        def process(self, frame):
            img = frame.image.copy()
            unsharp = UnsharpMaskProcessor()
            frame = unsharp.process(frame)
            sharp = frame.output.copy()

            gray = __import__("cv2").cvtColor(sharp, __import__("cv2").COLOR_BGR2GRAY)
            edges = __import__("cv2").Canny(gray, 50, 150)
            import numpy as np
            edges_bgr = __import__("cv2").cvtColor(edges, __import__("cv2").COLOR_GRAY2BGR)
            frame.processed = __import__("cv2").addWeighted(sharp, 0.8, edges_bgr, 0.5, 0)
            return frame

    p.add(EdgeOverlayProcessor())
    return p


def build_face_cartoon() -> Pipeline:
    """얼굴 검출 → 카툰 효과 복합 파이프라인."""
    return (
        Pipeline("face_cartoon", "얼굴 검출 + 카툰 효과")
        .add(FaceDetectorProcessor())
        .add(CartoonProcessor())
    )


# ══════════════════════════════════════════════════════════════════════
# 레지스트리 자동 등록
# ══════════════════════════════════════════════════════════════════════

# (이름, 팩토리, 설명) 목록
_PIPELINE_DEFS = [
    # ── 기본 ──
    ("passthrough",    build_passthrough,    "원본 영상 (No Processing)"),
    # ── 색상 ──
    ("grayscale",      build_grayscale,      "그레이스케일 변환"),
    ("invert",         build_invert,         "색상 반전"),
    ("hsv",            build_hsv,            "HSV 채널 시각화"),
    ("hist_equal",     build_hist_equal,     "히스토그램 평활화 (CLAHE)"),
    ("channel_split",  build_channel_split,  "BGR 채널 분리"),
    ("white_balance",  build_white_balance,  "그레이월드 화이트밸런스"),
    # ── 블러/샤프 ──
    ("gaussian_blur",  build_gaussian_blur,  "가우시안 블러"),
    ("median_blur",    build_median_blur,    "미디안 블러"),
    ("bilateral",      build_bilateral,      "바이래터럴 필터"),
    ("motion_blur",    build_motion_blur,    "모션 블러"),
    ("unsharp_mask",   build_unsharp,        "언샤프 마스킹 (선명화)"),
    ("detail_enhance", build_detail_enhance, "디테일 향상"),
    # ── 엣지 ──
    ("canny",          build_canny,          "Canny 엣지 검출"),
    ("sobel",          build_sobel,          "Sobel 엣지 검출"),
    ("laplacian",      build_laplacian,      "Laplacian 엣지 검출"),
    ("scharr",         build_scharr,         "Scharr 엣지 검출"),
    # ── 형태학 ──
    ("morphology",     build_morphology,     "형태학 연산 (팽창/침식)"),
    ("thresh_morph",   build_thresh_morph,   "이진화 + 형태학"),
    # ── 검출 ──
    ("face_detect",    build_face_detect,    "얼굴 검출 (Haar Cascade)"),
    ("contour_detect", build_contour_detect, "윤곽선 검출"),
    ("hough_line",     build_hough_line,     "허프 직선 검출"),
    ("corner_detect",  build_corner_detect,  "코너 검출 (Shi-Tomasi)"),
    # ── 효과 ──
    ("cartoon",        build_cartoon,        "카툰 효과"),
    ("sketch",         build_sketch,         "연필 스케치"),
    ("emboss",         build_emboss,         "엠보싱 효과"),
    ("pixelate",       build_pixelate,       "픽셀화 효과"),
    ("thermal",        build_thermal,        "열화상 컬러맵"),
    ("oil_paint",      build_oil_paint,      "오일 페인팅"),
    # ── 광류 ──
    ("lk_flow",        build_lk_flow,        "LK 희소 옵티컬 플로우"),
    ("dense_flow",     build_dense_flow,     "Farneback 밀집 옵티컬 플로우"),
    # ── 복합 ──
    ("edge_enhance",   build_edge_enhance,   "엣지 강조 (복합)"),
    ("face_cartoon",   build_face_cartoon,   "얼굴 검출 + 카툰 (복합)"),
]


def register_all(registry: PipelineRegistry) -> None:
    """
    사전 정의된 모든 파이프라인을 레지스트리에 등록합니다.

    Args:
        registry: PipelineRegistry 인스턴스
    """
    for name, factory, desc in _PIPELINE_DEFS:
        registry.register(name, factory, desc)
