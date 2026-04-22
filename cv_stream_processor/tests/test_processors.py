"""
프로세서 단위 테스트 (tests/test_processors.py)
GUI 없이 모든 프로세서의 동작을 검증합니다.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

import numpy as np
import pytest

from cv_stream_processor.core.frame import Frame
from cv_stream_processor.processors import (
    PassthroughProcessor,
    GrayscaleProcessor,
    InvertProcessor,
    HSVProcessor,
    HistEqualProcessor,
    CannyProcessor,
    SobelProcessor,
    LaplacianProcessor,
    GaussianBlurProcessor,
    MedianBlurProcessor,
    BilateralProcessor,
    UnsharpMaskProcessor,
    MorphologyProcessor,
    ContourDetectorProcessor,
    CartoonProcessor,
    SketchProcessor,
    EmbossProcessor,
    PixelateProcessor,
    ThermalProcessor,
)


# ── 픽스처 ───────────────────────────────────────────────────────────

@pytest.fixture
def sample_frame():
    """480x640 BGR 테스트 프레임."""
    img = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
    return Frame(image=img, frame_id=1)


@pytest.fixture
def black_frame():
    """전체 검정 프레임."""
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    return Frame(image=img, frame_id=2)


@pytest.fixture
def white_frame():
    """전체 흰색 프레임."""
    img = np.full((480, 640, 3), 255, dtype=np.uint8)
    return Frame(image=img, frame_id=3)


# ── 헬퍼 ─────────────────────────────────────────────────────────────

def run_processor(processor_cls, frame: Frame, **params) -> Frame:
    """프로세서를 초기화하고 프레임을 처리한 결과를 반환합니다."""
    proc = processor_cls()
    for k, v in params.items():
        proc.set_param(k, v)
    proc.initialize()
    result = proc.process(frame)
    proc.release()
    return result


def assert_valid_output(frame: Frame) -> None:
    """처리 결과가 유효한 BGR 이미지인지 확인합니다."""
    assert frame.processed is not None, "processed 이미지가 None입니다."
    assert frame.processed.dtype == np.uint8, "dtype이 uint8이 아닙니다."
    assert frame.processed.ndim == 3, "채널이 3이 아닙니다."
    assert frame.processed.shape[2] == 3, "BGR 채널 수가 3이 아닙니다."
    assert frame.processed.shape[:2] == frame.image.shape[:2], "해상도가 변경되었습니다."


# ══════════════════════════════════════════════════════════════════════
# 기본 프로세서 테스트
# ══════════════════════════════════════════════════════════════════════

class TestPassthrough:
    def test_output_equals_input(self, sample_frame):
        result = run_processor(PassthroughProcessor, sample_frame)
        assert_valid_output(result)
        np.testing.assert_array_equal(result.processed, sample_frame.image)


class TestGrayscale:
    def test_output_is_gray_converted_to_bgr(self, sample_frame):
        result = run_processor(GrayscaleProcessor, sample_frame)
        assert_valid_output(result)
        # 그레이→BGR 변환 시 B=G=R 확인
        b, g, r = result.processed[:, :, 0], result.processed[:, :, 1], result.processed[:, :, 2]
        np.testing.assert_array_equal(b, g)
        np.testing.assert_array_equal(g, r)

    def test_black_frame(self, black_frame):
        result = run_processor(GrayscaleProcessor, black_frame)
        assert_valid_output(result)
        np.testing.assert_array_equal(result.processed, 0)


class TestInvert:
    def test_white_becomes_black(self, white_frame):
        result = run_processor(InvertProcessor, white_frame)
        assert_valid_output(result)
        np.testing.assert_array_equal(result.processed, 0)

    def test_black_becomes_white(self, black_frame):
        result = run_processor(InvertProcessor, black_frame)
        assert_valid_output(result)
        np.testing.assert_array_equal(result.processed, 255)

    def test_double_invert_equals_original(self, sample_frame):
        proc = InvertProcessor()
        r1 = proc.process(sample_frame)
        sample_frame.processed = r1.processed
        r2 = proc.process(sample_frame)
        np.testing.assert_array_equal(r2.processed, sample_frame.image)


# ══════════════════════════════════════════════════════════════════════
# 엣지 검출 테스트
# ══════════════════════════════════════════════════════════════════════

class TestCanny:
    def test_output_valid(self, sample_frame):
        result = run_processor(CannyProcessor, sample_frame)
        assert_valid_output(result)

    def test_black_frame_all_zero(self, black_frame):
        result = run_processor(CannyProcessor, black_frame)
        assert_valid_output(result)
        np.testing.assert_array_equal(result.processed, 0)


class TestSobel:
    def test_output_valid(self, sample_frame):
        result = run_processor(SobelProcessor, sample_frame)
        assert_valid_output(result)

    def test_direction_x(self, sample_frame):
        result = run_processor(SobelProcessor, sample_frame, direction="x")
        assert_valid_output(result)

    def test_direction_y(self, sample_frame):
        result = run_processor(SobelProcessor, sample_frame, direction="y")
        assert_valid_output(result)


class TestLaplacian:
    def test_output_valid(self, sample_frame):
        result = run_processor(LaplacianProcessor, sample_frame)
        assert_valid_output(result)


# ══════════════════════════════════════════════════════════════════════
# 블러/샤프닝 테스트
# ══════════════════════════════════════════════════════════════════════

class TestGaussianBlur:
    def test_output_valid(self, sample_frame):
        result = run_processor(GaussianBlurProcessor, sample_frame)
        assert_valid_output(result)

    def test_black_stays_black(self, black_frame):
        result = run_processor(GaussianBlurProcessor, black_frame)
        np.testing.assert_array_equal(result.processed, 0)


class TestMedianBlur:
    def test_output_valid(self, sample_frame):
        result = run_processor(MedianBlurProcessor, sample_frame)
        assert_valid_output(result)


class TestBilateral:
    def test_output_valid(self, sample_frame):
        result = run_processor(BilateralProcessor, sample_frame)
        assert_valid_output(result)


class TestUnsharpMask:
    def test_output_valid(self, sample_frame):
        result = run_processor(UnsharpMaskProcessor, sample_frame)
        assert_valid_output(result)


# ══════════════════════════════════════════════════════════════════════
# 형태학 / 검출 테스트
# ══════════════════════════════════════════════════════════════════════

class TestMorphology:
    def test_dilate(self, sample_frame):
        result = run_processor(MorphologyProcessor, sample_frame, operation="dilate")
        assert_valid_output(result)

    def test_erode(self, sample_frame):
        result = run_processor(MorphologyProcessor, sample_frame, operation="erode")
        assert_valid_output(result)

    def test_open(self, sample_frame):
        result = run_processor(MorphologyProcessor, sample_frame, operation="open")
        assert_valid_output(result)

    def test_close(self, sample_frame):
        result = run_processor(MorphologyProcessor, sample_frame, operation="close")
        assert_valid_output(result)


class TestContourDetector:
    def test_output_valid(self, sample_frame):
        result = run_processor(ContourDetectorProcessor, sample_frame)
        assert_valid_output(result)
        assert "contour_count" in result.metadata


# ══════════════════════════════════════════════════════════════════════
# 효과 테스트
# ══════════════════════════════════════════════════════════════════════

class TestEffects:
    def test_cartoon_output_valid(self, sample_frame):
        result = run_processor(CartoonProcessor, sample_frame)
        assert_valid_output(result)

    def test_sketch_gray(self, sample_frame):
        result = run_processor(SketchProcessor, sample_frame)
        assert_valid_output(result)

    def test_emboss_output_valid(self, sample_frame):
        result = run_processor(EmbossProcessor, sample_frame)
        assert_valid_output(result)

    def test_pixelate_output_valid(self, sample_frame):
        result = run_processor(PixelateProcessor, sample_frame)
        assert_valid_output(result)

    def test_thermal_output_valid(self, sample_frame):
        result = run_processor(ThermalProcessor, sample_frame)
        assert_valid_output(result)


# ══════════════════════════════════════════════════════════════════════
# 파이프라인 체인 테스트
# ══════════════════════════════════════════════════════════════════════

class TestPipelineChain:
    def test_grayscale_then_canny(self, sample_frame):
        """그레이스케일 → Canny 체인 테스트."""
        from cv_stream_processor.core.pipeline import Pipeline
        pipeline = (
            Pipeline("test_chain", "테스트 체인")
            .add(GrayscaleProcessor())
            .add(CannyProcessor())
        )
        pipeline.initialize()
        result = pipeline.run(sample_frame)
        pipeline.release()
        assert_valid_output(result)

    def test_pipeline_registry(self, sample_frame):
        """PipelineRegistry 등록/선택/실행 테스트."""
        from cv_stream_processor.core.pipeline_registry import PipelineRegistry
        from cv_stream_processor.core.pipeline_factory import register_all

        registry = PipelineRegistry()
        register_all(registry)
        assert len(registry.pipeline_names) > 0

        assert registry.select("grayscale")
        result = registry.run(sample_frame)
        assert_valid_output(result)

        name = registry.next_pipeline()
        assert name != "grayscale"

        registry.release_all()


# ── 직접 실행 ────────────────────────────────────────────────────────

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
