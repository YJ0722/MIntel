"""
modules/ingestion/text_recognizer.py
======================================
EasyOCR 기반 문자 인식(OCR) 모듈.

역할:
    - storage/frames/ 에 저장된 Key-frame 이미지 한 장을 입력받아, 그 안에 있는
      표지판/간판/자막 등의 텍스트를 인식하여 문자열 목록으로 반환한다.
    - 추후 pipeline_storage.py에서 Qdrant에 프레임을 색인할 때, 이 텍스트들을
      payload(메타데이터)로 함께 저장해 검색/필터링에 활용할 수 있다.

Mac M3 Pro 환경:
    - easyocr(1.7.x)는 gpu=True일 때 내부적으로 CUDA -> MPS(Apple Silicon) -> CPU
      순으로 자동 감지하여 사용하므로, 별도 설정 없이 M3 Pro의 MPS 가속이 적용된다.
"""

from __future__ import annotations

from pathlib import Path

import easyocr

# 영어 + 한국어 동시 인식 (표지판 텍스트, 자막 등 대응)
DEFAULT_LANGUAGES = ["en", "ko"]
DEFAULT_CONF_THRESHOLD = 0.4


class TextRecognizer:
    """EasyOCR 기반 문자 인식기.

    사용 예:
        recognizer = TextRecognizer()
        texts = recognizer.recognize("storage/frames/sample_frame_000000.jpg")
        # -> ['STOP', '신호등']
    """

    def __init__(
        self,
        languages: list[str] | None = None,
        gpu: bool = True,
    ) -> None:
        """
        Args:
            languages: 인식할 언어 목록 (기본값: 영어 + 한국어)
            gpu: True면 easyocr이 CUDA -> MPS(Apple Silicon) -> CPU 순으로 자동 감지해 사용한다.
        """
        self.languages = languages or DEFAULT_LANGUAGES
        self.reader = easyocr.Reader(self.languages, gpu=gpu)
        print(
            f"[TextRecognizer] EasyOCR Reader 로드 완료 "
            f"(languages={self.languages}, device={self.reader.device})"
        )

    def recognize(
        self,
        frame_path: str,
        conf_threshold: float = DEFAULT_CONF_THRESHOLD,
    ) -> list:
        """
        이미지 한 장에서 텍스트를 인식하여, 신뢰도가 낮은 결과를 걸러낸 문자열 리스트를 반환한다.

        Args:
            frame_path: 인식할 이미지 파일 경로
            conf_threshold: 이 값 미만의 confidence를 가진 인식 결과는 무시한다.

        Returns:
            인식된 텍스트 문자열 목록 (예: ['STOP', '신호등']).
            아무 텍스트도 인식되지 않으면 빈 리스트를 반환한다.
        """
        if not Path(frame_path).exists():
            raise FileNotFoundError(f"이미지 파일을 찾을 수 없습니다: {frame_path}")

        # readtext()는 (bbox, text, confidence) 튜플의 리스트를 반환한다.
        results = self.reader.readtext(str(frame_path))

        recognized_texts = [
            text.strip()
            for _bbox, text, confidence in results
            if confidence >= conf_threshold and text.strip()
        ]

        return recognized_texts


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("사용법: python text_recognizer.py <이미지_경로>")
    else:
        recognizer = TextRecognizer()
        found_texts = recognizer.recognize(sys.argv[1])
        print(f"인식된 텍스트: {found_texts}")
