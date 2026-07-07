"""
modules/ingestion/object_detector.py
=====================================
YOLOv8 기반 사물 인식(Object Detection) 모듈.

역할:
    - storage/frames/ 에 저장된 Key-frame 이미지 한 장을 입력받아, 그 안에 어떤
      사물(자동차, 사람, 신호등 등)이 있는지 탐지하고 클래스 이름 목록을 반환한다.
    - 추후 pipeline_storage.py에서 Qdrant에 프레임을 색인할 때, 이 클래스 이름들을
      payload(메타데이터)로 함께 저장해 검색/필터링에 활용할 수 있다.

Mac M3 Pro 환경:
    - Apple Silicon의 MPS(Metal Performance Shaders) 백엔드를 우선 사용하고,
      사용 불가능한 환경(GPU 미지원, torch 빌드가 MPS를 지원하지 않는 경우 등)에는
      자동으로 CPU로 폴백한다.
"""

from __future__ import annotations

from pathlib import Path

import torch
from ultralytics import YOLO

# 속도를 우선한 가장 가벼운 YOLOv8 모델 (최초 실행 시 ultralytics가 자동으로 다운로드함)
DEFAULT_MODEL_NAME = "yolov8n.pt"
DEFAULT_CONF_THRESHOLD = 0.25


class ObjectDetector:
    """YOLOv8 기반 사물 인식기.

    사용 예:
        detector = ObjectDetector()
        classes = detector.detect("storage/frames/sample_frame_000000.jpg")
        # -> ['car', 'person', 'traffic light']
    """

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_NAME,
        device: str | None = None,
    ) -> None:
        """
        Args:
            model_path: 로드할 YOLO 가중치 경로/이름 (기본값: yolov8n.pt)
            device: 추론에 사용할 디바이스. None이면 MPS -> CPU 순으로 자동 선택한다.
        """
        self.device = device or self._select_device()
        self.model = YOLO(model_path)
        self.model.to(self.device)
        print(
            f"[ObjectDetector] YOLO 모델 '{model_path}' 로드 완료 (device={self.device})"
        )

    @staticmethod
    def _select_device() -> str:
        """Mac M3 Pro의 MPS 백엔드를 우선 사용하고, 사용 불가능하면 CPU로 폴백한다."""
        try:
            if torch.backends.mps.is_available():
                return "mps"
        except AttributeError:
            # 구버전 torch 등 backends.mps 자체가 없는 환경에 대한 방어 코드
            pass
        return "cpu"

    def detect(
        self,
        frame_path: str,
        conf_threshold: float = DEFAULT_CONF_THRESHOLD,
    ) -> list:
        """
        이미지 한 장에서 사물을 탐지하여, 중복이 제거된 클래스 이름 리스트를 반환한다.

        Args:
            frame_path: 탐지할 이미지 파일 경로
            conf_threshold: 이 값 미만의 confidence를 가진 탐지 결과는 무시한다.

        Returns:
            탐지된 사물의 클래스 이름 목록 (예: ['car', 'person', 'traffic light']).
            중복은 제거되어 있으며, 아무것도 탐지되지 않으면 빈 리스트를 반환한다.
        """
        if not Path(frame_path).exists():
            raise FileNotFoundError(f"이미지 파일을 찾을 수 없습니다: {frame_path}")

        results = self.model.predict(
            source=str(frame_path),
            conf=conf_threshold,
            device=self.device,
            verbose=False,
        )

        detected_classes = [
            self.model.names[int(box.cls)]
            for result in results
            for box in result.boxes
        ]

        return list(set(detected_classes))


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("사용법: python object_detector.py <이미지_경로>")
    else:
        detector = ObjectDetector()
        found_classes = detector.detect(sys.argv[1])
        print(f"탐지된 사물: {found_classes}")
