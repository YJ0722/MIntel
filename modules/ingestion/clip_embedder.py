"""
modules/ingestion/clip_embedder.py
====================================
CLIP(Contrastive Language-Image Pretraining) 기반 이미지 임베딩 모듈.

역할:
    - storage/frames/ 에 저장된 Key-frame 이미지 한 장을 입력받아, CLIP 이미지
      인코더를 통과시켜 고차원 시각 특징 벡터(임베딩)를 추출한다.
    - 이 벡터는 pipeline_storage.py에서 Qdrant(video_keyframes 컬렉션, dim=512)에
      그대로 색인되어, 이후 텍스트/이미지 쿼리로 유사한 장면을 검색하는 데 쓰인다.

Mac M3 Pro 환경:
    - Apple Silicon의 MPS(Metal Performance Shaders) 백엔드를 우선 사용하고,
      사용 불가능한 환경에서는 자동으로 CPU로 폴백한다.
    - openai/clip-vit-base-patch32는 이미지 임베딩 차원이 512로, pipeline_storage.py의
      DEFAULT_VECTOR_SIZE(512)와 그대로 맞아떨어진다.
"""

from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

DEFAULT_MODEL_NAME = "openai/clip-vit-base-patch32"


class ClipEmbedder:
    """CLIP 기반 이미지 임베딩 추출기.

    사용 예:
        embedder = ClipEmbedder()
        vector = embedder.embed_image("storage/frames/sample_frame_000000.jpg")
        # -> [0.0123, -0.0456, ...] (길이 512의 float 리스트)
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        device: str | None = None,
    ) -> None:
        """
        Args:
            model_name: 로드할 Hugging Face CLIP 모델 이름 (기본값: openai/clip-vit-base-patch32)
            device: 추론에 사용할 디바이스. None이면 MPS -> CPU 순으로 자동 선택한다.
        """
        self.device = device or self._select_device()
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        print(
            f"[ClipEmbedder] CLIP 모델 '{model_name}' 로드 완료 (device={self.device})"
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

    def embed_image(self, frame_path: str) -> list:
        """
        이미지 한 장을 CLIP 이미지 인코더에 통과시켜 특징 벡터를 추출한다.

        Args:
            frame_path: 임베딩을 추출할 이미지 파일 경로

        Returns:
            Qdrant 등 벡터 DB에 바로 색인할 수 있는 순수 파이썬 float 리스트
            (openai/clip-vit-base-patch32 기준 길이 512).
        """
        if not Path(frame_path).exists():
            raise FileNotFoundError(f"이미지 파일을 찾을 수 없습니다: {frame_path}")

        image = Image.open(frame_path).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)

        with torch.no_grad():
            output = self.model.get_image_features(**inputs)

        # transformers 최신 버전은 get_image_features()가 (투영 전 vision 인코더의)
        # BaseModelOutputWithPooling 객체를 반환하며, 실제 투영된 이미지 임베딩은
        # pooler_output에 담겨 있다. 구버전은 (1, 512) 텐서를 직접 반환하므로 두 경우를
        # 모두 처리해 버전에 상관없이 동작하도록 한다.
        image_features = getattr(output, "pooler_output", output)

        # (1, 512) 텐서 -> CPU로 이동 -> numpy -> 순수 파이썬 float 리스트
        embedding = image_features[0].cpu().numpy().tolist()
        return embedding


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("사용법: python clip_embedder.py <이미지_경로>")
    else:
        embedder = ClipEmbedder()
        vector = embedder.embed_image(sys.argv[1])
        print(f"임베딩 차원: {len(vector)}")
        print(f"앞 5개 값: {vector[:5]}")
