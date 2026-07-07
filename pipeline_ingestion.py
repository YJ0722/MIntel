"""
pipeline_ingestion.py
=====================
[1번 방] Ingestion Layer - 영상 수집, Key-frame 추출, 시각/텍스트 가공

역할 (다이어그램 기준 "1. Ingestion Layer (데이터 가공)"):
    - storage/videos/ 폴더(하위 폴더 포함)에 있는 원본 .mp4 영상을 찾아
      modules.ingestion.keyframe_extractor.KeyframeExtractor로 Key-frame을
      추출하고, storage/frames/ 폴더에 저장한다. (OpenCV: 핵심 프레임 동적 추출)
    - 추출된 Key-frame마다 아래 가공을 수행해 벡터/메타데이터를 채운다.
        1. ClipEmbedder    : 시각 특징 벡터 변환
        2. ObjectDetector  : YOLOv8 사물 인식
        3. TextRecognizer  : EasyOCR 글자 인식
    - 이렇게 "가공이 끝난" KeyFrame 목록을 반환하며, 2번 방(pipeline_storage)은
      이 결과를 Qdrant에 저장하기만 하면 된다 (pipeline_storage는 ML 모델을
      전혀 알 필요가 없다).
    - 실제 Key-frame 추출 알고리즘(히스토그램 비교, 최소 프레임 간격 등)은
      modules/ingestion/keyframe_extractor.py 로 분리되어 있다. 이 파일은 여러
      영상을 순회하며 추출기/가공 모듈들을 호출하는 오케스트레이션 역할을 담당한다.
"""

from __future__ import annotations

from pathlib import Path

import config
from modules.ingestion.clip_embedder import ClipEmbedder
from modules.ingestion.keyframe_extractor import (
    DEFAULT_MIN_FRAME_INTERVAL,
    DEFAULT_SIMILARITY_THRESHOLD,
    KeyFrame,
    KeyframeExtractor,
    list_video_files,
)
from modules.ingestion.object_detector import ObjectDetector
from modules.ingestion.text_recognizer import TextRecognizer


def extract_keyframes(
    videos_dir: Path = config.VIDEOS_DIR,
    frames_dir: Path = config.FRAMES_DIR,
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    min_frame_interval: int = DEFAULT_MIN_FRAME_INTERVAL,
) -> list[KeyFrame]:
    """
    storage/videos/ 안의 모든 영상을 순회하며 KeyframeExtractor로 동적
    Key-frame 추출 알고리즘을 적용하고 storage/frames/ 에 저장한 뒤, 저장된
    KeyFrame(가공 전, 메타데이터만 포함) 목록을 반환한다.
    """
    video_files = list_video_files(videos_dir)
    if not video_files:
        print(f"[Ingestion] 처리할 영상이 없습니다. 확인 경로: {videos_dir}")
        return []

    extractor = KeyframeExtractor(
        similarity_threshold=similarity_threshold,
        min_frame_interval=min_frame_interval,
    )

    all_keyframes: list[KeyFrame] = []

    for video_path in video_files:
        print(f"[Ingestion] 영상 처리 시작: {video_path.name}")
        keyframes = extractor.extract_from_video(
            video_path, frames_dir=frames_dir, videos_dir=videos_dir
        )
        all_keyframes.extend(keyframes)
        print(f"[Ingestion] 영상 처리 완료: {video_path.name}")

    return all_keyframes


def enrich_keyframes(
    keyframes: list[KeyFrame],
    object_detector: ObjectDetector | None = None,
    text_recognizer: TextRecognizer | None = None,
    clip_embedder: ClipEmbedder | None = None,
) -> list[KeyFrame]:
    """
    추출된 Key-frame마다 CLIP 임베딩 + YOLOv8 객체 탐지 + EasyOCR 텍스트 인식을
    수행해 KeyFrame.embedding/objects/texts를 채운다 (Ingestion Layer의 "가공" 단계).

    Args:
        keyframes: extract_keyframes()가 반환한, 아직 가공되지 않은 KeyFrame 목록
        object_detector, text_recognizer, clip_embedder: 이미 생성된 인스턴스를
            재사용하고 싶을 때 전달한다. 생략하면 이 함수 안에서 1회 생성한다.
            (모델 로딩 비용이 크므로 여러 영상을 처리할 때 재사용을 권장)

    Returns:
        embedding/objects/texts가 채워진 동일한 KeyFrame 목록 (in-place 수정 후 반환)
    """
    if not keyframes:
        print("[Ingestion] 가공할 Key-frame이 없습니다.")
        return keyframes

    object_detector = object_detector or ObjectDetector()
    text_recognizer = text_recognizer or TextRecognizer()
    clip_embedder = clip_embedder or ClipEmbedder()

    for keyframe in keyframes:
        if keyframe.saved_path is None:
            continue

        frame_path = str(keyframe.saved_path)
        keyframe.embedding = clip_embedder.embed_image(frame_path)
        keyframe.objects = object_detector.detect(frame_path)
        keyframe.texts = text_recognizer.recognize(frame_path)

        print(
            f"[Ingestion] 가공 완료: {keyframe.saved_path.name} "
            f"(objects={keyframe.objects}, texts={keyframe.texts})"
        )

    return keyframes


def run_ingestion(
    videos_dir: Path = config.VIDEOS_DIR,
    frames_dir: Path = config.FRAMES_DIR,
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    min_frame_interval: int = DEFAULT_MIN_FRAME_INTERVAL,
) -> list[KeyFrame]:
    """
    1번 방(Ingestion Layer) 파이프라인의 진입점.

    영상 -> Key-frame 추출(extract_keyframes) -> CLIP/YOLOv8/EasyOCR 가공
    (enrich_keyframes) 까지 마친 뒤, 완전히 가공된 KeyFrame 목록을 반환한다.
    이 결과를 그대로 pipeline_storage.index_keyframes()에 넘기면 Qdrant에
    저장할 수 있다.
    """
    keyframes = extract_keyframes(
        videos_dir=videos_dir,
        frames_dir=frames_dir,
        similarity_threshold=similarity_threshold,
        min_frame_interval=min_frame_interval,
    )
    return enrich_keyframes(keyframes)


if __name__ == "__main__":
    run_ingestion()
