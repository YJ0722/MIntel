"""
pipeline_storage.py
====================
[2번 방] Storage Layer - 벡터 DB 적재 제어 타워

역할 (다이어그램 기준 "2. Storage Layer (데이터 저장)"):
    - 1번 방(pipeline_ingestion)에서 CLIP 임베딩 + YOLOv8 객체 탐지 + EasyOCR
      텍스트 인식까지 "가공"이 끝난 KeyFrame 목록(ingested_data)을 넘겨받는다.
    - 각 KeyFrame을 Qdrant가 이해하는 [ID + Vector + Payload] 딕셔너리로
      변환한 뒤, modules/storage/qdrant_client.QdrantStorageManager에게
      실제 DB 적재를 위임한다.
    - 이 파일은 "제어 타워(오케스트레이터)" 역할만 담당하고, Qdrant와의 실제
      통신(연결/컬렉션 관리/upsert)은 전부 modules/storage/qdrant_client.py에
      위임한다. Ingestion Layer가 modules/ingestion/*을 오케스트레이션 하는
      것과 동일한 구조다.

메모리 관리 (M3 Pro 18GB 환경):
    - Qdrant를 별도 서버 프로세스로 띄우지 않고, 로컬 파일(storage/db/) 기반으로
      구동한다 (Docker 불필요, 별도 메모리 상주 서버 불필요).
"""

from __future__ import annotations

import uuid
from pathlib import Path

import config
from modules.ingestion.keyframe_extractor import KeyFrame
from modules.storage.qdrant_client import QdrantStorageManager


def _make_point_id(frame_path: str) -> str:
    """
    프레임 파일 경로로부터 결정적(deterministic)인 UUID를 생성한다.

    같은 프레임을 다시 적재해도 항상 같은 point_id가 나오므로, 파이프라인을
    여러 번 재실행해도 포인트가 무한정 늘어나지 않고 덮어써진다(idempotent).
    """
    return str(uuid.uuid5(uuid.NAMESPACE_URL, frame_path))


def _keyframe_to_point(keyframe: KeyFrame) -> dict | None:
    """
    가공이 끝난 KeyFrame 하나를 QdrantStorageManager.upsert_data()가 이해하는
    {"id", "vector", "payload"} 딕셔너리("패키지")로 변환한다.

    saved_path 또는 embedding이 비어 있으면(=아직 가공되지 않은 프레임) None을
    반환해 상위 함수가 건너뛸 수 있도록 한다.
    """
    if keyframe.saved_path is None or keyframe.embedding is None:
        print(f"[Storage] 가공되지 않은 Key-frame은 건너뜁니다: {keyframe.saved_path}")
        return None

    frame_path = str(keyframe.saved_path)
    return {
        "id": _make_point_id(frame_path),
        "vector": keyframe.embedding,
        "payload": {
            "video_name": keyframe.video_name,
            "frame_index": keyframe.frame_index,
            "timestamp_sec": keyframe.timestamp_sec,
            "frame_path": frame_path,
            "objects": keyframe.objects or [],
            "texts": keyframe.texts or [],
        },
    }


def run_storage_pipeline(
    ingested_data: list[KeyFrame],
    db_path: Path = config.DB_DIR,
) -> int:
    """
    2번 방(Storage Layer) 파이프라인의 진입점.

    1번 방(pipeline_ingestion.run_ingestion())에서 가공까지 끝낸 KeyFrame
    목록(ingested_data)을 받아 [ID + Vector + Payload] 포인트로 변환하고,
    QdrantStorageManager를 통해 로컬 Qdrant DB에 최종 적재한다.

    Args:
        ingested_data: pipeline_ingestion에서 넘어온, CLIP/YOLOv8/EasyOCR
            가공까지 끝난 KeyFrame 리스트
        db_path: Qdrant 로컬 DB가 저장될 폴더 경로 (기본값: config.DB_DIR)

    Returns:
        실제로 Qdrant DB에 적재된 프레임 수 (실패 시 0)
    """
    if not ingested_data:
        print("[Storage] 적재할 데이터가 없습니다.")
        return 0

    try:
        manager = QdrantStorageManager(db_path=db_path)
    except Exception as exc:
        print(f"[Storage] Qdrant 연결/초기화에 실패해 적재를 중단합니다: {exc}")
        return 0

    points: list[dict] = []
    for keyframe in ingested_data:
        point = _keyframe_to_point(keyframe)
        if point is not None:
            points.append(point)

    if not points:
        print("[Storage] 유효하게 가공된 Key-frame이 없어 적재를 건너뜁니다.")
        return 0

    try:
        saved_count = manager.upsert_data(points)
    except Exception as exc:
        print(f"[Storage] DB 적재 중 예기치 못한 오류가 발생했습니다: {exc}")
        return 0

    print(f"[Storage] 성공적으로 {saved_count}개의 프레임 데이터를 Qdrant DB에 적재했습니다.")
    return saved_count


if __name__ == "__main__":
    manager = QdrantStorageManager(db_path=config.DB_DIR)
    print(f"[Storage] 현재 저장된 포인트 수: {manager.count()}")
