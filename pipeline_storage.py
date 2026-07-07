"""
pipeline_storage.py
====================
[2번 방] Storage Layer - 벡터 DB 저장

역할 (다이어그램 기준 "2. Storage Layer (데이터 저장)"):
    - 1번 방(pipeline_ingestion)에서 이미 CLIP 임베딩 + YOLOv8 객체 탐지 +
      EasyOCR 텍스트 인식까지 "가공"이 끝난 KeyFrame들을 넘겨받아, 그 벡터와
      메타데이터를 로컬 Qdrant DB(storage/db/)에 그대로 저장(보관)한다.
    - 이 파일은 의도적으로 ML 모델(CLIP/YOLOv8/EasyOCR)을 전혀 알지 못한다.
      "가공"은 전부 Ingestion Layer(pipeline_ingestion.py)의 책임이고, 이
      Storage Layer는 순수하게 Qdrant 연결/컬렉션 관리/저장만 담당한다.

메모리 관리 (M3 Pro 18GB 환경):
    - Qdrant를 별도 서버 프로세스로 띄우지 않고, path 옵션을 사용해
      임베딩 DB 자체를 로컬 파일(storage/db/) 기반으로 구동한다.
      (Docker 불필요, 별도 메모리 상주 서버 불필요)
    - 포인트(벡터)를 대량으로 한 번에 올리지 않고, batch 단위로
      upsert 하도록 인터페이스를 설계해 메모리 사용량을 제어한다.
"""

from __future__ import annotations

import uuid
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

import config
from modules.ingestion.keyframe_extractor import KeyFrame

# ------------------------------------------------------------------
# 기본 설정 (DB 경로는 config.py에서 가져온다)
# ------------------------------------------------------------------
DEFAULT_COLLECTION_NAME = "video_keyframes"

# CLIP류 이미지 임베딩 모델을 염두에 둔 기본 벡터 차원 (추후 사용 모델에 맞게 조정)
DEFAULT_VECTOR_SIZE = 512
DEFAULT_DISTANCE = qmodels.Distance.COSINE


class QdrantStorage:
    """
    로컬 Qdrant DB 연결 및 컬렉션 관리를 담당하는 클래스.

    사용 예:
        storage = QdrantStorage()
        storage.create_collection()
        storage.upsert_frame(point_id=1, vector=[...], payload={"video": "a.mp4"})
    """

    def __init__(
        self,
        db_dir: Path = config.DB_DIR,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        vector_size: int = DEFAULT_VECTOR_SIZE,
        distance: qmodels.Distance = DEFAULT_DISTANCE,
    ) -> None:
        self.db_dir = db_dir
        self.collection_name = collection_name
        self.vector_size = vector_size
        self.distance = distance

        self.db_dir.mkdir(parents=True, exist_ok=True)

        # 로컬 파일 기반 모드 (별도 서버 프로세스 없이 storage/db/ 에 직접 저장)
        self.client = QdrantClient(path=str(self.db_dir))

    def create_collection(self, recreate: bool = False) -> None:
        """
        컬렉션이 없으면 새로 생성한다.

        Args:
            recreate: True일 경우 기존 컬렉션을 삭제하고 새로 생성한다.
        """
        exists = self.client.collection_exists(self.collection_name)

        if exists and not recreate:
            print(f"[storage] 컬렉션 '{self.collection_name}' 이미 존재함. 재사용합니다.")
            return

        if exists and recreate:
            print(f"[storage] 기존 컬렉션 '{self.collection_name}' 삭제 후 재생성합니다.")
            self.client.delete_collection(self.collection_name)

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=qmodels.VectorParams(
                size=self.vector_size,
                distance=self.distance,
            ),
        )
        print(
            f"[storage] 컬렉션 '{self.collection_name}' 생성 완료 "
            f"(dim={self.vector_size}, distance={self.distance})"
        )

    def upsert_frame(
        self,
        point_id: int | str,
        vector: list[float],
        payload: dict | None = None,
    ) -> None:
        """
        Key-frame 하나에 대한 임베딩 벡터와 메타데이터(payload)를 저장한다.

        Args:
            point_id: 포인트 고유 ID (프레임 파일명 해시나 순번 등)
            vector: 임베딩 벡터 (예: CLIP 이미지 임베딩)
            payload: 프레임 메타데이터 (예: 영상명, 타임스탬프, OCR 텍스트 등)
        """
        self.client.upsert(
            collection_name=self.collection_name,
            points=[
                qmodels.PointStruct(
                    id=point_id,
                    vector=vector,
                    payload=payload or {},
                )
            ],
        )

    def upsert_frames_batch(self, points: list[qmodels.PointStruct]) -> None:
        """여러 개의 포인트를 batch 단위로 한 번에 저장한다 (메모리 효율을 위한 배치 처리)."""
        if not points:
            return
        self.client.upsert(collection_name=self.collection_name, points=points)

    def search(self, query_vector: list[float], top_k: int = 5):
        """쿼리 벡터와 가장 유사한 Key-frame들을 검색한다."""
        return self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=top_k,
        )

    def count(self) -> int:
        """현재 컬렉션에 저장된 포인트 개수를 반환한다."""
        result = self.client.count(collection_name=self.collection_name, exact=True)
        return result.count


def run_storage_setup(db_dir: Path = config.DB_DIR) -> QdrantStorage:
    """
    2번 방 파이프라인의 진입점.

    QdrantStorage 인스턴스를 생성하고 컬렉션을 준비한 뒤 반환한다.
    """
    storage = QdrantStorage(db_dir=db_dir)
    storage.create_collection()
    print(f"[storage] 현재 저장된 포인트 수: {storage.count()}")
    return storage


def _make_point_id(frame_path: str) -> str:
    """
    프레임 파일 경로로부터 결정적(deterministic)인 UUID를 생성한다.

    같은 프레임을 다시 색인하면 항상 같은 point_id가 나오므로, 파이프라인을
    여러 번 재실행해도 포인트가 무한정 늘어나지 않고 덮어써진다(idempotent).
    """
    return str(uuid.uuid5(uuid.NAMESPACE_URL, frame_path))


def index_keyframes(storage: QdrantStorage, keyframes: list[KeyFrame]) -> int:
    """
    이미 가공(임베딩/객체 탐지/텍스트 인식)이 끝난 KeyFrame 목록을 받아 Qdrant에
    저장한다. CLIP/YOLOv8/EasyOCR 추론은 전혀 수행하지 않으며, 각 KeyFrame의
    embedding/objects/texts 필드를 그대로 읽어 (벡터, payload) 형태로
    batch upsert 하기만 한다.

    Args:
        storage: 저장 대상 QdrantStorage 인스턴스 (run_storage_setup()으로 준비)
        keyframes: 1번 방(pipeline_ingestion.run_ingestion())에서 가공까지 끝난
            KeyFrame 목록 (embedding이 채워져 있어야 한다)

    Returns:
        실제로 저장된 프레임 수
    """
    if not keyframes:
        print("[storage] 저장할 Key-frame이 없습니다.")
        return 0

    points: list[qmodels.PointStruct] = []

    for keyframe in keyframes:
        if keyframe.saved_path is None or keyframe.embedding is None:
            print(
                f"[storage] 가공되지 않은 Key-frame은 건너뜁니다: "
                f"{keyframe.saved_path}"
            )
            continue

        frame_path = str(keyframe.saved_path)
        payload = {
            "video_name": keyframe.video_name,
            "frame_index": keyframe.frame_index,
            "timestamp_sec": keyframe.timestamp_sec,
            "frame_path": frame_path,
            "objects": keyframe.objects or [],
            "texts": keyframe.texts or [],
        }

        points.append(
            qmodels.PointStruct(
                id=_make_point_id(frame_path),
                vector=keyframe.embedding,
                payload=payload,
            )
        )

    storage.upsert_frames_batch(points)
    print(f"[storage] 총 {len(points)}개의 Key-frame을 Qdrant에 저장 완료")

    return len(points)


if __name__ == "__main__":
    run_storage_setup()
