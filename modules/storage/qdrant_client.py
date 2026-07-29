"""
modules/storage/qdrant_client.py
==================================
로컬 파일 기반 Qdrant 벡터 DB 연결 / 컬렉션 관리 / 저장을 담당하는 저수준(low-level) 모듈.

역할:
    - Docker로 별도 Qdrant 서버를 띄우지 않고, QdrantClient(path=...)를 사용해
      디스크 파일 기반으로 벡터 DB를 가볍게 구동한다.
    - "mintel_collection" 컬렉션이 없으면 새로 생성하고, [ID + Vector + Payload]
      형태의 포인트 리스트를 Qdrant의 PointStruct로 변환해 저장(upsert)한다.
    - 이 모듈은 어떤 임베딩 모델(CLIP 등)을 쓰는지, 데이터가 어느 파이프라인에서
      왔는지 전혀 알지 못한다. KeyFrame -> 포인트 변환은 pipeline_storage.py가
      책임지고, 이 모듈은 순수하게 Qdrant와의 통신(연결/컬렉션/upsert/검색)만
      담당한다.

DB 경로:
    - DB 저장 경로는 이 파일에서 자체적으로 정하지 않고, 항상 프로젝트 루트의
      config.py(config.DB_DIR)에서 가져온다. 경로를 바꾸고 싶으면 config.py만
      수정하면 된다.

주의 (직접 실행 금지):
    - 이 파일을 `python modules/storage/qdrant_client.py`처럼 직접 실행하면
      안 된다. 파일명이 실제 의존 패키지인 qdrant_client와 같아서, 직접
      실행 시 sys.path 우선순위 때문에 `from qdrant_client import QdrantClient`가
      진짜 패키지 대신 이 파일 자신을 가져오려 해 ImportError가 난다.
      (pipeline_storage.py나 app.py처럼 프로젝트 루트의 스크립트를 통해
      import해서 쓰면 이 문제가 발생하지 않는다.)

Mac M3 Pro 환경:
    - 로컬 파일 기반 모드라 별도 서버 프로세스나 상주 메모리가 필요 없어
      가볍게 개발/실행할 수 있다.
"""

from __future__ import annotations

from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

import config

DEFAULT_COLLECTION_NAME = "mintel_collection"

# CLIP(openai/clip-vit-base-patch32) 기준 512차원.
# ViT-L/14 등 768차원 계열 모델로 바꾸면 이 값도 함께 맞춰줘야 한다.
DEFAULT_VECTOR_SIZE = 512
DEFAULT_DISTANCE = qmodels.Distance.COSINE


class QdrantStorageManager:
    """
    로컬 Qdrant DB 연결, 컬렉션 관리, 포인트 저장/검색을 담당하는 매니저 클래스.

    사용 예:
        manager = QdrantStorageManager()
        saved_count = manager.upsert_data([
            {"id": "uuid-...", "vector": [0.1, 0.2, ...], "payload": {"video_name": "a"}},
        ])
    """

    def __init__(
        self,
        db_path: str | Path = config.DB_DIR,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        vector_size: int = DEFAULT_VECTOR_SIZE,
        distance: qmodels.Distance = DEFAULT_DISTANCE,
    ) -> None:
        """
        Args:
            db_path: Qdrant 로컬 DB 파일이 저장될 폴더 경로 (기본값: config.DB_DIR)
            collection_name: 사용할 컬렉션 이름 (기본값: mintel_collection)
            vector_size: 벡터 차원 (CLIP 기본 512, ViT-L/14 계열은 768)
            distance: 벡터 유사도 계산 방식 (기본값: 코사인 유사도)

        Raises:
            Exception: DB 연결 또는 컬렉션 준비에 실패하면 그대로 전파한다
                (호출 측에서 파이프라인 중단 여부를 판단할 수 있도록).
        """
        self.db_path = Path(db_path)
        self.collection_name = collection_name
        self.vector_size = vector_size
        self.distance = distance

        try:
            self.db_path.mkdir(parents=True, exist_ok=True)
            # 로컬 파일 기반 모드: 별도 서버 프로세스 없이 db_path에 직접 저장한다.
            self.client = QdrantClient(path=str(self.db_path))
        except Exception as exc:
            print(f"[QdrantStorageManager] Qdrant 연결 실패 (path={self.db_path}): {exc}")
            raise

        self._ensure_collection()

    def _ensure_collection(self) -> None:
        """컬렉션이 없으면 새로 생성하고, 있으면 그대로 재사용한다."""
        try:
            exists = self.client.collection_exists(self.collection_name)
        except Exception as exc:
            print(f"[QdrantStorageManager] 컬렉션 존재 여부 확인 실패: {exc}")
            raise

        if exists:
            print(
                f"[QdrantStorageManager] 컬렉션 '{self.collection_name}' 이미 존재함. "
                "재사용합니다."
            )
            return

        try:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=qmodels.VectorParams(
                    size=self.vector_size,
                    distance=self.distance,
                ),
            )
            print(
                f"[QdrantStorageManager] 컬렉션 '{self.collection_name}' 생성 완료 "
                f"(dim={self.vector_size}, distance={self.distance})"
            )
        except Exception as exc:
            print(f"[QdrantStorageManager] 컬렉션 생성 실패: {exc}")
            raise

    def upsert_data(self, points: list[dict]) -> int:
        """
        [ID + Vector + Payload] 딕셔너리 리스트를 Qdrant PointStruct로 변환해 저장한다.

        Args:
            points: 각 원소가 다음 형태인 리스트
                {"id": <int|str>, "vector": [float, ...], "payload": {...}}
                형식이 잘못된 원소(필수 키 누락, 벡터 차원 불일치 등)는 로그를
                남기고 건너뛰며, 나머지 유효한 포인트는 정상적으로 저장한다.

        Returns:
            실제로 저장에 성공한 포인트 수 (전부 실패하거나 입력이 비어 있으면 0)
        """
        if not points:
            print("[QdrantStorageManager] 저장할 포인트가 없습니다.")
            return 0

        point_structs: list[qmodels.PointStruct] = []

        for i, point in enumerate(points):
            try:
                point_id = point["id"]
                vector = point["vector"]
                payload = point.get("payload") or {}

                if vector is None:
                    raise ValueError("vector 값이 비어 있습니다.")
                if len(vector) != self.vector_size:
                    raise ValueError(
                        f"벡터 차원 불일치 (기대={self.vector_size}, 실제={len(vector)})"
                    )

                point_structs.append(
                    qmodels.PointStruct(id=point_id, vector=vector, payload=payload)
                )
            except (KeyError, ValueError, TypeError) as exc:
                print(f"[QdrantStorageManager] {i}번째 포인트를 건너뜁니다 (잘못된 데이터): {exc}")
                continue

        if not point_structs:
            print("[QdrantStorageManager] 유효한 포인트가 하나도 없어 저장을 건너뜁니다.")
            return 0

        try:
            self.client.upsert(collection_name=self.collection_name, points=point_structs)
        except Exception as exc:
            print(f"[QdrantStorageManager] Qdrant upsert 중 오류 발생: {exc}")
            return 0

        return len(point_structs)

    def count(self) -> int:
        """현재 컬렉션에 저장된 포인트 개수를 반환한다 (조회 실패 시 0)."""
        try:
            result = self.client.count(collection_name=self.collection_name, exact=True)
            return result.count
        except Exception as exc:
            print(f"[QdrantStorageManager] 포인트 개수 조회 실패: {exc}")
            return 0

    def search(
        self,
        query_vector: list[float],
        top_k: int = 5,
        query_filter: qmodels.Filter | None = None,
    ) -> list:
        """쿼리 벡터와 가장 유사한 포인트들을 검색한다 (실패 시 빈 리스트).

        qdrant-client 1.10+ 부터 기존 `search()` API는 제거되었고 `query_points()`로
        대체되었다. 이 메서드는 `query_points()`를 호출한 뒤 `QueryResponse.points`
        (ScoredPoint 리스트, 각 원소는 `.id`/`.score`/`.payload` 속성을 가짐)만
        꺼내 돌려줘서, 호출하는 쪽(SearchEngine 등)은 API 변경과 무관하게 동일한
        인터페이스를 사용할 수 있다.

        Args:
            query_vector: 유사도를 계산할 쿼리 벡터 (CLIP 임베딩)
            top_k: 반환할 최대 포인트 수
            query_filter: Dense 벡터 검색과 함께 같은 호출에서 적용할 Sparse
                payload 필터(예: objects/texts 필드 조건). None이면 필터 없이
                순수 벡터 검색만 수행한다. 벡터 유사도 계산과 payload 필터링이
                Qdrant 내부에서 한 번의 쿼리로 함께 처리되므로 "1-Stage
                하이브리드 검색"이 된다 (SearchEngine이 필터를 만들어 전달).
        """
        try:
            response = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                query_filter=query_filter,
                limit=top_k,
            )
            return response.points
        except Exception as exc:
            print(f"[QdrantStorageManager] 검색 중 오류 발생: {exc}")
            return []


if __name__ == "__main__":
    manager = QdrantStorageManager()
    print(f"[QdrantStorageManager] 현재 저장된 포인트 수: {manager.count()}")
