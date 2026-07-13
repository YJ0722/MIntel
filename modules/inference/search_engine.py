"""
modules/inference/search_engine.py
====================================
사용자의 자연어 질문으로 Qdrant에서 후보 Key-frame을 검색하는 모듈.

역할:
    - 사용자 질문(텍스트)을 ClipEmbedder.embed_text()로 벡터화한다.
    - CLIP은 이미지와 텍스트를 같은 벡터 공간에 투영하므로, 이 텍스트 벡터로
      Qdrant("mintel_collection")에 저장된 이미지 임베딩과 바로 유사도 검색을
      할 수 있다.
    - 검색된 상위 top_k개 포인트에서 frame_path와 payload(메타데이터)를 추려
      pipeline_inference.py로 반환한다. 이 모듈은 LLaVA/Llama-3에 대해서는
      전혀 알지 못하며, 순수하게 "질문 -> 후보 프레임 목록" 변환만 담당한다.
"""

from __future__ import annotations

from pathlib import Path

import config
from modules.ingestion.clip_embedder import ClipEmbedder
from modules.storage.qdrant_client import QdrantStorageManager

DEFAULT_TOP_K = 5


class SearchEngine:
    """텍스트 질문 -> CLIP 임베딩 -> Qdrant 유사도 검색을 담당하는 검색 엔진.

    사용 예:
        engine = SearchEngine()
        candidates = engine.search("빨간 차가 보이는 장면", top_k=3)
        # -> [{"frame_path": "...", "score": 0.83, "payload": {...}}, ...]
    """

    def __init__(
        self,
        clip_embedder: ClipEmbedder | None = None,
        storage_manager: QdrantStorageManager | None = None,
        db_path: Path = config.DB_DIR,
    ) -> None:
        """
        Args:
            clip_embedder: 재사용할 ClipEmbedder 인스턴스. None이면 새로 로드한다.
            storage_manager: 재사용할 QdrantStorageManager 인스턴스. None이면
                db_path를 사용해 새로 연결한다.
            db_path: storage_manager를 새로 만들 때 사용할 Qdrant DB 경로
                (기본값: config.DB_DIR)
        """
        self.clip_embedder = clip_embedder or ClipEmbedder()
        self.storage_manager = storage_manager or QdrantStorageManager(db_path=db_path)

    def search(self, user_query: str, top_k: int = DEFAULT_TOP_K) -> list[dict]:
        """
        사용자 질문과 가장 유사한 후보 Key-frame들을 Qdrant에서 검색한다.

        Args:
            user_query: 사용자의 자연어 질문 (예: "신호등이 보이는 장면 있어?")
            top_k: 검색할 후보 프레임 개수

        Returns:
            각 원소가 {"frame_path": str, "score": float, "payload": dict} 형태인
            리스트. frame_path가 실제로 디스크에 없는(삭제/이동된) 포인트는
            건너뛴다.
        """
        query_vector = self.clip_embedder.embed_text(user_query)
        hits = self.storage_manager.search(query_vector, top_k=top_k)

        candidates: list[dict] = []
        for hit in hits:
            payload = hit.payload or {}
            frame_path = payload.get("frame_path")

            if not frame_path or not Path(frame_path).exists():
                print(f"[SearchEngine] 프레임 파일을 찾을 수 없어 건너뜁니다: {frame_path}")
                continue

            candidates.append(
                {
                    "frame_path": frame_path,
                    "score": hit.score,
                    "payload": payload,
                }
            )

        print(
            f"[SearchEngine] '{user_query}' 검색 완료: "
            f"후보 프레임 {len(candidates)}개 (top_k={top_k})"
        )
        return candidates


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("사용법: python -m modules.inference.search_engine <질문>")
    else:
        engine = SearchEngine()
        for candidate in engine.search(" ".join(sys.argv[1:])):
            print(candidate)
