"""
modules/inference/search_engine.py
====================================
사용자의 자연어 질문으로 Qdrant에서 후보 Key-frame을 검색하는 모듈.

역할 (Dense + Sparse 1-Stage 하이브리드 검색):
    - Dense: 사용자 질문(텍스트)을 ClipEmbedder.embed_text()로 벡터화한다.
      CLIP은 이미지와 텍스트를 같은 벡터 공간에 투영하므로, 이 텍스트 벡터로
      Qdrant("mintel_collection")에 저장된 이미지 임베딩과 바로 유사도 검색을
      할 수 있다.
    - Sparse: keyword_matcher.py로 질문에서 YOLOv8 객체 클래스("신호등" ->
      "traffic light" 등)와 OCR 텍스트 힌트("STOP" 등)를 추출해, Qdrant
      payload(objects/texts)에 대한 필터 조건으로 변환한다.
    - 위 Dense 쿼리 벡터와 Sparse 필터를 하나의 Qdrant query_points() 호출에
      함께 전달해(=1-Stage), 벡터 유사도 계산과 payload 필터링이 별도의
      후처리 단계 없이 DB 내부에서 한 번에 이뤄지도록 한다.
    - Sparse 필터를 걸었는데 후보가 하나도 없으면(YOLO/EasyOCR이 놓친 경우 등
      탐지 누락 가능성 고려), 결과가 0개로 죽지 않도록 Dense 전용 검색으로
      안전하게 재시도한다.
    - 검색된 상위 top_k개 포인트에서 frame_path와 payload(메타데이터)를 추려
      pipeline_inference.py로 반환한다. 이 모듈은 LLaVA/Llama-3에 대해서는
      전혀 알지 못하며, 순수하게 "질문 -> 후보 프레임 목록" 변환만 담당한다.
"""

from __future__ import annotations

from pathlib import Path

from qdrant_client.http import models as qmodels

import config
from modules.ingestion.clip_embedder import ClipEmbedder
from modules.inference.keyword_matcher import extract_object_classes, extract_text_hints
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

    @staticmethod
    def _build_sparse_filter(
        matched_classes: list[str], matched_texts: list[str]
    ) -> qmodels.Filter | None:
        """
        추출된 객체 클래스/텍스트 힌트를 Qdrant payload 필터(Filter)로 변환한다.

        objects 조건과 texts 조건을 should(OR)로 묶어, 질문에서 추출된 신호 중
        하나라도 해당 프레임의 payload와 일치하면 후보로 남긴다. 추출된 신호가
        전혀 없으면 필터를 만들지 않고 None을 반환해(=Dense 전용 검색) 불필요한
        제약을 걸지 않는다.
        """
        should_conditions: list[qmodels.FieldCondition] = []

        if matched_classes:
            should_conditions.append(
                qmodels.FieldCondition(key="objects", match=qmodels.MatchAny(any=matched_classes))
            )
        if matched_texts:
            should_conditions.append(
                qmodels.FieldCondition(key="texts", match=qmodels.MatchAny(any=matched_texts))
            )

        if not should_conditions:
            return None
        return qmodels.Filter(should=should_conditions)

    def search(self, user_query: str, top_k: int = DEFAULT_TOP_K) -> list[dict]:
        """
        사용자 질문과 가장 유사한 후보 Key-frame들을 Qdrant에서 검색한다
        (Dense 벡터 유사도 + Sparse 객체/텍스트 필터링을 결합한 1-Stage 하이브리드 검색).

        Args:
            user_query: 사용자의 자연어 질문 (예: "신호등이 보이는 장면 있어?")
            top_k: 검색할 후보 프레임 개수

        Returns:
            각 원소가 {"frame_path": str, "score": float, "payload": dict} 형태인
            리스트. frame_path가 실제로 디스크에 없는(삭제/이동된) 포인트는
            건너뛴다.
        """
        query_vector = self.clip_embedder.embed_text(user_query)

        matched_classes = extract_object_classes(user_query)
        matched_texts = extract_text_hints(user_query)
        sparse_filter = self._build_sparse_filter(matched_classes, matched_texts)

        if sparse_filter is not None:
            print(
                f"[SearchEngine] Sparse 신호 감지 -> objects={matched_classes or '없음'}, "
                f"texts={matched_texts or '없음'} (Dense+Sparse 하이브리드 검색 수행)"
            )

        hits = self.storage_manager.search(query_vector, top_k=top_k, query_filter=sparse_filter)

        if sparse_filter is not None and not hits:
            print(
                "[SearchEngine] Sparse 필터를 만족하는 후보가 없어 Dense 전용 검색으로 재시도합니다..."
            )
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
