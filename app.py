"""
app.py
======
하이브리드 비전 인덱싱 및 검증 파이프라인 시스템 - 메인 진입점

전체 흐름:
    1번 방 (pipeline_ingestion, Ingestion Layer)
        : storage/videos/ 에서 영상을 읽어 Key-frame을 추출하고,
          CLIP 임베딩 + YOLOv8 객체 탐지 + EasyOCR 텍스트 인식까지 가공한다.
        -> 2번 방 (pipeline_storage, Storage Layer)
        : 가공이 끝난 Key-frame(벡터 + 메타데이터)을 로컬 Qdrant DB에 저장한다.
        -> 3번 방 (pipeline_inference, Inference Layer)
        : Ollama(llama3, llava)로 프레임 내용을 검증/설명한다.

입출력 경로:
    모든 입력/출력 데이터 경로(영상, 프레임, 벡터 DB)는 config.py 한 곳에서
    관리하며, 이 파일에서 각 파이프라인 함수에 명시적으로 전달한다. 경로를
    바꾸고 싶으면 config.py만 수정하면 된다.

실행 방법 (mintel 가상환경이 이미 활성화되어 있다고 가정):
    (mintel) $ python app.py
"""

from __future__ import annotations

import config
import pipeline_ingestion
import pipeline_inference
import pipeline_storage


def main() -> None:
    print("=" * 60)
    print("하이브리드 비전 인덱싱 및 검증 파이프라인 시스템 - 시작")
    print("=" * 60)

    # ------------------------------------------------------------
    # 1번 방 (Ingestion Layer): 영상(config.VIDEOS_DIR) -> Key-frame 추출
    #   (config.FRAMES_DIR) -> CLIP/YOLOv8/EasyOCR 가공까지 완료
    # ------------------------------------------------------------
    print("\n[1/3] 영상 프레임 추출 + 가공 파이프라인 실행 (pipeline_ingestion)")
    saved_keyframes = pipeline_ingestion.run_ingestion(
        videos_dir=config.VIDEOS_DIR,
        frames_dir=config.FRAMES_DIR,
    )
    print(f"      -> 추출/가공된 Key-frame 수: {len(saved_keyframes)}")

    # ------------------------------------------------------------
    # 2번 방 (Storage Layer): 가공된 Key-frame -> Qdrant 벡터 DB(config.DB_DIR) 저장
    # ------------------------------------------------------------
    print("\n[2/3] 벡터 DB 저장 파이프라인 실행 (pipeline_storage)")
    indexed_count = pipeline_storage.run_storage_pipeline(
        saved_keyframes, db_path=config.DB_DIR
    )
    print(f"      -> Qdrant에 적재된 Key-frame 수: {indexed_count}")

    # ------------------------------------------------------------
    # 3번 방: Ollama(llama3, llava) 기반 검증/추론
    # ------------------------------------------------------------
    print("\n[3/3] 로컬 LLM/VLM 추론 파이프라인 실행 (pipeline_inference)")
    pipeline_inference.run_inference_smoke_test(frames_dir=config.FRAMES_DIR)
    # TODO: 추후 2번 방에서 검색된(혹은 새로 추출된) Key-frame들을 llava로 설명시키고,
    #       그 결과를 llama3로 재검증/요약하는 하이브리드 검증 로직이 들어올 예정.

    print("\n" + "=" * 60)
    print("파이프라인 실행 완료")
    print("=" * 60)


if __name__ == "__main__":
    main()
