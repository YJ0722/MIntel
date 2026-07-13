"""
config.py
=========
프로젝트 전역 입출력 경로 설정.

이 파일은 파이프라인이 어디서 데이터를 읽고(input) 어디에 결과를 쓰는지(output)를
한 곳에 모아 관리한다. app.py는 이 상수들을 각 파이프라인 함수(run_ingestion,
run_storage_setup 등)에 인자로 명시적으로 전달하고, 각 pipeline_*.py 파일은
단독 실행(python pipeline_ingestion.py 등)할 때를 대비해 이 값을 기본값으로 사용한다.

나중에 입출력 경로를 바꾸고 싶을 때(예: 외장 드라이브의 다른 데이터셋을 쓰고
싶을 때) 이 파일만 수정하면 된다 — modules/ 안의 알고리즘 코드는 건드릴 필요가 없다.
"""

from __future__ import annotations

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
STORAGE_DIR = BASE_DIR / "storage"

# 1번 방(pipeline_ingestion) 입력: 원본 영상(.mp4)이 위치한 폴더
VIDEOS_DIR = STORAGE_DIR / "videos"

# 1번 방(pipeline_ingestion) 출력 / 2번 방(pipeline_storage) 입력: 추출된 Key-frame 이미지 폴더
FRAMES_DIR = STORAGE_DIR / "frames"

# 2번 방(pipeline_storage) 출력: 로컬 Qdrant 벡터 DB 파일이 저장되는 폴더
DB_DIR = STORAGE_DIR / "db"

# 3번 방(pipeline_inference) 입력: app.py 실행 시 고정으로 사용하는 데모용 질문.
# 아직 사용자가 실제로 질문을 입력하는 창구(웹/CLI 입력 등)가 없어서, app.py는
# 항상 이 값 하나로만 파이프라인을 실행한다. 다른 질문으로 테스트하고 싶으면
# 이 값을 바꾸거나, `python pipeline_inference.py "질문"`처럼 커맨드라인
# 인자로 직접 넘겨서 실행하면 된다 (그 경우 이 값은 쓰이지 않는다).
SAMPLE_QUERY = "차가 보이는 장면을 찾아줘"
