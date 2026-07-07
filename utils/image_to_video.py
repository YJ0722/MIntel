"""
utils/image_to_video.py
========================
PNG 이미지 시퀀스를 하나의 .mp4 영상 파일로 합쳐주는 일회성 데이터 준비 유틸리티.

사용 목적:
    - 이 프로젝트의 pipeline_ingestion.py(1번 방)는 storage/videos/ 안의 .mp4
      파일만 입력으로 받는다.
    - 그런데 일부 공개 데이터셋은 영상이 아니라 프레임 단위의
      개별 PNG 이미지 파일 연속 시퀀스로 배포된다.
    - 이런 PNG 시퀀스를 파이프라인에 바로 넣을 수 없으므로, 이 스크립트로 미리
      mp4 영상 하나로 변환해두는 전처리 단계에 사용한다.

사용법:
    1. 아래 "실행 설정" 부분의 image_dir을 변환하려는 PNG 이미지 폴더의
       실제 경로로 수정한다.
    2. (mintel) $ python utils/image_to_video.py 로 실행하면 해당 폴더에
       image_to_video.mp4가 생성된다.
    3. 생성된 mp4 파일을 storage/videos/ 아래로 옮기면 pipeline_ingestion.py가
       자동으로 찾아서 Key-frame 추출 대상으로 사용한다.

참고:
    - 이 스크립트는 app.py의 파이프라인 실행 흐름에는 포함되어 있지 않다.
      새로운 PNG 시퀀스 데이터셋을 준비할 때만 필요에 따라 직접 실행하는 용도이다.
"""

import cv2
import os
import glob
from pathlib import Path

def images_to_video(image_folder, output_video_path, fps=10):
    # 1. 해당 폴더 내의 모든 PNG 파일 가져와서 이름순으로 정렬
    images = sorted(glob.glob(os.path.join(image_folder, "*.png")))
    if not images:
        print("정지 이미지(PNG)를 찾을 수 없습니다. 경로를 확인해 주세요.")
        return

    # 2. 첫 번째 이미지에서 해상도(가로, 세로) 정보 추출
    frame = cv2.imread(images[0])
    height, width, layers = frame.shape
    size = (width, height)

    # 3. 비디오 라이터 세팅 (Mac 환경에서 호환성이 좋은 mp4v 코덱 사용)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, size)

    print(f"총 {len(images)}장의 이미지를 mp4 비디오로 합치는 중...")
    
    # 4. 이미지 한 장씩 비디오 객체에 쓰기
    for img_path in images:
        img = cv2.imread(img_path)
        out.write(img)

    out.release()
    print(f"변환 완료! 파일 저장 위치: {output_video_path}")

# --- 실행 설정 ---
# 이 파일(utils/image_to_video.py) 기준으로 한 단계 위가 프로젝트 루트이다.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
VIDEOS_DIR = PROJECT_ROOT / "storage" / "videos"

# 다운로드받은 PNG 파일들이 모여있는 실제 경로로 수정하세요
image_dir = VIDEOS_DIR / "kitti" / "2011_09_29_drive_0071_sync" / "image_02" / "data"
output_mp4 = image_dir / "image_to_video.mp4"

images_to_video(str(image_dir), str(output_mp4), fps=10)