"""
modules/ingestion/keyframe_extractor.py
=========================================
히스토그램 기반 동적 Key-frame 추출 모듈.

역할:
    - storage/videos/ 폴더(하위 폴더 포함)에 있는 원본 .mp4 영상을 OpenCV(cv2)로
      읽어들여 프레임 단위로 분해하고, 동적 Key-frame 추출 알고리즘으로 의미 있는
      프레임만 골라 storage/frames/ 폴더에 이미지로 저장한다.
    - pipeline_ingestion.py(1번 방)는 이 모듈의 KeyframeExtractor를 호출해
      영상 -> Key-frame 변환을 수행한다.

Key-frame 추출 알고리즘 (히스토그램 기반 동적 추출):
    1. 프레임을 하나씩 순차적으로 읽는다 (cv2.VideoCapture 스트리밍, 전체를
       메모리에 올리지 않음).
    2. "바로 직전에 저장된 키프레임"과 현재 프레임의 색상 히스토그램 유사도를
       cv2.compareHist(HISTCMP_CORREL)로 계산한다. (1.0=완전히 동일, 0에 가까울수록
       다른 장면)
    3. 마지막 키프레임 이후 min_frame_interval 프레임이 지나지 않았다면, 유사도와
       무관하게 후보에서 제외한다 (너무 촘촘하게 뽑히는 것을 방지).
    4. 유사도가 similarity_threshold 미만으로 떨어지면(=장면이 충분히 바뀌면) 새
       Key-frame으로 지정하고 저장한다.

메모리 관리 (M3 Pro 18GB 환경):
    - 영상을 한 번에 전부 메모리에 올리지 않고, cv2.VideoCapture의
      스트리밍 방식(read() 반복 호출)을 사용해 프레임을 한 장씩만
      메모리에 유지한다.
    - Key-frame으로 선택되지 않은 프레임은 즉시 참조를 버려(변수 재할당) 메모리
      누적을 방지하고, 영상 하나 처리가 끝나면 cap.release()를 반드시 호출한다.
    - 히스토그램 비교 덕분에 중복도가 높은 프레임을 걸러내어, 뒤이은 임베딩/객체
      탐지/VLM 추론 단계의 연산 부하도 함께 줄어든다.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

# ------------------------------------------------------------------
# 경로 설정
# (이 파일은 modules/ingestion/ 아래에 있으므로 두 단계 상위가 프로젝트 루트)
# ------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent.parent
VIDEOS_DIR = BASE_DIR / "storage" / "videos"
FRAMES_DIR = BASE_DIR / "storage" / "frames"

# ------------------------------------------------------------------
# Key-frame 추출 파라미터
# ------------------------------------------------------------------
# 직전 키프레임과의 히스토그램 유사도(0~1, 1=완전히 동일)가 이 값 미만으로
# 떨어지면 장면이 바뀐 것으로 보고 새 Key-frame으로 지정한다.
DEFAULT_SIMILARITY_THRESHOLD = 0.90

# 유사도와 무관하게 최소 이 프레임 수만큼은 지나야 다음 Key-frame 후보가 될 수 있다.
DEFAULT_MIN_FRAME_INTERVAL = 10

# 히스토그램 계산에 사용할 채널(Hue, Saturation) / 구간(bin) 수 / 값 범위
HIST_CHANNELS = [0, 1]
HIST_BINS = [50, 60]
HIST_RANGES = [0, 180, 0, 256]


@dataclass
class KeyFrame:
    """
    추출된 하나의 Key-frame을 표현하는 데이터 클래스.

    embedding/objects/texts는 KeyframeExtractor 단계에서는 비어 있고, 이후
    pipeline_ingestion.enrich_keyframes()가 CLIP/YOLOv8/EasyOCR을 적용해
    채워 넣는다. 즉, 이 객체는 "Ingestion Layer"의 최종 산출물(가공된 데이터 +
    벡터)로서 pipeline_storage.py에 그대로 전달되어 저장되는 단위이다.
    """

    video_name: str
    frame_index: int
    timestamp_sec: float
    image: np.ndarray | None  # BGR 이미지 (OpenCV 기본 포맷). 저장 후에는 None으로 비워짐
    saved_path: Path | None = None
    embedding: list[float] | None = None
    objects: list[str] | None = None
    texts: list[str] | None = None


def list_video_files(videos_dir: Path = VIDEOS_DIR) -> list[Path]:
    """storage/videos/ 폴더(하위 폴더 포함) 내의 .mp4 파일 목록을 반환한다."""
    if not videos_dir.exists():
        return []
    return sorted(p for p in videos_dir.rglob("*.mp4") if p.is_file())


class KeyframeExtractor:
    """히스토그램 기반 동적 Key-frame 추출기.

    사용 예:
        extractor = KeyframeExtractor()
        saved_paths = extractor.extract_from_video(video_path, frames_dir)
    """

    def __init__(
        self,
        similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
        min_frame_interval: int = DEFAULT_MIN_FRAME_INTERVAL,
    ) -> None:
        """
        Args:
            similarity_threshold: 직전 키프레임과의 유사도가 이 값 미만이면 새 키프레임으로 지정
            min_frame_interval: 마지막 키프레임 이후 최소 이만큼의 프레임이 지나야 다음 후보가 될 수 있음
        """
        self.similarity_threshold = similarity_threshold
        self.min_frame_interval = min_frame_interval

    @staticmethod
    def _compute_histogram(image: np.ndarray) -> np.ndarray:
        """이미지의 HSV 색상 히스토그램을 계산하고 정규화하여 반환한다."""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], HIST_CHANNELS, None, HIST_BINS, HIST_RANGES)
        cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        return hist

    @staticmethod
    def _histogram_similarity(hist_a: np.ndarray, hist_b: np.ndarray) -> float:
        """
        두 히스토그램의 유사도를 반환한다.

        cv2.HISTCMP_CORREL 기준: 1.0에 가까울수록 완전히 동일한 장면,
        0 또는 음수에 가까울수록 서로 다른 장면을 의미한다.
        """
        return float(cv2.compareHist(hist_a, hist_b, cv2.HISTCMP_CORREL))

    @staticmethod
    def _build_video_identifier(video_path: Path, videos_dir: Path) -> str:
        """
        영상 파일명이 서로 다른 폴더에서 겹치는 경우를 대비해 고유한 식별자를 만든다.

        예를 들어 KITTI 데이터셋처럼 여러 시퀀스를 각각 변환해도 모두
        "kitti_sample.mp4"라는 동일한 파일명이 되는 경우, 바로 위 부모 폴더명만으로는
        (예: 둘 다 .../image_02/data/kitti_sample.mp4 라서 "data"로 동일) 구분이 안
        되므로, videos_dir 기준 상대 경로 전체를 이어붙여 고유성을 보장한다.
        (예: kitti_2011_09_26_drive_0001_sync_image_02_data_kitti_sample)

        videos_dir 하위 경로가 아니어서 상대 경로를 구할 수 없는 경우에는
        안전하게 파일명만 사용한다.
        """
        try:
            relative = video_path.resolve().relative_to(videos_dir.resolve())
            return "_".join(relative.with_suffix("").parts)
        except ValueError:
            return video_path.stem

    @staticmethod
    def save_frame(frame: KeyFrame, frames_dir: Path = FRAMES_DIR) -> Path:
        """KeyFrame 객체의 이미지를 frames_dir에 jpg로 저장한다."""
        frames_dir.mkdir(parents=True, exist_ok=True)

        # 여러 영상을 함께 처리해도 파일명이 겹치지 않도록 영상 이름을 접두어로 붙이고,
        # 프레임 번호는 원본 인덱스를 그대로 유지한다. (예: kitti_sample_frame_000012.jpg)
        filename = f"{frame.video_name}_frame_{frame.frame_index:06d}.jpg"
        save_path = frames_dir / filename

        cv2.imwrite(str(save_path), frame.image)
        frame.saved_path = save_path
        return save_path

    def extract_from_video(
        self,
        video_path: Path,
        frames_dir: Path = FRAMES_DIR,
        videos_dir: Path = VIDEOS_DIR,
    ) -> list[KeyFrame]:
        """
        영상 하나에 동적 Key-frame 추출 알고리즘을 적용하고, 선택된 프레임들을
        frames_dir에 저장한 뒤 저장된 KeyFrame(메타데이터 포함) 목록을 반환한다.

        Args:
            video_path: 읽어들일 .mp4 파일 경로
            frames_dir: Key-frame 이미지를 저장할 폴더
            videos_dir: video_path의 기준 폴더. 서로 다른 폴더의 영상이 같은 파일명을
                가지더라도 저장되는 프레임 파일명이 겹치지 않도록 식별자 생성에 사용된다.

        Returns:
            저장된 KeyFrame 목록. 메모리 절약을 위해 각 KeyFrame의 image는 저장 직후
            None으로 비워지며, saved_path/video_name/frame_index/timestamp_sec 등
            메타데이터만 유지된다 (후속 단계는 saved_path로 이미지를 다시 읽는다).
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise IOError(f"영상을 열 수 없습니다: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_identifier = self._build_video_identifier(video_path, videos_dir)

        saved_keyframes: list[KeyFrame] = []
        last_keyframe_hist: np.ndarray | None = None
        frames_since_last_keyframe = 0
        frame_index = 0

        try:
            while True:
                ok, image = cap.read()
                if not ok:
                    break  # 영상 끝

                is_new_keyframe = False

                if last_keyframe_hist is None:
                    # 영상의 첫 프레임은 비교 대상이 없으므로 무조건 최초 Key-frame으로 지정
                    is_new_keyframe = True
                elif frames_since_last_keyframe >= self.min_frame_interval:
                    current_hist = self._compute_histogram(image)
                    similarity = self._histogram_similarity(last_keyframe_hist, current_hist)
                    if similarity < self.similarity_threshold:
                        is_new_keyframe = True

                if is_new_keyframe:
                    frame = KeyFrame(
                        video_name=video_identifier,
                        frame_index=frame_index,
                        timestamp_sec=frame_index / fps,
                        image=image,
                    )
                    saved_path = self.save_frame(frame, frames_dir)

                    last_keyframe_hist = self._compute_histogram(image)
                    frames_since_last_keyframe = 0

                    print(
                        f"  - Key-frame 저장: {saved_path.name} "
                        f"(frame={frame_index}, t={frame.timestamp_sec:.2f}s)"
                    )

                    # 이미지는 이미 디스크에 저장했으므로 메모리에서는 참조를 비운다.
                    frame.image = None
                    saved_keyframes.append(frame)
                else:
                    frames_since_last_keyframe += 1

                # 다음 반복을 위해 현재 프레임 참조를 넘기고 지역 변수는 재사용되도록 둔다
                # (Key-frame이 아닌 프레임은 별도로 들고 있지 않으므로 즉시 GC 대상이 된다)
                frame_index += 1
        finally:
            cap.release()  # 메모리 누수 방지를 위해 항상 명시적으로 해제

        extraction_rate = (len(saved_keyframes) / total_frames * 100) if total_frames else 0.0
        print(
            f"[Ingestion] 총 {total_frames}프레임 중 {len(saved_keyframes)}개의 키프레임 추출 완료 "
            f"(추출률: {extraction_rate:.1f}%)"
        )

        return saved_keyframes


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("사용법: python keyframe_extractor.py <영상_경로> [저장_폴더]")
    else:
        video_path_arg = Path(sys.argv[1])
        frames_dir_arg = Path(sys.argv[2]) if len(sys.argv) > 2 else FRAMES_DIR
        extractor = KeyframeExtractor()
        extractor.extract_from_video(video_path_arg, frames_dir_arg)
