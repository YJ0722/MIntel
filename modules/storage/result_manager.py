"""
modules/storage/result_manager.py
===================================
[3번 방] Inference Layer의 최종 산출물(결과 JSON + 실행 로그) 저장을 담당하는
결과 관리 모듈.

역할:
    - pipeline_inference.py가 검색(SearchEngine) + 시각 추론(LlavaAnalyzer)까지
      끝낸 뒤, 그 결과를 지정된 규격의 result.json으로 정리해 저장한다.
    - 파이프라인이 실행되는 동안 콘솔에 출력되는 로그(print)를 그대로 실행 시각
      폴더 안의 run.log 파일에도 함께 남긴다 (콘솔 출력은 그대로 유지되고, 파일에도
      동일하게 기록되는 tee 방식).
    - 이 모듈은 검색/추론 로직을 전혀 알지 못하며, 순수하게 "정리된 데이터 ->
      파일 저장" 만 담당한다 (Storage Layer의 qdrant_client.py가 벡터 DB 저장만
      담당하는 것과 동일한 책임 분리 원칙).

저장 구조:
    storage/results/<YYYYMMDD_HHMMSS>/
        result.json   : 검색+시각 추론 결과 (아래 "결과 JSON 규격" 참고)
        run.log        : 해당 실행의 전체 콘솔 로그(tee)

결과 JSON 규격:
    {
      "status": "success" | "no_match" | "error",
      "matches": [
        {
          "rank": 1,
          "video_source": "traffic_clip_01.mp4",
          "timestamp": "00:14:22",
          "frame_path": "./storage/frames/key_0942.jpg",
          "confidence_score": 0.94,
          "analysis_report": "..."
        },
        ...
      ]
    }
    - status가 "error"인 경우 "error_message" 키가 추가로 포함된다.
"""

from __future__ import annotations

import json
import sys
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Iterator, TextIO

import config

DEFAULT_RESULT_FILENAME = "result.json"
DEFAULT_LOG_FILENAME = "run.log"


class _Tee:
    """여러 출력 스트림(콘솔 + 로그 파일)에 동시에 쓰는 얕은 래퍼.

    sys.stdout/sys.stderr를 이 객체로 교체해두면, print()로 찍히는 모든 내용이
    콘솔에는 그대로 보이면서 동시에 파일에도 기록된다.
    """

    def __init__(self, *streams: TextIO) -> None:
        self._streams = streams

    def write(self, data: str) -> None:
        for stream in self._streams:
            stream.write(data)
            stream.flush()

    def flush(self) -> None:
        for stream in self._streams:
            stream.flush()


class ResultManager:
    """실행 시각 폴더 생성 + 결과 JSON 저장 + 실행 로그 캡처를 담당하는 매니저.

    사용 예:
        result_manager = ResultManager()
        with result_manager.capture_logs():
            ...  # 검색 + 시각 추론 수행
            matches = result_manager.build_matches(candidates, visual_findings)
            result_manager.save_result(matches, status="success")
    """

    def __init__(
        self,
        results_dir: Path = config.RESULTS_DIR,
        run_id: str | None = None,
    ) -> None:
        """
        Args:
            results_dir: 실행 시각 서브 폴더들을 담을 상위 폴더 (기본값: config.RESULTS_DIR)
            run_id: 서브 폴더 이름으로 쓸 식별자. None이면 현재 시각 기준
                YYYYMMDD_HHMMSS 형식으로 자동 생성한다.
        """
        self.results_dir = Path(results_dir)
        self.run_id = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = self.results_dir / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.result_path = self.run_dir / DEFAULT_RESULT_FILENAME
        self.log_path = self.run_dir / DEFAULT_LOG_FILENAME

    @contextmanager
    def capture_logs(self) -> Iterator[None]:
        """실행 구간 동안의 모든 print 출력을 콘솔 + run.log 파일에 동시에 남긴다."""
        started_at = datetime.now()
        log_file = self.log_path.open("a", encoding="utf-8")
        divider = "=" * 60

        log_file.write(f"{divider}\n")
        log_file.write(f"[ResultManager] 파이프라인 실행 로그 (run_id={self.run_id})\n")
        log_file.write(f"[ResultManager] 시작 시각: {started_at:%Y-%m-%d %H:%M:%S}\n")
        log_file.write(f"{divider}\n")
        log_file.flush()

        original_stdout, original_stderr = sys.stdout, sys.stderr
        sys.stdout = _Tee(original_stdout, log_file)
        sys.stderr = _Tee(original_stderr, log_file)
        try:
            yield
        finally:
            sys.stdout, sys.stderr = original_stdout, original_stderr
            finished_at = datetime.now()
            elapsed_sec = (finished_at - started_at).total_seconds()
            log_file.write(f"{divider}\n")
            log_file.write(
                f"[ResultManager] 종료 시각: {finished_at:%Y-%m-%d %H:%M:%S} "
                f"(총 소요 {elapsed_sec:.1f}초)\n"
            )
            log_file.write(f"{divider}\n")
            log_file.close()

    @staticmethod
    def _format_timestamp(seconds: float) -> str:
        """초 단위 시각을 "HH:MM:SS" 형식으로 변환한다."""
        total_seconds = max(0, int(round(seconds)))
        hours, remainder = divmod(total_seconds, 3600)
        minutes, secs = divmod(remainder, 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"

    @staticmethod
    def _resolve_video_source(payload: dict) -> str:
        """payload["video_name"]으로부터 원본 영상 파일명(예: traffic_clip_01.mp4)을 만든다."""
        video_name = str(payload.get("video_name") or "unknown")
        return video_name if video_name.lower().endswith(".mp4") else f"{video_name}.mp4"

    @staticmethod
    def _to_relative_frame_path(frame_path: str, base_dir: Path = config.BASE_DIR) -> str:
        """저장된 프레임의 절대 경로를 "./storage/frames/xxx.jpg" 형태의 상대 경로로 바꾼다."""
        try:
            relative = Path(frame_path).resolve().relative_to(base_dir.resolve())
            return f"./{relative.as_posix()}"
        except ValueError:
            return frame_path

    def build_matches(
        self,
        candidates: list[dict],
        visual_findings: list[dict],
        only_verified: bool = True,
    ) -> list[dict]:
        """
        SearchEngine의 검색 결과(candidates)와 LlavaAnalyzer의 시각 검증 결과
        (visual_findings)를 결합해, 결과 JSON 규격의 "matches" 배열을 만든다.

        Args:
            candidates: SearchEngine.search()가 반환한 리스트
                (각 원소: {"frame_path", "score", "payload"})
            visual_findings: LlavaAnalyzer.analyze()가 반환한 결과 리스트로,
                candidates와 같은 순서/개수로 대응된다
                (각 원소: {"image_path", "exists", "description"}).
            only_verified: True면 LLaVA가 "존재함(exists=True)"으로 판정한
                프레임만 최종 매치로 남긴다 (기본값: True).

        Returns:
            confidence_score(검색 유사도) 내림차순으로 정렬되고 rank(1부터)가
            새로 매겨진 매치 딕셔너리 리스트.
        """
        unranked: list[dict] = []
        for candidate, finding in zip(candidates, visual_findings):
            if only_verified and not finding.get("exists"):
                continue

            payload = candidate.get("payload") or {}
            unranked.append(
                {
                    "video_source": self._resolve_video_source(payload),
                    "timestamp": self._format_timestamp(payload.get("timestamp_sec", 0.0)),
                    "frame_path": self._to_relative_frame_path(candidate.get("frame_path", "")),
                    "confidence_score": round(float(candidate.get("score", 0.0)), 2),
                    "analysis_report": str(finding.get("description", "")).strip(),
                }
            )

        unranked.sort(key=lambda item: item["confidence_score"], reverse=True)

        matches: list[dict] = []
        for rank, item in enumerate(unranked, start=1):
            matches.append(
                {
                    "rank": rank,
                    "video_source": item["video_source"],
                    "timestamp": item["timestamp"],
                    "frame_path": item["frame_path"],
                    "confidence_score": item["confidence_score"],
                    "analysis_report": item["analysis_report"],
                }
            )
        return matches

    def save_result(
        self,
        matches: list[dict],
        status: str = "success",
        error_message: str | None = None,
    ) -> Path:
        """
        최종 결과를 규격에 맞춰 result.json으로 저장한다.

        Args:
            matches: build_matches()가 만든 매치 리스트 (오류 상태일 때는 보통 빈 리스트)
            status: "success" | "no_match" | "error"
            error_message: status="error"일 때 함께 기록할 오류 메시지

        Returns:
            저장된 result.json의 경로
        """
        result_payload: dict = {"status": status, "matches": matches}
        if error_message:
            result_payload["error_message"] = error_message

        self.result_path.write_text(
            json.dumps(result_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"[ResultManager] 결과 JSON 저장 완료: {self.result_path}")
        return self.result_path


if __name__ == "__main__":
    demo_manager = ResultManager()
    with demo_manager.capture_logs():
        print("[ResultManager] 데모 실행: 샘플 결과를 저장합니다.")
        demo_matches = demo_manager.build_matches(
            candidates=[
                {
                    "frame_path": str(config.FRAMES_DIR / "sample_frame_000000.jpg"),
                    "score": 0.94,
                    "payload": {"video_name": "traffic_clip_01", "timestamp_sec": 862.0},
                }
            ],
            visual_findings=[
                {
                    "image_path": str(config.FRAMES_DIR / "sample_frame_000000.jpg"),
                    "exists": True,
                    "description": "STOP 표지판 직전 정지선에 백색 SUV 차량이 정차 중.",
                }
            ],
        )
        demo_manager.save_result(demo_matches, status="success")
