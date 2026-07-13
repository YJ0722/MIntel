"""
modules/inference/llava_analyzer.py
=====================================
LLaVA(비전-언어 모델) 기반 후보 프레임 정밀 검증 모듈.

역할:
    - SearchEngine이 Qdrant에서 찾아온 후보 프레임 이미지들을 하나씩 LLaVA에게
      보여주고, 사용자 질문에 해당하는 상황/사물이 실제로 그 이미지에 존재하는지
      true/false로 판단시키는 동시에 상세 묘사(2차 설명)도 함께 받아온다.
    - Ollama 서버에 이미 pull된 로컬 llava 모델을 통해 추론하며, GPU가 아니라
      Ollama 런타임이 자체적으로 Apple Silicon 가속(Metal)을 활용한다.

응답 형식:
    - LLaVA가 자유 서술형으로 답하면 true/false 판정을 프로그램이 파싱하기
      어려우므로, Ollama의 `format="json"` 옵션으로 항상 유효한 JSON
      ({"exists": bool, "description": str})만 반환하도록 강제한다.

메모리 관리 (M3 Pro 18GB 환경):
    - Ollama의 keep_alive 옵션으로 모델을 일정 시간(기본 5분) 동안 메모리에
      유지해 후보 프레임 여러 장을 재로딩 없이 순차 분석한다.
    - 모든 프레임 분석이 끝나면 unload()를 호출해 Llama-3를 로드하기 전에
      LLaVA를 즉시 메모리에서 내려, 두 대형 모델이 동시에 상주하지 않도록 한다.
"""

from __future__ import annotations

import json
from pathlib import Path

import ollama

DEFAULT_MODEL_NAME = "llava"
DEFAULT_KEEP_ALIVE = "5m"


class LlavaAnalyzer:
    """LLaVA 기반 이미지 존재 여부 판단 + 상세 묘사 추출기.

    사용 예:
        analyzer = LlavaAnalyzer()
        result = analyzer.analyze("빨간 차가 보여?", "storage/frames/sample_0.jpg")
        # -> {"image_path": "...", "exists": True, "description": "..."}
        analyzer.unload()  # 다음 모델(Llama-3)을 위해 메모리 확보
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL_NAME,
        keep_alive: str = DEFAULT_KEEP_ALIVE,
    ) -> None:
        """
        Args:
            model: 사용할 Ollama 비전 모델 이름 (기본값: llava)
            keep_alive: 추론 후 모델을 메모리에 유지할 시간 (기본값: "5m").
                여러 후보 프레임을 연속으로 분석할 때 매번 재로딩하지 않도록
                한다. "0"으로 주면 응답 즉시 언로드된다.
        """
        self.model = model
        self.keep_alive = keep_alive
        print(f"[LlavaAnalyzer] '{self.model}' 모델 사용 준비 완료 (keep_alive={self.keep_alive}).")

    def analyze(self, user_query: str, image_path: str | Path) -> dict:
        """
        이미지 한 장에 대해 사용자 질문 기준으로 존재 여부와 상세 묘사를 판단한다.

        Args:
            user_query: 사용자의 자연어 질문
            image_path: 분석할 Key-frame 이미지 경로

        Returns:
            {"image_path": str, "exists": bool, "description": str} 형태의 딕셔너리.
            모델 호출이나 응답 파싱에 실패하면 exists=False, description에
            실패 사유를 담아 반환한다 (예외를 던지지 않고 파이프라인이 계속
            진행될 수 있도록 한다).
        """
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"이미지 파일을 찾을 수 없습니다: {image_path}")

        prompt = (
            f'사용자 질문: "{user_query}"\n'
            "이 이미지를 자세히 살펴보고, 사용자가 찾는 상황이나 사물이 이 이미지에 "
            "실제로 존재하는지 판단해줘.\n"
            "반드시 다음 JSON 형식으로만 답변해 (다른 문장 없이 JSON 하나만):\n"
            '{"exists": true 또는 false, "description": "이미지에 대한 상세한 한국어 설명"}'
        )

        content = ""
        try:
            response = ollama.chat(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                        "images": [str(image_path)],
                    }
                ],
                format="json",
                keep_alive=self.keep_alive,
            )
            content = response["message"]["content"]
            parsed = json.loads(content)
            exists = bool(parsed.get("exists", False))
            description = str(parsed.get("description", "")).strip()
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            print(
                f"[LlavaAnalyzer] '{image_path.name}' 응답을 JSON으로 해석하지 못해 "
                f"원문을 설명으로 사용합니다: {exc}"
            )
            exists = False
            description = content
        except Exception as exc:  # noqa: BLE001 - Ollama 연결/모델 오류를 광범위하게 캐치
            print(f"[LlavaAnalyzer] '{image_path.name}' 분석 중 오류 발생: {exc}")
            exists = False
            description = f"(분석 실패: {exc})"

        return {
            "image_path": str(image_path),
            "exists": exists,
            "description": description,
        }

    def unload(self) -> None:
        """다음 모델(Llama-3) 로드를 위해 LLaVA를 메모리에서 즉시 내린다."""
        try:
            ollama.generate(model=self.model, prompt="", keep_alive=0)
            print(f"[LlavaAnalyzer] '{self.model}' 모델을 메모리에서 언로드했습니다.")
        except Exception as exc:  # noqa: BLE001
            print(f"[LlavaAnalyzer] 모델 언로드 중 오류 발생 (무시 가능): {exc}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("사용법: python -m modules.inference.llava_analyzer <질문> <이미지_경로>")
    else:
        analyzer = LlavaAnalyzer()
        print(analyzer.analyze(sys.argv[1], sys.argv[2]))
