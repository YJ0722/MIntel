"""
modules/inference/llama_responder.py
======================================
Llama-3(텍스트 전용 LLM) 기반 최종 답변 생성 모듈.

역할:
    - LlavaAnalyzer가 후보 프레임별로 판단한 "존재 여부(true/false) + 상세 묘사"
      목록과 사용자 질문을 함께 받아, 이를 종합해 사람이 읽기 좋은 자연스러운
      한국어 문장 하나로 최종 답변을 생성한다.
    - 이 모듈은 이미지나 Qdrant를 전혀 알지 못하며, 순수하게 "텍스트로 정리된
      시각 검증 결과 -> 자연어 답변" 변환만 담당한다.

메모리 관리 (M3 Pro 18GB 환경):
    - LlavaAnalyzer가 언로드된 뒤에 로드되도록 파이프라인 상에서 순서를 강제하고,
      keep_alive로 응답 후 유지 시간을 제어한다. 답변 생성이 끝나면 unload()로
      메모리를 정리할 수 있다.
"""

from __future__ import annotations

from pathlib import Path

import ollama

DEFAULT_MODEL_NAME = "llama3"
DEFAULT_KEEP_ALIVE = "5m"


class LlamaResponder:
    """LLaVA 검증 결과를 종합해 최종 한국어 답변을 생성하는 Llama-3 래퍼.

    사용 예:
        responder = LlamaResponder()
        answer = responder.generate_answer(
            "빨간 차가 보이는 장면 있어?",
            [{"image_path": "...", "exists": True, "description": "..."}],
        )
        responder.unload()
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL_NAME,
        keep_alive: str = DEFAULT_KEEP_ALIVE,
    ) -> None:
        """
        Args:
            model: 사용할 Ollama 텍스트 모델 이름 (기본값: llama3)
            keep_alive: 추론 후 모델을 메모리에 유지할 시간 (기본값: "5m")
        """
        self.model = model
        self.keep_alive = keep_alive
        print(f"[LlamaResponder] '{self.model}' 모델 사용 준비 완료 (keep_alive={self.keep_alive}).")

    def generate_answer(self, user_query: str, visual_findings: list[dict]) -> str:
        """
        LLaVA의 프레임별 검증 결과를 종합해 사용자 질문에 대한 최종 답변을 만든다.

        Args:
            user_query: 사용자의 원래 질문
            visual_findings: LlavaAnalyzer.analyze()가 반환한 딕셔너리들의 리스트
                (각 원소: {"image_path", "exists", "description", ...})

        Returns:
            한국어 자연어 최종 답변 문자열. Ollama 호출에 실패하면 사용자에게
            보여줄 안내 문구를 대신 반환한다 (예외를 던지지 않는다).
        """
        findings_text = self._format_findings(visual_findings)

        '''
        system_prompt = (
            "너는 한국어로만 답변하는 어시스턴트야. 영어 단어나 문장, 번역문을 "
            "절대 섞어 쓰지 마. 서론이나 부가 설명 없이, 사용자에게 바로 보여줄 "
            "최종 답변 문장만 출력해."
        )
        '''
        '''
        system_prompt = (
            "당신은 비디오 검색 시스템의 최종 답변 생성기입니다. "
            "전달받은 LLaVA의 이미지 분석 결과에 존재하는 모든 후보 프레임(1번, 2번, 3번 등)을 "
            "단 하나도 누락하지 말고 각각 차례대로 언급하며 사용자의 질문에 답변하세요. "
            "확실한 정보만 바탕으로 한국어로 친절하게 답변하세요."
        )
        '''
        system_prompt = (
            "너는 비디오 검색 시스템의 최종 답변 생성기야. "
            "전달받은 LLaVA의 이미지 분석 결과에 존재하는 모든 후보 프레임(1번, 2번, 3번 등)을 "
            "단 하나도 누락하지 말고 각각 차례대로 언급하며 사용자의 질문에 답변해. "
            "영어 단어나 문장, 번역문을 섞어쓰지 말고 한국어로만 답변해."
            "답변은 처음에 '대상프레임 : (이미지 파일명)'과 같이 먼저 표시하고, 그 뒤에 설명을 추가해."
        )
        prompt = (
            f'사용자 질문: "{user_query}"\n\n'
            "아래는 영상에서 검색된 후보 장면들을 LLaVA 비전 모델이 분석한 결과야:\n"
            f"{findings_text}\n\n"
            "위 분석 결과를 종합해서 사용자 질문에 대한 최종 답변을 자연스러운 "
            "한국어 문장으로 작성해줘. 영어 단어나 문장, 번역문을 섞어쓰지 말고 "
            "한국어로만 답변해. 모든 후보 프레임은 단 하나도 누락하지 말고 확인해."
            "실제로 존재하는 장면이 있다면 어떤 내용인지 "
            "추가 설명 시 구체적으로 설명해."
        )

        try:
            response = ollama.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                keep_alive=self.keep_alive,
            )
            return response["message"]["content"].strip()
        except Exception as exc:  # noqa: BLE001 - Ollama 연결/모델 오류를 광범위하게 캐치
            print(f"[LlamaResponder] 최종 답변 생성 중 오류 발생: {exc}")
            return "죄송합니다. 답변을 생성하는 중 오류가 발생했습니다. Ollama 서버 상태를 확인해주세요."

    @staticmethod
    def _format_findings(visual_findings: list[dict]) -> str:
        """LLaVA 분석 결과 리스트를 프롬프트에 넣을 사람이 읽기 쉬운 텍스트로 정리한다."""
        if not visual_findings:
            return "검색된 후보 장면이 없습니다."

        lines = []
        for i, finding in enumerate(visual_findings, start=1):
            exists_label = "실제로 존재함" if finding.get("exists") else "존재하지 않음"
            lines.append(
                f"{i}번 후보 ({Path(finding['image_path']).name if finding.get('image_path') else '?'}):\n"
                f"   - 판정: {exists_label}\n"
                f"   - 상세 설명: {finding.get('description', '(설명 없음)')}"
            )
        return "\n".join(lines)

    def unload(self) -> None:
        """다음 모델 로드를 위해 Llama-3를 메모리에서 즉시 내린다."""
        try:
            ollama.generate(model=self.model, prompt="", keep_alive=0)
            print(f"[LlamaResponder] '{self.model}' 모델을 메모리에서 언로드했습니다.")
        except Exception as exc:  # noqa: BLE001
            print(f"[LlamaResponder] 모델 언로드 중 오류 발생 (무시 가능): {exc}")


if __name__ == "__main__":
    responder = LlamaResponder()
    sample_answer = responder.generate_answer(
        "빨간 차가 보이는 장면 있어?",
        [
            {
                "image_path": "sample_frame_000000.jpg",
                "exists": True,
                "description": "도로 위에 빨간 승용차 한 대가 주차되어 있다.",
            }
        ],
    )
    print(sample_answer)
