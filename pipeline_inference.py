"""
pipeline_inference.py
======================
[3번 방] 로컬 LLM/VLM 추론 및 검증 파이프라인

역할:
    - 로컬에 구동 중인 Ollama 서버(llama3, llava 모델)에게 프롬프트(및 이미지)를
      전달하고 응답을 받아온다.
    - llava(비전 모델)로 프레임 이미지를 설명/검증하고, llama3(텍스트 모델)로
      후처리(요약, 정제, 재검증 등)를 수행하는 하이브리드 구조를 염두에 둔다.

사전 준비 사항 (로컬 환경):
    1. Ollama 앱/서버가 실행 중이어야 한다. (예: `ollama serve` 또는 macOS 앱 실행)
    2. 아래 모델들이 미리 pull 되어 있어야 한다.
         $ ollama pull llama3
         $ ollama pull llava

메모리 관리 (M3 Pro 18GB 환경):
    - llama3와 llava를 동시에 메모리에 상주시키면 18GB 한도에서 부담이 될 수 있으므로,
      Ollama의 자동 모델 로드/언로드(keep_alive) 동작에 맡기고,
      한 번에 하나의 모델만 활발히 사용하는 순차 호출 방식을 기본으로 한다.
"""

from __future__ import annotations

from pathlib import Path

import ollama

import config

# ------------------------------------------------------------------
# 모델 이름 상수
# ------------------------------------------------------------------
LLAMA_MODEL = "llama3"
LLAVA_MODEL = "llava"


def ask_llama3(prompt: str, model: str = LLAMA_MODEL) -> str:
    """
    텍스트 전용 LLM(llama3)에게 프롬프트를 던지고 응답 텍스트를 받아온다.

    Args:
        prompt: 사용자 프롬프트
        model: 사용할 Ollama 모델 이름 (기본값: llama3)

    Returns:
        모델의 응답 텍스트
    """
    response = ollama.chat(
        model=model,
        messages=[
            {"role": "user", "content": prompt},
        ],
    )
    return response["message"]["content"]


def ask_llava(prompt: str, image_path: Path | str, model: str = LLAVA_MODEL) -> str:
    """
    비전-언어 모델(llava)에게 이미지와 프롬프트를 함께 던지고 응답을 받아온다.

    Args:
        prompt: 이미지에 대해 묻고 싶은 질문/지시문 (예: "이 프레임에 무엇이 보이나요?")
        image_path: 분석할 이미지 파일 경로 (예: storage/frames/ 안의 Key-frame)
        model: 사용할 Ollama 모델 이름 (기본값: llava)

    Returns:
        모델의 응답 텍스트
    """
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"이미지 파일을 찾을 수 없습니다: {image_path}")

    response = ollama.chat(
        model=model,
        messages=[
            {
                "role": "user",
                "content": prompt,
                "images": [str(image_path)],
            },
        ],
    )
    return response["message"]["content"]


def run_inference_smoke_test(frames_dir: Path = config.FRAMES_DIR) -> None:
    """
    3번 방 파이프라인의 진입점 (연결 확인용 스모크 테스트).

    Ollama 서버가 정상 응답하는지, llama3/llava 모델이 사용 가능한지
    간단한 프롬프트로 확인한다. (아직 실제 프레임 데이터를 연동하기 전 단계)
    """
    print("[inference] llama3 연결 테스트 중...")
    try:
        text_response = ask_llama3("한 문장으로 너 자신을 소개해줘.")
        print(f"[inference] llama3 응답: {text_response}")
    except Exception as exc:  # noqa: BLE001 - 뼈대 단계에서는 광범위하게 캐치 후 로깅
        print(f"[inference] llama3 호출 실패 (Ollama 서버/모델 확인 필요): {exc}")

    print("[inference] llava 연결 테스트 중...")
    # TODO: 실제 파이프라인에서는 2번 방에서 색인된 Key-frame 경로를 사용한다.
    sample_image_path = frames_dir / "sample.jpg"
    if not sample_image_path.exists():
        print(
            "[inference] 테스트용 샘플 이미지가 없어 llava 호출을 건너뜁니다. "
            f"(확인 경로: {sample_image_path})"
        )
        return

    try:
        vision_response = ask_llava("이 이미지에 무엇이 보이는지 설명해줘.", sample_image_path)
        print(f"[inference] llava 응답: {vision_response}")
    except Exception as exc:  # noqa: BLE001
        print(f"[inference] llava 호출 실패 (Ollama 서버/모델 확인 필요): {exc}")


if __name__ == "__main__":
    run_inference_smoke_test()
