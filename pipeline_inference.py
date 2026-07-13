"""
pipeline_inference.py
======================
[3번 방] Inference Layer - 하이브리드 검증 및 최종 답변 생성

전체 흐름 (다이어그램 기준):
    사용자 질문(user_query)
        -> SearchEngine (modules/inference/search_engine.py)
           : CLIP 텍스트 임베딩으로 Qdrant("mintel_collection")에서 후보
             Key-frame top_k개를 검색한다.
        -> LlavaAnalyzer (modules/inference/llava_analyzer.py)
           : 각 후보 프레임 이미지를 LLaVA에게 보여주고, 사용자가 찾는
             상황/사물이 실제로 존재하는지(true/false) + 상세 묘사를 얻는다.
        -> LlamaResponder (modules/inference/llama_responder.py)
           : LLaVA 검증 결과와 사용자 질문을 종합해 Llama-3가 자연스러운
             한국어 최종 답변을 생성한다.

이 파일은 위 3개 모듈을 순서대로 호출하는 오케스트레이터(제어 타워) 역할만
담당하며, 각 모델의 세부 구현(프롬프트, 응답 파싱, 메모리 관리 등)은 해당
모듈에 위임한다. Ingestion/Storage Layer가 modules/ingestion, modules/storage를
오케스트레이션하는 것과 동일한 구조다.

사전 준비 사항 (로컬 환경):
    1. Ollama 앱/서버가 실행 중이어야 한다. (예: `ollama serve` 또는 macOS 앱 실행)
    2. 아래 모델들이 미리 pull 되어 있어야 한다.
         $ ollama pull llama3
         $ ollama pull llava
    3. 2번 방(pipeline_storage)에서 Qdrant DB(config.DB_DIR)에 Key-frame이
       이미 적재되어 있어야 검색할 후보가 존재한다.

메모리 관리 (M3 Pro 18GB 환경):
    - LLaVA로 모든 후보 프레임 분석을 마친 뒤 LlavaAnalyzer.unload()로 즉시
      메모리에서 내리고, 그 다음에 Llama-3(LlamaResponder)를 로드한다. 두
      대형 모델을 동시에 메모리에 상주시키지 않고 순차적으로 로드/언로드해
      18GB 한도에서 안전하게 동작하도록 한다.
"""

from __future__ import annotations

from pathlib import Path

import ollama

import config
from modules.inference.llama_responder import LlamaResponder
from modules.inference.llava_analyzer import LlavaAnalyzer
from modules.inference.search_engine import SearchEngine

DEFAULT_TOP_K = 3

# ------------------------------------------------------------------
# 저수준(low-level) 헬퍼: Ollama 연결 확인용 스모크 테스트에서 사용한다.
# 실제 파이프라인(run_inference_pipeline)은 이 함수들을 쓰지 않고,
# LlavaAnalyzer/LlamaResponder가 각자 필요한 프롬프트/응답 처리를 직접 담당한다.
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
    Ollama 서버 연결 확인용 스모크 테스트.

    Ollama 서버가 정상 응답하는지, llama3/llava 모델이 사용 가능한지 간단한
    프롬프트로 확인한다. 실제 검색/검증 흐름은 run_inference_pipeline()을
    사용한다.
    """
    print("[inference] llama3 연결 테스트 중...")
    try:
        text_response = ask_llama3("한 문장으로 너 자신을 소개해줘.")
        print(f"[inference] llama3 응답: {text_response}")
    except Exception as exc:  # noqa: BLE001 - 뼈대 단계에서는 광범위하게 캐치 후 로깅
        print(f"[inference] llama3 호출 실패 (Ollama 서버/모델 확인 필요): {exc}")

    print("[inference] llava 연결 테스트 중...")
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


def run_inference_pipeline(
    user_query: str,
    top_k: int = DEFAULT_TOP_K,
    db_path: Path = config.DB_DIR,
) -> str:
    """
    3번 방(Inference Layer) 파이프라인의 진입점.

    [Qdrant 검색 -> LLaVA 이미지 검증 -> Llama-3 최종 답변 생성] 흐름을
    순서대로 실행한다.

    Args:
        user_query: 사용자의 자연어 질문 (예: "빨간 차가 보이는 장면 있어?")
        top_k: Qdrant에서 검색할 후보 프레임 개수 (기본값: 3)
        db_path: Qdrant 로컬 DB 폴더 경로 (기본값: config.DB_DIR)

    Returns:
        Llama-3가 생성한 최종 한국어 답변 문자열. 후보 프레임을 하나도 찾지
        못하면 안내 문구를 대신 반환한다.
    """
    print(f'\n[Inference] 사용자 질문: "{user_query}"')

    print("[Inference] Qdrant에서 CLIP 임베딩 기반으로 후보 장면을 검색 중입니다...")
    search_engine = SearchEngine(db_path=db_path)
    candidates = search_engine.search(user_query, top_k=top_k)

    if not candidates:
        print("[Inference] 후보 장면을 찾지 못해 파이프라인을 종료합니다.")
        return "죄송합니다. 질문과 관련된 장면을 영상 데이터에서 찾지 못했습니다."

    print(f"[Inference] LLaVA 모델이 후보 장면 {len(candidates)}개를 정밀 분석 중입니다...")
    llava_analyzer = LlavaAnalyzer()
    visual_findings = []
    for candidate in candidates:
        finding = llava_analyzer.analyze(user_query, candidate["frame_path"])
        finding["payload"] = candidate["payload"]
        visual_findings.append(finding)

        exists_label = "존재함" if finding["exists"] else "존재하지 않음"
        print(f"      - {Path(candidate['frame_path']).name}: {exists_label}")

    # Llama-3를 로드하기 전에 LLaVA를 메모리에서 내려 18GB 환경의 부담을 줄인다.
    llava_analyzer.unload()

    print("[Inference] Llama-3 모델이 최종 답변을 생성 중입니다...")
    llama_responder = LlamaResponder()
    final_answer = llama_responder.generate_answer(user_query, visual_findings)
    llama_responder.unload()

    print(f"[Inference] 최종 답변:\n{final_answer}")
    return final_answer


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    else:
        query = "차가 보이는 장면을 찾아줘"

    run_inference_pipeline(query)
