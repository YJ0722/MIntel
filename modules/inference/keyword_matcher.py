"""
modules/inference/keyword_matcher.py
======================================
사용자 질문(한국어 자연어)에서 Sparse 검색 신호(YOLOv8 객체 클래스 / OCR 텍스트
힌트)를 추출하는 모듈.

역할:
    - SearchEngine이 "Dense(CLIP 벡터 유사도) + Sparse(객체/글자 필터링)를 결합한
      1-Stage 하이브리드 검색"을 하기 위해, 사용자 질문에서 두 가지 Sparse 신호를
      뽑아낸다.
        1. extract_object_classes(): 질문에 등장한 단어를 YOLOv8n(COCO 80종)이
           실제로 탐지할 수 있는 클래스 이름으로 매핑한다.
           (예: "신호등이 보이는 장면" -> ["traffic light"])
        2. extract_text_hints(): 질문에 등장한 영문 대문자 토큰(표지판/간판 등에서
           EasyOCR이 그대로 인식했을 가능성이 높은 문자열)을 그대로 추출한다.
           (예: "STOP 표지판 보여줘" -> ["STOP"])
    - 이 모듈은 특정 객체(예: "차")에만 하드코딩되어 있지 않고, COCO_KOREAN_MAP에
      등록된 어떤 한국어/영어 키워드든 코드 수정 없이 그대로 필터링에 활용된다.
      즉 "신호등", "사람", "강아지"처럼 사용자 질문만 바뀌어도 자동으로 대응되는
      키워드는 그대로 적용된다.
    - YOLOv8n이 원래 탐지할 수 없는 대상(COCO 80종 밖의 사물)은 매핑 자체가
      존재하지 않으므로, 이 경우 자동으로 Sparse 신호가 비어 반환되어
      Dense(CLIP) 검색 결과만 사용하게 된다 (SearchEngine이 처리).
"""

from __future__ import annotations

import re

# ------------------------------------------------------------------
# YOLOv8n(COCO 80종) 클래스 <- 한국어/영어 키워드 매핑.
# ultralytics YOLO("yolov8n.pt").names 기준 80개 클래스를 모두 커버한다.
# 하나의 클래스에 여러 한국어 동의어를 등록해두면, 사용자가 어떤 표현을
# 쓰든(예: "차"/"자동차"/"승용차") 코드 수정 없이 동일한 클래스로 매핑된다.
# ------------------------------------------------------------------
COCO_KOREAN_MAP: dict[str, list[str]] = {
    "person": ["사람", "사람들", "행인", "보행자", "person"],
    "bicycle": ["자전거", "bicycle"],
    "car": ["차", "차량", "자동차", "승용차", "세단", "car"],
    "motorcycle": ["오토바이", "모터사이클", "motorcycle"],
    "airplane": ["비행기", "항공기", "airplane"],
    "bus": ["버스", "bus"],
    "train": ["기차", "열차", "train"],
    "truck": ["트럭", "화물차", "truck"],
    "boat": ["보트", "배", "선박", "boat"],
    "traffic light": ["신호등", "traffic light"],
    "fire hydrant": ["소화전", "fire hydrant"],
    "stop sign": ["정지 표지판", "정지표지판", "stop", "stop sign"],
    "parking meter": ["주차 미터기", "parking meter"],
    "bench": ["벤치", "bench"],
    "bird": ["새", "bird"],
    "cat": ["고양이", "cat"],
    "dog": ["강아지", "개", "dog"],
    "horse": ["말", "horse"],
    "sheep": ["양", "sheep"],
    "cow": ["소", "cow"],
    "elephant": ["코끼리", "elephant"],
    "bear": ["곰", "bear"],
    "zebra": ["얼룩말", "zebra"],
    "giraffe": ["기린", "giraffe"],
    "backpack": ["배낭", "가방", "backpack"],
    "umbrella": ["우산", "umbrella"],
    "handbag": ["핸드백", "handbag"],
    "tie": ["넥타이", "tie"],
    "suitcase": ["여행 가방", "suitcase"],
    "frisbee": ["frisbee"],
    "skis": ["스키", "skis"],
    "snowboard": ["스노보드", "snowboard"],
    "sports ball": ["공", "sports ball"],
    "kite": ["연", "kite"],
    "baseball bat": ["야구 배트", "baseball bat"],
    "baseball glove": ["야구 글러브", "baseball glove"],
    "skateboard": ["스케이트보드", "skateboard"],
    "surfboard": ["서프보드", "surfboard"],
    "tennis racket": ["테니스 라켓", "tennis racket"],
    "bottle": ["병", "bottle"],
    "wine glass": ["와인잔", "wine glass"],
    "cup": ["컵", "cup"],
    "fork": ["포크", "fork"],
    "knife": ["칼", "knife"],
    "spoon": ["숟가락", "spoon"],
    "bowl": ["그릇", "bowl"],
    "banana": ["바나나", "banana"],
    "apple": ["사과", "apple"],
    "sandwich": ["샌드위치", "sandwich"],
    "orange": ["오렌지", "orange"],
    "broccoli": ["브로콜리", "broccoli"],
    "carrot": ["당근", "carrot"],
    "hot dog": ["핫도그", "hot dog"],
    "pizza": ["피자", "pizza"],
    "donut": ["도넛", "donut"],
    "cake": ["케이크", "cake"],
    "chair": ["의자", "chair"],
    "couch": ["소파", "couch"],
    "potted plant": ["화분", "potted plant"],
    "bed": ["침대", "bed"],
    "dining table": ["식탁", "dining table"],
    "toilet": ["화장실", "변기", "toilet"],
    "tv": ["텔레비전", "tv"],
    "laptop": ["노트북", "laptop"],
    "mouse": ["마우스", "mouse"],
    "remote": ["리모컨", "remote"],
    "keyboard": ["키보드", "keyboard"],
    "cell phone": ["휴대폰", "핸드폰", "스마트폰", "cell phone"],
    "microwave": ["전자레인지", "microwave"],
    "oven": ["오븐", "oven"],
    "toaster": ["토스터", "toaster"],
    "sink": ["세면대", "싱크대", "sink"],
    "refrigerator": ["냉장고", "refrigerator"],
    "book": ["책", "book"],
    "clock": ["시계", "clock"],
    "vase": ["꽃병", "vase"],
    "scissors": ["가위", "scissors"],
    "teddy bear": ["테디 베어", "인형", "teddy bear"],
    "hair drier": ["헤어드라이어", "hair drier"],
    "toothbrush": ["칫솔", "toothbrush"],
}

# 영문 대문자 토큰(2자 이상)을 표지판/간판 텍스트 힌트로 취급한다.
# (예: "STOP 표지판이 보이는 장면" -> "STOP")
_TEXT_HINT_PATTERN = re.compile(r"[A-Z]{2,}")


def extract_object_classes(user_query: str) -> list[str]:
    """
    사용자 질문에서 YOLOv8n(COCO 80종)이 실제로 탐지 가능한 클래스 이름을 추출한다.

    COCO_KOREAN_MAP에 등록된 키워드라면 어떤 것이든(예: "신호등", "사람", "강아지")
    코드 수정 없이 그대로 매칭되며, 매핑에 없는 단어는 무시된다.

    Args:
        user_query: 사용자의 자연어 질문 (예: "신호등이 보이는 장면을 찾아줘")

    Returns:
        매칭된 YOLO 클래스 이름 리스트 (중복 제거, 매칭 순서 유지).
        아무 키워드도 매칭되지 않으면 빈 리스트.
    """
    matched_classes: list[str] = []
    for class_name, keywords in COCO_KOREAN_MAP.items():
        if any(keyword in user_query for keyword in keywords):
            matched_classes.append(class_name)
    return matched_classes


def extract_text_hints(user_query: str) -> list[str]:
    """
    사용자 질문에서 표지판/간판 등 OCR 텍스트와 매칭될 만한 영문 대문자 토큰을 추출한다.

    Args:
        user_query: 사용자의 자연어 질문 (예: "STOP 표지판 앞에 정차한 차 있어?")

    Returns:
        추출된 텍스트 힌트 리스트 (중복 제거). 없으면 빈 리스트.
    """
    hints = _TEXT_HINT_PATTERN.findall(user_query)
    # 순서를 유지하며 중복만 제거
    return list(dict.fromkeys(hints))


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("사용법: python -m modules.inference.keyword_matcher <질문>")
    else:
        query = " ".join(sys.argv[1:])
        print(f"objects: {extract_object_classes(query)}")
        print(f"texts:   {extract_text_hints(query)}")
