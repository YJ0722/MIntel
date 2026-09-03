# MIntel
Mintel (multimodal intelligence search)

로컬 영상에서 Key-frame을 추출해 CLIP/YOLOv8/EasyOCR로 가공한 뒤 Qdrant에 색인하고,
자연어 질문에 대해 Dense(CLIP)+Sparse(객체/텍스트) 하이브리드 검색과 LLaVA/Llama-3
검증을 거쳐 답변을 생성하는 하이브리드 비전 인덱싱 및 검증 파이프라인입니다.

## 문서

- [D002. 개발환경 구축 및 설치 가이드](./docs/D002_개발환경_구축_및_설치_가이드.md)
- [D003. 데이터 규격 및 폴더 구조 정의](./docs/D003_데이터_규격_및_폴더_구조_정의.md)
- [D004. 실행 및 테스트 방법](./docs/D004_실행_및_테스트_방법.md) (작성 예정)
