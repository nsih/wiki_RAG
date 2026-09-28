# Wiki RAG Chat bot

1. 사내 Wiki 기반의 한국어 질의응답
2. PDF, xml -> convert to wiki page

# Search

1. 질의
2. 벡터 검색으로 위키 페이지 3개 선정
3. 알고리즘(BM25)으로 청크 3개 선정
4. 조각 붙여서 LLM에 넘김
5. LLM이 가공해서 출력

# Fine Tuning Model 교체

* jhgan/ko-sroberta-multitask -> models/ft-ep2
* LLM이 만들어낸 위키 기반의 질문 정답 데이터로 학습
* 정답률 67.5%에서 77.5%로 개선
