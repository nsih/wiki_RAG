# CSU Wiki AI — 사내 위키 RAG 챗봇

사내 Wiki.js 문서를 기반으로 한국어 질의응답을 제공하는 RAG(Retrieval-Augmented Generation) 챗봇입니다.
PDF를 위키 페이지로 자동 변환하는 기능도 함께 제공합니다.

---

## 목차

1. [시스템 아키텍처](#시스템-아키텍처)
2. [인프라 사양](#인프라-사양)
3. [프로젝트 구조](#프로젝트-구조)
4. [검색 전략 — Hybrid Search](#검색-전략--hybrid-search)
5. [설치 및 실행](#설치-및-실행)
6. [설정 파일](#설정-파일)
7. [주요 기능](#주요-기능)
8. [데이터 흐름](#데이터-흐름)
9. [운영 유의사항](#운영-유의사항)

---

## 시스템 아키텍처

```
사용자 브라우저
      │
      ▼
 Streamlit (app.py)
      │
      ├─── Hybrid Search ──────────────────────────────────┐
      │         │                                          │
      │    ChromaDB (벡터 검색)          BM25 (bm25_index.pkl)
      │    jhgan/ko-sroberta-multitask    rank_bm25 + tokenizer.py
      │         └─────────── RRF Fusion ──────────────────┘
      │
      ├─── LM Studio (별도 호스트)
      │    Gemma 3n E4B  /v1/chat/completions
      │
      └─── Wiki.js (GraphQL API)
```

---

## 인프라 사양

### Streamlit 앱 서버 (가상화 환경)

| 항목 | 사양 |
|---|---|
| CPU | Intel Xeon E5-2630 v4 (Broadwell, 10코어, AVX2 지원) |
| RAM | **4GB** (스왑 미설정) |
| 환경 | 가상화 (VM) |
| OS | Linux |
| 형태소 분석기 | ❌ 미사용 (메모리 제약 — kiwipiepy/konlpy 제외) |
| 토큰화 방식 | 정규식 + 끝조사 스트리핑 휴리스틱 |

> **메모리 제약 설계 원칙**
> 4GB 환경에서 ChromaDB + SentenceTransformer + BM25 인덱스가 동시에 상주해야 합니다.
> 형태소 분석기(kiwipiepy 등)는 로딩 시 수백 MB를 소비하므로 의도적으로 제외했습니다.
> BM25 재빌드(청크 5,000개 기준)는 약 5~15초가 소요됩니다.

### LLM 호스트 (별도 PC)

| 항목 | 사양 |
|---|---|
| CPU | Intel Core Ultra 7 265 |
| RAM | 16GB |
| GPU | 내장 그래픽 (iGPU) |
| 런타임 | LM Studio |
| 모델 | Gemma 3n E4B (`gemma-3n-e4b-it-text`) |
| API | OpenAI 호환 `/v1/chat/completions` |
| 타임아웃 | 120초 (iGPU 연산 지연 고려) |

---

## 프로젝트 구조

```
.
├── app.py              # Streamlit 메인 (Search AI / PDF→Wiki 모드)
├── wiki_builder.py     # PDF 추출 + Wiki.js GraphQL 뮤테이션
├── indexer.py          # Wiki.js → ChromaDB 배치 인덱싱 + BM25 빌드
├── chunker.py          # chunk_text(text) → list[str]
├── tokenizer.py        # tokenize_ko(text) → list[str]  ← 신규
├── bm25_store.py       # BM25 인덱스 빌드/저장/로드/패치  ← 신규
├── retriever.py        # hybrid_search() — BM25 + 벡터 RRF 융합  ← 신규
├── requirements.txt
├── bm25_index.pkl      # BM25 인덱스 파일 (indexer 실행 후 생성)
└── .streamlit/
    └── secrets.toml
```

---

## 검색 전략 — Hybrid Search

### 개요

벡터 검색(의미 유사도)과 BM25(키워드 빈도) 두 검색기를 결합해 단독 검색 대비 재현율과 정밀도를 함께 높입니다.
행정 문서 특성상 고유명사·코드·날짜 등 키워드 매칭이 중요해 하이브리드가 유리합니다.

### RRF (Reciprocal Rank Fusion)

```
RRF Score = Σ [ 1 / (k + rank_i) ]   (k = 60)
```

각 검색기에서 후보 20개씩 추출 → RRF로 재순위 → 상위 5개를 LLM 컨텍스트에 주입합니다.

### 한국어 토큰화 (tokenizer.py)

형태소 분석기 없이 정규식과 조사 스트리핑으로 처리합니다.

```
정규식 : [가-힣]+|[a-zA-Z]+|[0-9]+
조사 제거: 을/를/이/가/은/는/의/에/과/와/도/로/으로/에서
필터링 : 2글자 미만 토큰 제거
```

`indexer.py`와 `app.py`가 동일한 `tokenize_ko()`를 공유해 인덱스 일관성을 보장합니다.

### BM25 인덱스 관리

| 시점 | 동작 |
|---|---|
| `indexer.py` 배치 실행 | ChromaDB 전체 스캔 → `bm25_index.pkl` 원자적 저장 (`os.replace`) |
| 앱 시작 | `@st.cache_resource`로 한 번만 로드 |
| PDF 업로드 시 | 메모리상 BM25 객체 in-place 패치 (디스크 미반영) |
| 다음 배치 실행 | 디스크 인덱스 정식 재빌드로 정합 회복 |
| 파일 없을 때 | 벡터 단독 검색으로 자동 폴백 |

---

## 설치 및 실행

### 의존성 설치

```bash
pip install -r requirements.txt
```

`requirements.txt` 주요 항목:

```
streamlit
chromadb
sentence-transformers
rank_bm25          # BM25 검색
pymupdf
pymupdf4llm
markdown
beautifulsoup4
requests
```

### BM25 인덱스 초기 빌드

ChromaDB 인덱싱과 BM25 빌드를 함께 수행합니다.

```bash
python indexer.py
```

### 앱 실행

```bash
streamlit run app.py
```

---

## 설정 파일

`.streamlit/secrets.toml` 예시:

```toml
WIKI_BASE_URL   = "https://wiki.example.com"
WIKI_API_TOKEN  = "your_api_token"

CHROMA_PATH     = "./chroma_db"
COLLECTION_NAME = "wiki_knowledge"
BM25_PATH       = "./bm25_index.pkl"

AI_WORKER_IP    = "192.168.x.x"
AI_WORKER_PORT  = 1234
AI_MODEL_NAME   = "gemma-3n-e4b-it-text"
```

---

## 주요 기능

### Search AI 모드

- 사내 위키 기반 한국어 Q&A
- Hybrid Search (BM25 + 벡터 RRF 융합)
- 최근 4턴 대화 이력 유지
- 답변 하단에 출처 문서 표시

### PDF → Wiki Data 모드

- PDF 업로드 → PyMuPDF로 마크다운 추출 (표·레이아웃 보존)
- Wiki.js 경로 중복·유사 문서 감지 후 신규 생성 / 덮어쓰기 선택
- Wiki.js 반영 즉시 ChromaDB·BM25 인덱스 실시간 패치
- `pdf` 태그 자동 부여 → 배치 indexer가 해당 페이지 재색인 스킵

---

## 데이터 흐름

### 인덱싱 배치 (indexer.py)

```
Wiki.js 페이지 목록 조회
  └─ PDF 태그 페이지 → 스킵 (임베딩 보존)
  └─ 일반 페이지
       ├─ 마크다운 정제 (코드블록 제거, 아스키아트 필터링)
       ├─ chunk_text() 청킹
       ├─ ChromaDB 저장
       └─ (전체 완료 후) BM25 인덱스 빌드 → bm25_index.pkl 저장
```

### 검색 (app.py)

```
사용자 질문
  ├─ 벡터 검색 (ChromaDB, candidates=20)
  ├─ BM25 검색 (bm25_index.pkl, candidates=20)
  └─ RRF 융합 → 상위 5개
       └─ LLM 컨텍스트 주입 → 답변 생성
```

---

## 운영 유의사항

- **스왑 미설정** 환경이므로 OOM 발생 시 프로세스가 즉시 종료됩니다. 메모리 사용량을 주기적으로 모니터링하세요.
- BM25 인덱스(`bm25_index.pkl`)가 없는 상태로 앱을 시작하면 벡터 단독 검색으로 폴백됩니다. 첫 배포 시 반드시 `python indexer.py`를 먼저 실행하세요.
- PDF 업로드 직후 BM25 패치는 메모리에만 반영됩니다. 앱 재시작 전에 `indexer.py`를 실행해 디스크 인덱스를 최신화하세요.
- Wiki.js GraphQL API 토큰 만료 시 인덱싱과 PDF 업로드가 모두 중단됩니다.