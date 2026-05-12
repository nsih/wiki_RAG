# CSU Wiki AI 시스템 아키텍처

> Hybrid Search 도입 전 시점 기준
> 작성일: 2026-05-12

---

## 1. 개요

사내 행정 문서(교무·학사·기숙사·전산) 기반의 한국어 RAG 챗봇.
Wiki.js를 지식 베이스로, ChromaDB를 벡터 저장소로, LM Studio를 LLM 런타임으로 사용한다.

## 2. 컴포넌트 스택

| 레이어 | 기술 | 역할 |
|---|---|---|
| 프론트엔드 | Streamlit | 챗봇 UI, PDF 업로드 UI |
| 위키 | Wiki.js (GraphQL API) | 문서 저장소, 디렉토리별 API 토큰 권한 분리 |
| LLM 런타임 | LM Studio (OpenAI 호환 endpoint) | `/v1/chat/completions` |
| LLM 모델 | Gemma 3n E4B (`gemma-3n-e4b-it-text`) | 한국어 응답 생성 |
| 임베딩 모델 | jhgan/ko-sroberta-multitask | 한국어 특화 sentence-BERT, 512 토큰 한계 |
| 벡터 DB | ChromaDB (PersistentClient, 로컬 디스크) | 의미 기반 검색 |
| PDF 추출 | PyMuPDF + pymupdf4llm | 표·레이아웃 보존 마크다운 변환 |
| 형태소 분석 | (없음) | 현재 BM25/하이브리드 미적용 |

## 3. 하드웨어

- **AI 워커**: Intel Core Ultra 7 265 + 16GB RAM + Intel Arc iGPU
- **추론 방식**: CPU + iGPU 오프로드 (LM Studio Vulkan 백엔드)
- **네트워크**: 사내 LAN 노출 (LM Studio `Serve on Local Network` 활성)

## 4. 파일 구조

```
project/
├── app.py                      # Streamlit 메인 (Search AI + PDF→Wiki 모드)
├── wiki_builder.py             # PDF 추출 + Wiki.js GraphQL 호출
├── indexer.py                  # Wiki.js → ChromaDB 배치 인덱싱
├── chunker.py                  # 텍스트 청킹 (app.py와 indexer.py 공용)
├── chroma_db/                  # ChromaDB 영속 저장소
└── .streamlit/secrets.toml     # 환경 설정
```

## 5. 두 가지 동작 모드

### 5-1. Search AI 모드 (사용자 질의 응답)

```
사용자 질문
  → ChromaDB.query(query_texts=[질문], n_results=3)
  → 상위 청크 3개를 [참고 문서] 컨텍스트로 결합
  → LM Studio /v1/chat/completions 호출 (system prompt + 참고 문서 + 질문)
  → LLM 응답 + 참고 문서 제목을 출처로 표시
```

**현재 검색 방식**: **순수 벡터 검색 (의미 기반)** 단일 채널.

### 5-2. PDF → Wiki Data 모드 (문서 업로드)

```
PDF 업로드
  → PyMuPDF (pymupdf4llm.to_markdown)로 마크다운 추출
  → 디렉토리·제목 입력, 중복/유사 페이지 검사
  → Wiki.js GraphQL `pages.create` 또는 `pages.update`
  → 같은 마크다운을 chunker로 분할하여 ChromaDB에 즉시 인덱싱
```

**참고**: LLM 마크다운 정제 단계는 **제거됨** (PyMuPDF 출력 품질이 충분, 정제는 수십 분 소요 대비 가치 미미).

## 6. 데이터 흐름 핵심 사실

### 6-1. ChromaDB 스키마

- **컬렉션명**: `wiki_knowledge` (기본값, secrets에서 변경 가능)
- **청크 ID 형식**: `page_{page_id}_chunk_{i}`
- **메타데이터**: `{"page_id": int, "title": str, "path": str}`
- **임베딩**: jhgan/ko-sroberta-multitask, 768차원

### 6-2. Wiki.js 태그 규약

- **`pdf` 태그**: app.py에서 PDF 업로드로 만든 페이지에 자동 부착
- **`indexer.py`의 스킵 판단**: `pdf` 태그는 재인덱싱 안 함 (이미 app.py에서 인덱싱했으니 중복 방지)
- **`updated` 태그**: 업데이트 시 자동 부착
- 사용자가 수동으로 추가한 태그는 update 시 보존됨

### 6-3. 인덱싱 일관성

- **app.py의 즉시 인덱싱**과 **indexer.py의 배치 인덱싱** 양쪽 모두 `chunker.chunk_text`를 사용 → 청크 구조 일관성 보장
- indexer.py 실행 시 ChromaDB에서 Wiki.js에 없는 page_id는 자동 정리 (orphan cleanup)

## 7. LLM 통신 규격

### 엔드포인트

```
http://{AI_WORKER_IP}:1234/v1/chat/completions
```

### 페이로드 (Search AI)

```json
{
  "model": "gemma-3n-e4b-it-text",
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."}
  ],
  "stream": false,
  "temperature": 0.2,
  "max_tokens": 1024
}
```

### 인코딩 주의사항

- 응답 JSON은 자동 UTF-8 처리됨 (`requests`가 `application/json; charset=utf-8` 자동 인식)
- 스트리밍 SSE 응답은 별도 처리 불필요 (현재 비스트리밍 모드)

## 8. 의도적으로 단순화된 부분

| 항목 | 현재 상태 | 의사결정 사유 |
|---|---|---|
| LLM 마크다운 정제 | 제거 | PyMuPDF 출력으로 충분 |
| 이미지 업로드 | 미지원 (플레이스홀더만) | Wiki.js GraphQL이 이미지 미지원, 옵션들 모두 trade-off 큼 |
| 시스템 프롬프트 | 일반론 (간단) | 도메인 특화 강화 여지 있음 |
| 검색 방식 | 순수 벡터 | **Hybrid 도입 예정** |
| `n_results` | 3 | 5로 늘리는 여지 있음 |

## 9. 현재 RAG 검색의 알려진 약점

이게 Hybrid Search 도입 동기다.

| 약점 | 예시 |
|---|---|
| 고유명사 정확 매칭 약함 | 학번, 규정 번호, 부서명 등 |
| 짧은 키워드 질의에서 정확도↓ | "기숙사 신청" 같은 2~3단어 검색 |
| 한국어 형태소 미적용 | "신청한다", "신청합니다", "신청"이 분리됨 |
| 정확한 문구 검색 불가 | 인용구나 규정 원문 검색에 약함 |

## 10. 환경 설정 (secrets.toml)

```toml
# Wiki.js
WIKI_BASE_URL = "http://wiki.example.local"
WIKI_API_TOKEN = "..."

# LM Studio
AI_WORKER_IP = "100.100.91.44"
AI_WORKER_PORT = 1234
AI_MODEL_NAME = "gemma-3n-e4b-it-text"

# ChromaDB
CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "wiki_knowledge"
```

## 11. 운영 원칙 (지금까지 확립된)

- **안정성 > 신기능**: 동작하는 코드는 건드리지 않는다
- **단순성 > 정교함**: 5분에 끝나는 정답이 있으면 3시간짜리 정답을 안 쓴다
- **로컬 우선**: 외부 SaaS 의존 최소화 (사내망에서 완결)
- **권한 분리**: Wiki.js API 토큰을 디렉토리별로 분리하여 사고 영향 최소화 (현재는 정보전산원 디렉토리만 존재)

## 12. 마이그레이션 히스토리

### 2026-05 — Ollama → LM Studio 전환

| Before | After |
|---|---|
| Ollama + Qwen 2.5 7B | LM Studio + Gemma 3n E4B |
| LLM 마크다운 정제 (수십 분) | PyMuPDF 직접 추출 (수 초) |
| `/api/chat` | `/v1/chat/completions` (OpenAI 호환) |
| 순차 청크 LLM 호출 | 단일 PDF 추출 호출 |

**주요 결정 사항**:
- Gemma 4 E4B는 강제 thinking 모드 문제로 포기, 3n E4B로 결정
- PDF 마크다운 LLM 정제는 PyMuPDF 출력 품질이 충분해 제거
- 이미지 업로드는 Wiki.js GraphQL 미지원으로 포기 (플레이스홀더 표시만)

## 13. 다음 단계 (Roadmap)

### 우선순위 높음
- [ ] **Hybrid Search 도입** (BM25 + 벡터, RRF 결합)
- [ ] 시스템 프롬프트 도메인 특화 + 환각 억제 강화
- [ ] `n_results` 3 → 5 증가 검토

### 우선순위 중간
- [ ] LM Studio 자동 시작 설정 (Windows 서비스화)
- [ ] Streamlit 서비스화 (nssm 등)
- [ ] 파일 기반 로깅 (`app.log`)

### 장기
- [ ] AI 모델 업그레이드 (젬마3 → 4)
- [ ] 임베딩 모델 업그레이드 (jhgan → bge-m3) — 전체 재인덱싱 필요
- [ ] 사용 통계 대시보드

---
