# 임베딩 모델 파인튜닝 작업계획

> 이 문서는 **계획**이다. 실제 진행 상황은 [PROGRESS.md](./PROGRESS.md)에 기록한다.
> 계획이 바뀌면 이 문서를 고치고, 바꾼 이유는 PROGRESS.md에 남긴다.

작성일: 2026-09-09

---

## 1. 배경과 목표

`rag-poc`(CSU Wiki AI)는 현재 **학습 코드가 전혀 없는 순수 추론 PoC**다.
검색 품질은 `jhgan/ko-sroberta-multitask`(범용 한국어 임베딩)에 전적으로 의존하는데,
이 모델은 교내 위키의 사내 용어·장비명·조직명을 학습한 적이 없다.
그래서 "10호관 무선AP" 같은 질의에서 벡터 검색이 약하고, 실질적으로 BM25가 성능을 떠받치고 있다.

목표는 두 가지이며 **순서가 있다**.

1. **PyTorch의 개념을 밑바닥부터 이해한다** — 텐서 / autograd / nn.Module / 학습 루프.
   남이 짠 `Trainer.train()`을 호출하는 게 아니라 학습 루프를 직접 쓸 수 있는 수준.
2. 그 이해를 이 프로젝트에 적용해 **임베딩 모델을 사내 위키 도메인에 파인튜닝**하고,
   검색 품질이 실제로 올랐는지 **숫자로** 확인한다.

두 목표는 하나로 이어진다. 이 파인튜닝의 핵심 손실 함수(in-batch negative contrastive loss)는
`F.cross_entropy` 한 줄이라, 기초를 제대로 배우면 그대로 구현할 수 있다.
학습용 장난감 예제와 실전 코드가 따로 놀지 않게 커리큘럼을 실제 목표로 수렴시킨다.

---

## 2. 하드웨어 현실 (설계를 지배하는 제약)

| | 이 서버 (`rag-poc` 호스트) | 91.44 (LM Studio 워커) |
|---|---|---|
| CPU | Xeon E5-2630v4, **2 vCPU** | **Core Ultra 7 265K (20코어)** |
| RAM | 3.8 GB (**여유 1.4 GB**) | 미확인 |
| GPU | 없음 | 없음 (내장 Xe만) |
| 접근 | 로컬 | **SSH 22번 차단**, `/v1` HTTP API만 열림 |

### 확인된 사실 (2026-09-09 실측)

- `torch 2.11.0+cu130` 설치됨, `cuda.is_available() == False` → **CPU 학습만 가능**.
  cu130 빌드도 CPU로 잘 돌므로 재설치 불필요.
- Wiki 서버(`100.100.103.213:3000`)는 이 셸에서 응답 없음(`curl` 000) → `indexer.py` 재실행 불가.
- **그러나 학습 코퍼스는 이미 로컬에 있다.** `chroma_db/chroma.sqlite3`에 1,136개 청크가
  본문 + `title`/`path`/`page_id` 메타데이터와 함께 저장돼 있고, Chroma 클라이언트 없이
  순수 `sqlite3`로 읽힌다. **위키 서버 의존성 없음.**
- 91.44는 SSH가 막혀 **자동화된 원격 학습은 불가**. 다만 `/v1/chat/completions`는 살아 있어
  **학습 데이터 생성에는 쓸 수 있다.** 학습 자체를 20코어에서 돌리려면 폴더 수동 복사가 필요하므로,
  스크립트는 그게 가능하도록 이식성 있게(경로 인자화, 상대경로) 작성한다.

### 메모리 계산 — 왜 전체 파인튜닝이 불가능한가

`ko-sroberta-multitask` = `klue/roberta-base` 백본, 12층 / hidden 768 / vocab 32,000 → **약 110M 파라미터**.

전체 파인튜닝 시 AdamW 기준:

```
가중치       110M x 4B  =   440 MB
그래디언트   110M x 4B  =   440 MB
AdamW 상태   110M x 8B  =   880 MB
                          --------
                           1,760 MB   <- 여유 1.4 GB 초과. 불가.
```

그래서 **상위 층만 학습(partial freeze)** 한다. 임베딩 층 + 0~7층 동결, 8~11층만 학습:

```
학습 대상 4개 층 x 7.09M  =  28.4M 파라미터
가중치(전체, 동결분 포함)  =   440 MB
그래디언트  28.4M x 4B     =   114 MB
AdamW 상태  28.4M x 8B     =   227 MB
활성값(batch 8 x seq 128)  =  ~150 MB
                             --------
                              ~930 MB   <- 들어간다.
```

이건 성능 타협이 아니라 오히려 정석에 가깝다. 하위 층은 일반적인 한국어 문법/형태를 담당하고
도메인 적응은 주로 상위 층에서 일어나므로, 소규모 데이터셋에서는 하위 층 동결이 과적합도 막아준다.

스크립트는 `--freeze-below N` 인자를 받는다. 이 서버 `8`, 91.44로 옮기면 `0`(전체 학습).

---

## 3. 전체 개요

| Stage | 목적 | 주요 산출물 | 실행 위치 |
|---|---|---|---|
| **0** | 환경 확인·격리 | `.gitignore` 3줄 | 로컬 |
| **1** | PyTorch 개념 습득 | `torch_lab/` 스크립트 7개 | 로컬 (가벼움) |
| **2** | 학습 데이터 생성 | `chunks.jsonl`, `pairs_*.jsonl` | 로컬 + 91.44 API |
| **3** | 파인튜닝 | `models/ft-ep{n}/` | 로컬 CPU |
| **4** | 평가·재색인 | 지표 비교표, `chroma_db_ft/` | 로컬 |
| **5** | 앱 연동 | `app.py` 등 수정 | 로컬 (**조건부**) |

**Stage 5는 Stage 4에서 개선이 확인될 때만 진행한다. 그전까지 기존 파일은 하나도 수정하지 않는다.**

### 디렉터리 구조

```
rag-poc/
├── FinetuningDocs/         # 이 문서들
│   ├── PLAN.md             # 작업계획 (이 문서)
│   ├── PROGRESS.md         # 진척사항 기록
│   └── ed/                 # 학습 노트 (개념 정리)
│       ├── README.md
│       ├── 01-pytorch-basics.md
│       ├── 02-cuda.md
│       └── 03-pytorch-cuda-relation.md
├── torch_lab/              # Stage 1: 학습용 (앱과 무관, 언제든 삭제 가능)
│   ├── 01_tensor.py
│   ├── 02_autograd.py
│   ├── 03_module_optim.py
│   ├── 04_data.py
│   ├── 05_save_load.py
│   ├── 06_encoder.py
│   └── 07_contrastive.py
└── finetune/               # Stage 2~4: 실전
    ├── export_corpus.py
    ├── gen_queries.py
    ├── evaluate.py
    ├── train_embedder.py
    ├── reindex.py
    └── data/               # 생성물 (.gitignore 대상)
        ├── chunks.jsonl
        ├── pairs.jsonl
        ├── pairs_train.jsonl
        ├── pairs_eval.jsonl
        └── eval_human.jsonl
```

주석·문서화는 기존 코드 스타일(한국어 docstring, `# ── 섹션 ──` 구분선)을 따른다.

---

## 4. Stage 0 — 준비

- `.gitignore`에 `finetune/data/`, `models/`, `chroma_db_ft/` 추가
- CPU 스레드 고정: 모든 스크립트 상단에 `torch.set_num_threads(2)`.
  2코어에서 PyTorch 기본값은 스레드 경합으로 오히려 느려지고 RAM을 더 쓴다.
- **추가 의존성 없음.** 이미 설치된 `torch` / `transformers` / `sentence-transformers` /
  `numpy` / `scikit-learn`만 사용한다.
  `peft`·`datasets`·`accelerate`는 **쓰지 않는다** — 배우는 게 목적인데 추상화 계층이 개념을 가린다.

---

## 5. Stage 1 — PyTorch 기초 (`torch_lab/`)

각 스크립트는 단독 실행 가능하고, 출력이 개념을 스스로 증명하게 만든다.
한국어 주석 + "왜 이렇게 되는가"를 설명하는 print. 각 파일 끝에 **직접 고쳐보는 연습 과제**를 남긴다.

| 파일 | 핵심 개념 | 이 프로젝트와의 연결 |
|---|---|---|
| `01_tensor.py` | shape, dtype, 브로드캐스팅, `matmul` | 코사인 유사도를 3가지 방법으로 계산해 일치 확인 → Chroma의 `distance = 1 - cosine` |
| `02_autograd.py` | `requires_grad`, `backward()`, `.grad` 누적, `no_grad()`/`detach()` | 옵티마이저 없이 손으로 경사하강 (`y=3x+2` 맞히기) |
| `03_module_optim.py` | `nn.Module`, `nn.Linear`, `AdamW`, **표준 학습 루프** | 여기서 익힌 루프 골격 = Stage 3 실전 코드와 동일 구조 |
| `04_data.py` | `Dataset`, `DataLoader`, 배치·셔플, `collate_fn` | 장난감 데이터 아님 — sqlite에서 **실제 위키 청크**를 읽는 Dataset |
| `05_save_load.py` | `state_dict`, `torch.save/load`, `train()` vs `eval()`, 시드 고정 | 체크포인트 저장·복원 검증 |
| `06_encoder.py` | 토크나이저 → `last_hidden_state (B,L,768)` → **mean pooling** → 정규화 | **핵심 아하 지점.** 수동 구현 == `SentenceTransformer.encode()` 를 `allclose`로 증명 |
| `07_contrastive.py` | 유사도 행렬 `(B,B)` → 정답은 대각선 → `F.cross_entropy(sim*scale, arange(B))` | 이 8줄이 `MultipleNegativesRankingLoss`의 전부. 배치가 클수록 유리한 이유까지 |

**목표: Stage 1을 마치면 Stage 3 코드에 새로운 개념이 하나도 없을 것.**

---

## 6. Stage 2 — 학습 데이터 만들기

라벨 데이터가 없다. `(질문, 정답 청크)` 쌍을 LLM으로 합성한다(doc2query / InPars 방식).

| 파일 | 하는 일 | 입력 → 출력 | 주의점 |
|---|---|---|---|
| `export_corpus.py` | sqlite 읽기 전용으로 청크 추출 | `chroma.sqlite3` → `chunks.jsonl` (1,136줄) | 위키 서버·Chroma 클라이언트·임베딩 모델 전부 불필요 |
| `gen_queries.py` | LLM으로 질문 합성 | 청크 → `pairs.jsonl` (~2,200쌍) | `--limit` / `--resume` **필수** |
| (분할) | train/eval 나누기 | → `pairs_train` / `pairs_eval` | **`chunk_id` 기준** 분할 (누수 방지) |

### 세부 사항

- **`export_corpus.py`** — `file:...?mode=ro` URI로 열어 `embedding_metadata`에서
  `chroma:document` / `title` / `path` / `page_id`를 조인. 청크 상당수가 마크다운 표
  (예: "기숙사 무선AP 관리대장")이므로 표 청크 필터링/표시 여부를 옵션으로 둔다.
- **`gen_queries.py`** — 91.44 LM Studio(`qwen/qwen3-4b-2507`)에 청크를 보내
  그 청크로 답할 수 있는 **한국어 질문 2개**를 생성.
  기존 `app.py:103-140`의 호출 규약을 그대로 재사용(`/no_think` 접두사, `temperature`, 단일 user 메시지).
  엔드포인트는 `.streamlit/secrets.toml`에서 읽어 하드코딩하지 않는다.
  91.44도 CPU 추론이라 전량은 1~2시간대 예상 → **중단·재개가 반드시 가능해야 한다**(한 줄씩 append).
  응답이 비었거나 형식이 깨지면 조용히 스킵하고 카운트만 기록.
- **8:2 분할은 반드시 `chunk_id` 기준.** 같은 청크에서 나온 질문이 train과 eval에 갈리면
  평가가 새어 오염된다.
- **`data/eval_human.jsonl`** — 사용자가 **직접 손으로 작성하는** 질문 30~50개.
  스크립트가 아니라 사용자 작업이며, 8절 "정직한 한계" 참고.

---

## 7. Stage 3~5 — 학습 · 평가 · 연동

### Stage 3: `finetune/train_embedder.py`

**Trainer를 쓰지 않고 순수 PyTorch 루프로 작성한다.** Stage 1에서 배운 것을 그대로 쓰는 게 목적이다.

```python
model = AutoModel.from_pretrained(BASE)          # 06_encoder.py에서 익힌 것
for name, p in model.named_parameters():         # partial freeze
    p.requires_grad = not is_frozen(name, args.freeze_below)
opt = AdamW([p for p in model.parameters() if p.requires_grad], lr=2e-5)

for epoch in range(args.epochs):
    for queries, positives in loader:            # 04_data.py의 Dataset/DataLoader
        q = mean_pool(model(**tok(queries)))     # 06
        d = mean_pool(model(**tok(positives)))
        loss = contrastive_loss(q, d, scale=20)  # 07 — 직접 구현한 그 함수
        loss.backward(); opt.step(); opt.zero_grad()   # 02, 03
```

| 항목 | 이 서버 | 91.44 이전 시 | 근거 |
|---|---|---|---|
| `--freeze-below` | **8** (상위 4층만) | 0 (전체) | 2절 메모리 계산 |
| `batch_size` | **8** | 32+ | RAM. 작을수록 negative가 적어 학습 불리 |
| `max_seq_len` | **128** | 256 | 청크 500자가 잘리지만 앞부분이 매칭에 중요 |
| `lr` / warmup | 2e-5 / 10% linear decay | 동일 | 파인튜닝 표준값 |
| `epochs` | 2~3 | 동일 | 데이터가 2천 쌍 수준이라 그 이상은 과적합 |
| OOM 시 후퇴 | `--freeze-below 10` 또는 `batch_size 4` | — | 학습 중 **Streamlit 앱 반드시 종료** |

- **매 에폭 종료 시** eval Recall@5를 찍고 `models/ft-ep{n}/`에 체크포인트 저장.
  마지막 에폭이 최선이라는 보장이 없다.
- 시드 고정, 스텝별 loss 로깅.
- 소요 시간은 **추정치일 뿐** — 첫 50스텝을 실측해 PROGRESS.md에 기록한다.

### Stage 4: 평가와 재색인

| 파일 | 하는 일 | 주의점 |
|---|---|---|
| `evaluate.py` | 모델 경로 인자 → Recall@1/@5, MRR@10 | **벡터 단독**으로 측정 (하이브리드로 재면 BM25가 개선분을 가림) |
| `reindex.py` | 새 모델로 1,136청크 전량 재색인 → `chroma_db_ft/` | **기존 `chroma_db/`는 절대 건드리지 않는다** |

- baseline과 파인튜닝 모델이 **동일한 절차**를 타야 비교가 성립한다.
- 임베딩 모델이 바뀌면 벡터 공간이 완전히 달라지므로 **전량 재색인이 필수**. 섞으면 검색이 무의미해진다.
- BM25(`bm25_index.pkl`)는 임베딩과 무관하므로 재빌드 불필요.

### Stage 5: 앱 연동 (선택)

- `.streamlit/secrets.toml`에 `EMBEDDING_MODEL` / `CHROMA_DIR` 키 추가,
  `app.py`·`indexer.py`가 하드코딩 대신 이 값을 읽도록 변경.
  기본값을 현재 값으로 두면 미설정 시 동작이 바뀌지 않는다.
- 여기서 **처음으로** 기존 파일을 수정한다.
  Stage 4 숫자가 개선을 보이지 않으면 **이 단계는 하지 않는다.**

---

## 8. 정직한 한계 (미리 알고 시작해야 할 것)

1. **합성 질문은 점수를 부풀린다.**
   LLM이 청크를 보고 만든 질문은 그 청크의 표현을 그대로 베끼는 경향이 있어,
   모델은 "질문에 답하기"가 아니라 "출처 문장 찾기"를 학습한다.
   합성 eval에서 Recall@5가 크게 올라도 실사용 개선을 보장하지 않는다.
   → **직접 작성한 `eval_human.jsonl` 30~50문항이 진짜 지표다.**
   이 계획에서 사용자 손이 가장 많이 가는 부분이며, 생략하면 결과를 신뢰할 수 없다.
2. **데이터가 작다.** 17개 위키 페이지 / 1,136 청크에서 나온 2천 쌍은 임베딩 파인튜닝으로는 소규모다.
   개선 폭이 미미하거나 **오히려 나빠질 수 있다.**
   다만 이 규모가 PyTorch를 배우기에는 오히려 적당하다 — 실험 한 사이클이 30분 안에 끝난다.
3. **표 형식 청크가 많다.** 관리대장류 마크다운 표는 문장형 임베딩 모델이 원래 약한 영역이고,
   그런 문서는 BM25가 더 잘 잡는다. 파인튜닝이 이 부분을 크게 개선하리라 기대하지 않는 게 맞다.
4. **RAM이 진짜 빡빡하다.** 여유 1.4 GB에 스왑이 이미 1.1 GB 사용 중이다.
   학습 중에는 **Streamlit 앱을 반드시 내려야** 한다(앱이 임베딩 모델을 상주시킨다).

---

## 9. 단계별 검증 게이트

각 단계는 다음 단계로 넘어가기 전에 스스로 증명한다.

| 시점 | 통과 조건 |
|---|---|
| Stage 1 완료 | `06_encoder.py`가 `allclose(수동, encode(), atol=1e-5) == True` 출력 |
| Stage 2 초반 | `wc -l data/chunks.jsonl == 1136` |
| Stage 2 중간 | `gen_queries.py --limit 5` 결과를 **눈으로 읽어** 한국어로 말이 되는지 확인 후 전량 실행 |
| **Stage 3 직전** | `evaluate.py --model jhgan/ko-sroberta-multitask` 로 **baseline 기록** ← 빠뜨리면 비교 불가 |
| Stage 3 초반 | 첫 50스텝 loss 하강 추세 + `free -h`로 RSS가 여유 안 |
| Stage 4 | 합성 eval과 `eval_human` **양쪽 보고**. 어긋나면 사람이 만든 쪽을 신뢰 |
| Stage 5 직전 | `reindex.py` 후 실제 질의 몇 개를 기존/신규 컬렉션에 던져 상위 결과를 나란히 눈으로 비교 |

---

## 10. 재사용할 기존 코드

새로 짜기 전에 이미 있는 것을 쓴다.

| 위치 | 무엇을 | 어디서 쓰나 |
|---|---|---|
| `chunker.py` | `chunk_text()` | 학습 데이터 청킹 규칙을 색인과 동일하게 유지 |
| `tokenizer.py` | `tokenize_ko()` | BM25 비교 평가 시 |
| `retriever.py:113` | `hybrid_search()` | 하이브리드 조건에서의 최종 확인용 |
| `app.py:103-140` | LLM 호출 규약 | `gen_queries.py`가 그대로 따름 |
| `indexer.py` | Chroma 색인 패턴 | `reindex.py`가 참고 |
