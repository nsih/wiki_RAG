"""06. 인코더 — 텍스트가 768차원 벡터가 되기까지

★ 이 파일이 Stage 1의 핵심 검증 지점이다.

indexer.py는 SentenceTransformerEmbeddingFunction 한 줄로 임베딩을 만든다.
그 안에서 무슨 일이 일어나는지 손으로 재현하고,
결과가 소수점 이하까지 일치하는지 torch.allclose로 증명한다.

전체 흐름:
    "안녕하세요"
      → 토크나이저 → input_ids (1, L), attention_mask (1, L)
      → AutoModel  → last_hidden_state (1, L, 768)      토큰마다 벡터
      → mean pooling (mask 가중)  → (1, 768)            문장 하나의 벡터
      → L2 정규화                 → (1, 768)            길이가 1인 벡터

실행: python torch_lab/06_encoder.py
"""

import gc
import os
import resource
import sqlite3
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

torch.set_num_threads(2)
torch.manual_seed(42)

MODEL_NAME = "jhgan/ko-sroberta-multitask"
CHROMA_DB = Path(__file__).parent.parent / "chroma_db" / "chroma.sqlite3"


def rss_mb() -> float:
    """지금 이 순간 쓰는 물리 메모리(MB).

    주의: resource.getrusage().ru_maxrss 는 '최댓값'이라 메모리를 해제해도 줄지 않는다.
    현재값을 보려면 /proc/self/status 의 VmRSS 를 읽어야 한다.
    """
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024
    return -1.0


def peak_mb() -> float:
    """프로세스가 지금까지 찍은 최대 RSS(MB). OOM 여유를 볼 때는 이쪽이 중요하다."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


print(f"시작 RSS: {rss_mb():.0f} MB\n")


# ── 1. 토크나이저 — 텍스트를 숫자로 ─────────────────────────────────────────

print("=" * 60)
print("1. 토크나이저 — 텍스트 → 정수 ID")
print("=" * 60)

tok = AutoTokenizer.from_pretrained(MODEL_NAME)
print(f"어휘 크기: {tok.vocab_size:,}개")
print(f"특수 토큰: {tok.cls_token}={tok.cls_token_id}, {tok.sep_token}={tok.sep_token_id}, "
      f"{tok.pad_token}={tok.pad_token_id}")

text = "기숙사 무선AP 관리대장"
enc = tok(text, return_tensors="pt")

print(f"\n입력: {text!r}")
print(f"\n{'위치':>4} {'토큰':>12} {'ID':>7}")
print("-" * 26)
for i, tid in enumerate(enc["input_ids"][0].tolist()):
    print(f"{i:>4} {tok.convert_ids_to_tokens(tid):>12} {tid:>7}")

print(f"""
'##'로 시작하는 토큰은 앞 토큰에 이어 붙는 조각이다 (subword).
한국어는 조사·어미 때문에 단어가 무한히 늘어나므로, 자주 나오는 조각으로 쪼개
32,000개 어휘로 모든 문장을 표현한다.

[CLS]는 문장 시작, [SEP]는 끝을 알리는 특수 토큰이다.
tokenizer.py의 tokenize_ko()(BM25용)와는 완전히 다른 방식이다:
  BM25용   : 정규식으로 단어 추출 + 조사 제거 → 사람이 읽을 수 있는 단어
  트랜스포머: 학습된 subword 사전 → 모델이 아는 조각
""")


# ── 2. 패딩과 attention_mask ────────────────────────────────────────────────

print("=" * 60)
print("2. 배치로 묶으면 패딩이 생긴다 (04번의 그 문제)")
print("=" * 60)

texts = ["무선AP", "기숙사 무선AP 관리대장", "IP 주소 대장에서 10호관 장비를 찾아줘"]
batch = tok(texts, padding=True, truncation=True, max_length=128, return_tensors="pt")

print(f"입력 3문장 → input_ids {tuple(batch['input_ids'].shape)}  (가장 긴 문장에 맞춰 패딩)\n")
for i, t in enumerate(texts):
    ids = batch["input_ids"][i]
    mask = batch["attention_mask"][i]
    print(f"  {t[:24]:<26} 실제 토큰 {mask.sum().item():>2}개 / 전체 {len(ids)}칸")
    print(f"  {'':26} mask = {mask.tolist()}")

print("""
attention_mask 의 1은 '진짜 토큰', 0은 '길이 맞추려고 채운 빈칸'이다.
04번에서 직접 만든 pad_collate의 mask와 정확히 같은 개념이다.
차이는 토크나이저가 padding=True 한 줄로 해준다는 것뿐.
""")


# ── 3. 모델 통과 ────────────────────────────────────────────────────────────

print("=" * 60)
print("3. AutoModel — 토큰마다 768차원 벡터")
print("=" * 60)

model = AutoModel.from_pretrained(MODEL_NAME)
model.eval()                       # ★ 05번: Dropout을 끈다. 빼먹으면 매번 값이 달라진다
print(f"모델 로드 후 RSS: {rss_mb():.0f} MB")

n_params = sum(p.numel() for p in model.parameters())
print(f"파라미터: {n_params:,}개 ({n_params*4/1024**2:.0f} MB, fp32)")
print(f"구조: {model.config.num_hidden_layers}층, hidden {model.config.hidden_size}, "
      f"헤드 {model.config.num_attention_heads}개")

with torch.no_grad():              # ★ 05번: 추론이므로 그래프를 만들지 않는다
    out = model(**batch)

hidden = out.last_hidden_state
print(f"\nlast_hidden_state: {tuple(hidden.shape)}")
print(f"  = (문장 {hidden.shape[0]}개, 토큰 {hidden.shape[1]}칸, 차원 {hidden.shape[2]})")
print("\n각 토큰이 768차원 벡터를 하나씩 갖는다.")
print("우리가 원하는 건 '문장 하나당 벡터 하나'이므로 토큰 축을 뭉개야 한다.")


# ── 4. mean pooling — 04번 연습 3번의 정답 ──────────────────────────────────

print("\n" + "=" * 60)
print("4. mean pooling — 패딩을 빼고 평균낸다")
print("=" * 60)


def mean_pool(last_hidden_state, attention_mask):
    """attention_mask로 패딩을 걸러낸 평균. 이게 이 프로젝트 임베딩의 정체다.

    Args:
        last_hidden_state: (B, L, H)
        attention_mask:    (B, L)
    Returns:
        (B, H) — 문장마다 벡터 하나
    """
    # (B, L) → (B, L, 1) 로 만들어 H축으로 브로드캐스팅 (01번 3절)
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    summed = (last_hidden_state * mask).sum(dim=1)          # 진짜 토큰만 더한다
    counts = mask.sum(dim=1).clamp(min=1e-9)                # 진짜 토큰 개수 (0 나눗셈 방지)
    return summed / counts


pooled = mean_pool(hidden, batch["attention_mask"])
naive = hidden.mean(dim=1)                                   # 패딩까지 포함한 잘못된 평균

print(f"mean_pool 결과: {tuple(pooled.shape)}")
print(f"\n{'문장':<28} {'올바른 평균과 단순 평균의 차이':>22}")
print("-" * 52)
for i, t in enumerate(texts):
    diff = (pooled[i] - naive[i]).abs().max().item()
    n_pad = (batch["attention_mask"][i] == 0).sum().item()
    print(f"{t[:26]:<28} {diff:>18.6f}  (패딩 {n_pad}칸)")

print("""
패딩이 없는 문장(가장 긴 것)은 차이가 0이다. 패딩이 많을수록 오차가 커진다.
★ 이걸 빼먹으면 에러 없이 조용히 검색 품질만 나빠진다.
""")


# ── 5. 정규화 ───────────────────────────────────────────────────────────────

print("=" * 60)
print("5. L2 정규화 — 길이를 1로")
print("=" * 60)

normalized = F.normalize(pooled, p=2, dim=1)
print(f"정규화 전 벡터 길이: {[round(v, 3) for v in pooled.norm(dim=1).tolist()]}")
print(f"정규화 후 벡터 길이: {[round(v, 3) for v in normalized.norm(dim=1).tolist()]}")
print("""
길이를 1로 맞추면 내적이 곧 코사인 유사도가 된다 (01번 4절의 '방법 C').
검색할 때마다 나눗셈을 반복할 필요가 없어진다.
""")


# ── 6. ★ 검증 — SentenceTransformer와 일치하는가 ────────────────────────────

print("=" * 60)
print("6. ★ 검증 — 손으로 만든 것 == SentenceTransformer.encode()")
print("=" * 60)

# 메모리 절약: AutoModel을 먼저 버린다 (같은 가중치를 두 벌 들고 있을 필요 없음)
manual_vecs = mean_pool(hidden, batch["attention_mask"])     # 정규화 전 값으로 비교
del model, out, hidden
gc.collect()
print(f"AutoModel 해제 후 RSS: {rss_mb():.0f} MB")

from sentence_transformers import SentenceTransformer

st = SentenceTransformer(MODEL_NAME, device="cpu")
print(f"SentenceTransformer 로드 후 RSS: {rss_mb():.0f} MB\n")

print("SentenceTransformer가 내부적으로 쓰는 모듈:")
for name, module in st.named_children():
    print(f"  {name}: {type(module).__name__}")

st_vecs = torch.tensor(st.encode(texts, convert_to_numpy=True))

print(f"\n수동 구현 : {tuple(manual_vecs.shape)}")
print(f"encode()  : {tuple(st_vecs.shape)}")

max_diff = (manual_vecs - st_vecs).abs().max().item()
ok = torch.allclose(manual_vecs, st_vecs, atol=1e-5)

print(f"\n최대 절대 오차: {max_diff:.3e}")
print(f"allclose(atol=1e-5): {ok}")
print(f"\n첫 문장 벡터 앞 5개 비교:")
print(f"  수동    : {[round(v, 6) for v in manual_vecs[0, :5].tolist()]}")
print(f"  encode(): {[round(v, 6) for v in st_vecs[0, :5].tolist()]}")

if ok:
    print("""
★ 통과. indexer.py가 쓰는 SentenceTransformerEmbeddingFunction의 내부는
  '토크나이즈 → 트랜스포머 → mask 가중 mean pooling' 이 전부였다.
  블랙박스가 아니다. Stage 3에서 이 mean_pool 함수를 그대로 쓴다.
""")
else:
    print("""
불일치. 원인 후보:
  - 모델이 CLS 풀링을 쓰는 경우 (1_Pooling/config.json 확인)
  - 정규화 여부 차이
  - max_length 설정 차이
""")


# ── 7. 실전 — 실제 위키 청크로 검색해보기 ───────────────────────────────────

print("=" * 60)
print("7. 실전 — 실제 위키 청크를 검색해본다")
print("=" * 60)

# ★ 전체를 다 읽어야 한다. LIMIT을 걸고 앞부분만 보면 page 33(전체의 76%)이
#   후보를 독점해서 정답 문서가 아예 후보에 없는 채로 "검색이 잘 된다"고 착각하게 된다.
con = sqlite3.connect(f"file:{CHROMA_DB}?mode=ro", uri=True)
rows = con.execute("""
    SELECT MAX(CASE WHEN m.key='chroma:document' THEN m.string_value END),
           MAX(CASE WHEN m.key='title' THEN m.string_value END),
           MAX(CASE WHEN m.key='page_id' THEN m.int_value END)
    FROM embeddings e JOIN embedding_metadata m ON m.id = e.id
    GROUP BY e.id
""").fetchall()
con.close()

# 페이지마다 가장 긴 청크 하나씩 — 18개 페이지가 모두 후보에 들어간다
best = {}
for doc, title, pid in rows:
    if doc and (pid not in best or len(doc) > len(best[pid][0])):
        best[pid] = (doc, title)
docs = [d for d, _ in best.values()]
titles = [t for _, t in best.values()]

print(f"전체 {len(rows):,}청크 → 페이지별 대표 1개씩 {len(docs)}개를 임베딩한다...")
doc_vecs = torch.tensor(st.encode(docs, convert_to_numpy=True, normalize_embeddings=True))

# 질의와, 사람이 보기에 정답이어야 할 문서 제목
cases = [
    ("랜섬웨어에 감염되면 어떻게 대응하나요?", "랜섬웨어"),
    ("기숙사 10호관 무선AP는 어디에 설치되어 있나요?", "기숙사"),
    ("외부 업체가 원격으로 작업하려면 어떤 절차가 필요한가요?", "원격작업"),
]

for query, expect_kw in cases:
    q_vec = torch.tensor(st.encode([query], convert_to_numpy=True, normalize_embeddings=True))
    sims = (q_vec @ doc_vecs.T)[0]           # 01번 4절: 정규화 후 내적 = 코사인 유사도
    order = sims.argsort(descending=True).tolist()

    # 기대 문서가 몇 위인지
    gold_rank = next((r for r, i in enumerate(order, 1) if expect_kw in titles[i]), None)

    print(f"\n질의: {query}")
    print(f"기대: 제목에 {expect_kw!r} 가 든 문서  →  실제 순위: "
          f"{gold_rank if gold_rank else '후보에 없음'}위")
    print(f"  {'순위':>4} {'유사도':>8}  제목")
    print("  " + "-" * 56)
    for rank, i in enumerate(order[:3], 1):
        mark = " ←정답" if expect_kw in titles[i] else ""
        print(f"  {rank:>4} {sims[i]:>8.4f}  {titles[i][:40]}{mark}")

print(f"""
이게 retriever.py의 벡터 검색 부분이다. ChromaDB는 이 계산을
HNSW 인덱스로 빠르게 근사할 뿐, 원리는 위 세 줄과 같다.

Stage 3의 파인튜닝은 이 유사도 순위를 개선하는 작업이다.
""")

print(f"\n최종 RSS: {rss_mb():.0f} MB / 최대 RSS: {peak_mb():.0f} MB")


# ── 연습 과제 ──────────────────────────────────────────────────────────────
#
# 1) 1절에서 text를 "무선AP" / "무선 AP" / "무선ap" 로 바꿔보라.
#    토큰이 어떻게 달라지는가? 임베딩 유사도는?
#    (사내 용어가 subword로 잘게 쪼개진다면, 그게 파인튜닝이 필요한 이유다)
#
# 2) 4절의 mean_pool에서 clamp(min=1e-9)를 빼면 언제 문제가 생기는가?
#    (힌트: attention_mask가 전부 0인 입력이 들어온다면?)
#
# 3) 7절의 질의를 "10호관 무선AP 설치 위치"로 바꿔보라. 원하는 문서가 1위인가?
#    아니라면 왜일까? PLAN.md의 '표 형식 청크' 한계와 연결해 생각해보라.
#
# 4) mean pooling 대신 [CLS] 토큰 벡터(hidden[:, 0])만 쓰면 어떻게 다른가?
#    두 방식의 유사도 순위를 비교해보라.
