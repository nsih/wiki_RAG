"""07. Contrastive Loss — 검색 모델은 이렇게 학습한다

지금까지: 텍스트 → 벡터 (06번)
이제:     "질문 벡터가 정답 문서 벡터와 가까워지도록" 학습시키는 법

문제 설정이 03번의 회귀와 다르다. 정답 벡터가 주어지지 않는다.
우리가 가진 건 (질문, 정답 문서) 쌍뿐이고, "가까워야 한다"는 관계만 안다.

해법: 배치 안의 다른 문서들을 '오답'으로 쓴다 (in-batch negative).
      배치 8이면 정답 1개 + 오답 7개 → 8지선다 객관식 문제가 된다.
      그럼 분류 문제가 되고, cross_entropy를 쓸 수 있다.

실행: python torch_lab/07_contrastive.py
"""

import resource
import time

import torch
import torch.nn.functional as F

torch.set_num_threads(2)
torch.manual_seed(42)

MODEL_NAME = "jhgan/ko-sroberta-multitask"


def rss_mb() -> float:
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024
    return -1.0


def peak_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


# ── 1. 유사도 행렬 — 정답은 대각선 ──────────────────────────────────────────

print("=" * 60)
print("1. 유사도 행렬 (B, B) — 정답이 대각선에 온다")
print("=" * 60)

B, H = 4, 768
q = F.normalize(torch.randn(B, H), dim=1)      # 질문 벡터 4개
d = F.normalize(torch.randn(B, H), dim=1)      # 정답 문서 벡터 4개

sim = q @ d.T                                   # (4, 768) @ (768, 4) → (4, 4)

print(f"q {tuple(q.shape)} @ d.T {tuple(d.T.shape)} = sim {tuple(sim.shape)}")
print(f"\nsim[i][j] = i번 질문과 j번 문서의 코사인 유사도\n")
print("        " + "".join(f"  문서{j}  " for j in range(B)))
for i in range(B):
    row = "".join(f" {sim[i,j]:+.3f}" for j in range(B))
    print(f"  질문{i} {row}   ← 정답은 [{i}][{i}]")

print("""
쌍으로 만든 데이터이므로 i번 질문의 정답은 항상 i번 문서다.
즉 각 행에서 '대각선 원소가 가장 커야 한다'.

이건 정확히 4지선다 분류 문제다:
  입력  = 각 행의 점수 4개
  정답  = i (대각선 위치)
→ cross_entropy를 그대로 쓸 수 있다.
""")


# ── 2. cross_entropy가 하는 일 ──────────────────────────────────────────────

print("=" * 60)
print("2. cross_entropy — 직접 계산해서 확인")
print("=" * 60)

scores = torch.tensor([[2.0, 0.5, 0.1, 0.3]])   # 4지선다 점수
label = torch.tensor([0])                        # 정답은 0번

probs = F.softmax(scores, dim=1)
manual = -torch.log(probs[0, label[0]])          # 정답 확률에 -log
builtin = F.cross_entropy(scores, label)

print(f"점수      : {scores[0].tolist()}")
print(f"softmax   : {[round(v, 4) for v in probs[0].tolist()]}   (합 = {probs.sum():.1f})")
print(f"정답(0번) 확률: {probs[0,0]:.4f}")
print(f"\n수동 -log(p): {manual:.6f}")
print(f"cross_entropy: {builtin:.6f}")
print(f"같은가: {torch.allclose(manual, builtin)}")

print("\n정답 확률과 손실의 관계:")
for p in [0.99, 0.9, 0.5, 0.25, 0.1, 0.01]:
    print(f"  정답 확률 {p:>5.2f} → 손실 {-torch.log(torch.tensor(p)):.4f}")
print("""
정답을 확신할수록 손실이 0에 가깝고, 틀릴수록 급격히 커진다.
4지선다에서 완전히 찍으면(p=0.25) 손실은 log(4)=1.386이다.
→ 학습 시작 시 손실이 log(배치크기) 근처면 정상이다. 진단에 쓸 수 있다.
""")


# ── 3. contrastive loss 구현 ────────────────────────────────────────────────

print("=" * 60)
print("3. ★ 손실 함수 — 이 8줄이 전부다")
print("=" * 60)


def contrastive_loss(q_vecs, d_vecs, scale: float = 20.0):
    """In-batch negative contrastive loss.

    sentence-transformers의 MultipleNegativesRankingLoss와 같은 계산이다.

    Args:
        q_vecs: (B, H) 질문 벡터
        d_vecs: (B, H) 정답 문서 벡터 — i번째가 i번 질문의 정답
        scale:  유사도를 몇 배로 키워 softmax에 넣을지 (온도의 역수)
    """
    q_vecs = F.normalize(q_vecs, p=2, dim=1)          # 코사인 유사도를 위해 정규화
    d_vecs = F.normalize(d_vecs, p=2, dim=1)
    sim = (q_vecs @ d_vecs.T) * scale                 # (B, B), 대각선이 정답
    labels = torch.arange(len(q_vecs), device=sim.device)   # [0, 1, 2, ..., B-1]
    return F.cross_entropy(sim, labels)


print("완벽한 경우 vs 최악인 경우의 손실:")
perfect_q = F.normalize(torch.randn(B, H), dim=1)
print(f"  q == d (완벽)      : {contrastive_loss(perfect_q, perfect_q):.6f}")
print(f"  무작위             : {contrastive_loss(q, d):.6f}")
shuffled = perfect_q[torch.tensor([1, 2, 3, 0])]     # 정답을 한 칸씩 밀어놓음
print(f"  정답 위치를 어긋냄 : {contrastive_loss(perfect_q, shuffled):.6f}")
print(f"\n  참고: 완전 무작위의 기댓값 = log(B) = log({B}) = {torch.log(torch.tensor(float(B))):.4f}")


# ── 4. scale(온도)의 역할 ───────────────────────────────────────────────────

print("\n" + "=" * 60)
print("4. scale — 왜 20을 곱하나")
print("=" * 60)

print("코사인 유사도는 [-1, 1] 범위다. 그대로 softmax에 넣으면:")
raw = torch.tensor([[0.9, 0.7, 0.5, 0.3]])          # 정답이 확실히 1등인데도
print(f"  유사도 {raw[0].tolist()}")
print(f"  softmax {[round(v,4) for v in F.softmax(raw, dim=1)[0].tolist()]}")
print(f"  → 정답 확률이 {F.softmax(raw, dim=1)[0,0]:.2f}밖에 안 된다. 1등인데도!")

print(f"\nscale을 곱하면 차이가 벌어진다:")
print(f"{'scale':>6} {'정답 확률':>10} {'손실':>10}")
print("-" * 28)
for s in [1, 5, 10, 20, 50, 100]:
    p = F.softmax(raw * s, dim=1)[0, 0]
    l = F.cross_entropy(raw * s, torch.tensor([0]))
    print(f"{s:>6} {p:>10.4f} {l:>10.4f}")

print("""
scale이 작으면 정답과 오답의 차이를 '충분히 잘했다'고 인정하지 않아
그래디언트가 계속 흐르고, 크면 조금만 앞서도 만족해 학습이 멈춘다.
20은 sentence-transformers의 기본값이고, 사실상 표준으로 쓰인다.
(04번 문서의 '관행' 항목에 해당한다 — 유도된 값이 아니다)
""")


# ── 5. 배치 크기 = 문제의 난이도 ────────────────────────────────────────────

print("=" * 60)
print("5. ★ 배치 크기가 왜 중요한가")
print("=" * 60)

print(f"{'배치':>6} {'객관식':>8} {'찍었을 때 손실':>14}")
print("-" * 32)
for bs in [2, 4, 8, 16, 64, 256]:
    print(f"{bs:>6} {bs:>6}지선다 {torch.log(torch.tensor(float(bs))):>14.4f}")

print("""
배치가 클수록 오답 후보가 많아져 문제가 어려워지고, 그만큼 잘 배운다.
그래서 검색 모델 학습은 보통 배치 64~512를 쓴다.

★ 우리는 RAM 때문에 batch_size=8 이다. 오답이 7개뿐인 쉬운 문제다.
  이게 Stage 3의 가장 아픈 제약이다. (아래 마스킹을 쓰면 유효 negative는 더 줄어든다)
""")

# false negative — 우리 코퍼스의 실제 문제
print("게다가 더 나쁜 문제가 있다: false negative")
print("""
in-batch negative는 '같은 배치의 다른 문서는 오답'이라고 **가정**한다.
그 가정이 깨지면(= 오답인 줄 알았는데 사실 정답과 다름없으면) 손실 함수가
"이 둘을 멀리 떨어뜨려라"라고 틀린 신호를 준다. 이게 false negative다.

★ 우리 코퍼스에서 실제로 얼마나 위험한지 측정했다 (2026-09-10):

  청크 쌍의 코사인 유사도       평균     >0.95    >0.80
  ---------------------------- ------  -------  -------
  page 33 내부 (IP 대장)        0.692     0.2%     9.9%
  다른 페이지 내부              0.691     0.3%    20.1%
  페이지 간                     0.470     0.0%     0.0%

  → page 33이 특별히 자기유사한 게 아니었다. 오히려 다른 페이지가 더 높다.
    IP 대장의 행들은 IP 주소가 전부 달라 생각보다 잘 구분된다.
    '같은 페이지 = 거의 같은 내용' 이라는 처음 가정은 틀렸다.

그래도 같은 페이지 청크가 페이지 간(0.470)보다 훨씬 유사한 건 사실이므로,
안전장치는 두는 게 맞다. 단 배치 구성을 제약하는 대신 **손실에서 마스킹**한다:

    same = page_ids[:, None] == page_ids[None, :]   # 같은 페이지면 True
    same.fill_diagonal_(False)                       # 대각선(정답)은 살린다
    sim = sim.masked_fill(same, -1e4)                # 같은 페이지는 후보에서 제외

배치 구성을 제약하면 페이지가 18개뿐이라 batch 8 기준 33배치가 한계다.
마스킹은 데이터를 100% 쓰면서 false negative만 없앤다.

★ 단, 배치 전체가 같은 페이지면 negative가 0개가 되어 학습 신호가 사라진다.
  --max-per-page 로 한 페이지 비중을 낮춰두면(60이면 18.9%) 거의 일어나지 않는다.
""")


# ── 6. 실제 모델로 미니 학습 — Stage 3 설정 그대로 ──────────────────────────

print("=" * 60)
print("6. 실제 모델로 10스텝 — Stage 3 설정 실측")
print("=" * 60)

from transformers import AutoModel, AutoTokenizer

FREEZE_BELOW = 8        # PLAN.md: 임베딩층 + 0~7층 동결, 8~11층만 학습
BATCH = 8
MAX_LEN = 128
LR = 2e-5

print(f"설정: freeze_below={FREEZE_BELOW}, batch={BATCH}, max_len={MAX_LEN}, lr={LR}")
print(f"모델 로드 전 RSS: {rss_mb():.0f} MB")

tok = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)
print(f"모델 로드 후 RSS: {rss_mb():.0f} MB")


def is_frozen(name: str, freeze_below: int) -> bool:
    """임베딩층과 freeze_below 미만 층을 동결 대상으로 판정."""
    if name.startswith("embeddings."):
        return True
    if name.startswith("encoder.layer."):
        layer_idx = int(name.split(".")[2])
        return layer_idx < freeze_below
    return False        # pooler 등은 학습


for name, p in model.named_parameters():
    p.requires_grad = not is_frozen(name, FREEZE_BELOW)

total = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\n파라미터: 전체 {total:,} / 학습 대상 {trainable:,} ({trainable/total*100:.1f}%)")
print(f"예상 메모리: 가중치 {total*4/1024**2:.0f} MB"
      f" + 그래디언트·상태 {trainable*12/1024**2:.0f} MB"
      f" = {(total*4 + trainable*12)/1024**2:.0f} MB")

opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR)


def mean_pool(hidden, mask):
    """06번에서 검증한 그 함수."""
    m = mask.unsqueeze(-1).expand(hidden.size()).float()
    return (hidden * m).sum(1) / m.sum(1).clamp(min=1e-9)


# 가짜 학습 데이터 — Stage 2가 만들 (질문, 정답청크) 쌍의 자리
queries = [
    "랜섬웨어에 감염되면 어떻게 하나요", "무선AP는 어디에 설치되어 있나요",
    "원격작업 승인 절차가 궁금합니다", "정보보안 담당자는 누구인가요",
    "홈페이지 취약점 점검 결과는", "구매 규정을 알려주세요",
    "침해사고가 나면 누구에게 보고하나요", "IP 주소는 어떻게 할당받나요",
]
positives = [
    "랜섬웨어 대응 가이드라인: 감염 발견 즉시 네트워크에서 분리하고 정보전산원에 신고한다",
    "기숙사 무선AP 관리대장: 10호관 3층 301호 iptime Extender-A6T 설치",
    "용역업체 원격작업 금지 및 승인 절차 매뉴얼: 사전 승인 신청서를 제출해야 한다",
    "정보보안 기본지침: 정보보호 최고책임자는 정보전산원장이 겸임한다",
    "2025학년도 홈페이지 취약점 점검 및 조치 결과 보고서: 총 12건 발견 및 조치 완료",
    "창신대학교 구매업무 규정: 500만원 이상은 공개입찰을 원칙으로 한다",
    "침해사고 대응 매뉴얼: 사고 인지 후 1시간 이내 정보전산원장에게 보고한다",
    "IP 주소 대장: 신규 할당은 정보전산원에 신청서를 제출하여 배정받는다",
]

model.train()          # 05번: 학습 모드 (Dropout 켜짐)
print(f"\n{'step':>5} {'loss':>9} {'소요(초)':>9} {'RSS(MB)':>9}")
print("-" * 36)

times = []
for step in range(10):
    t0 = time.perf_counter()

    q_enc = tok(queries, padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt")
    d_enc = tok(positives, padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt")

    q_vec = mean_pool(model(**q_enc).last_hidden_state, q_enc["attention_mask"])
    d_vec = mean_pool(model(**d_enc).last_hidden_state, d_enc["attention_mask"])

    loss = contrastive_loss(q_vec, d_vec)      # ← 3절에서 만든 그 함수
    loss.backward()                            # ← 02번
    opt.step()                                 # ← 03번
    opt.zero_grad()                            # ← 03번

    dt = time.perf_counter() - t0
    times.append(dt)
    print(f"{step:>5} {loss.item():>9.4f} {dt:>9.2f} {rss_mb():>9.0f}")

avg = sum(times[1:]) / len(times[1:])          # 첫 스텝은 워밍업이라 제외 (01번 교훈)
print("-" * 36)
print(f"\n스텝당 평균: {avg:.2f}초 (첫 스텝 제외)")
print(f"최대 RSS   : {peak_mb():.0f} MB")
print(f"""
[해석] 시작 손실이 log({BATCH})={torch.log(torch.tensor(float(BATCH))):.4f} 가 아니라 0.12였다.
  이 8개 쌍은 주제가 전부 달라서 사전학습 모델이 **이미 풀 수 있는** 문제였다.
  즉 배울 게 없는 데이터다. 손실이 순식간에 0에 수렴한 것도 그 때문이다.

  실제 Stage 2 데이터는 이렇지 않다:
    - 같은 페이지에서 나온 비슷한 청크들이 섞인다 (5절의 false negative)
    - 질문이 LLM 생성이라 표현이 다양하다
  → 진짜 학습에서는 시작 손실이 log(batch)에 훨씬 가까울 것이다.

  ★ 진단 기준으로 쓰기: 시작 손실이 log(batch)보다 한참 낮으면
    '데이터가 너무 쉽다'는 신호다. 그런 데이터로는 모델이 나아지지 않는다.
""")

print(f"""
★ Stage 3 소요 시간 추정 (이 실측 기준):
  학습 쌍 2,000개 / batch {BATCH} = 250스텝
  250 x {avg:.2f}초 = {250*avg/60:.1f}분/에폭
  2에폭이면 약 {2*250*avg/60:.0f}분

  PLAN.md의 '에폭당 15~30분' 추정과 비교해 PROGRESS.md에 기록한다.
""")


# ── 연습 과제 ──────────────────────────────────────────────────────────────
#
# 1) 6절의 positives 8개 중 2개를 거의 같은 내용으로 바꿔보라 (false negative 재현).
#    손실이 어떻게 달라지는가? 그 상태로 학습하면 모델이 무엇을 배우게 되는가?
#
# 2) 3절 contrastive_loss에서 F.normalize 두 줄을 빼면 어떻게 되는가?
#    (힌트: 벡터 길이가 유사도에 섞여 들어간다)
#
# 3) 6절에서 FREEZE_BELOW를 0(전체 학습)으로 바꾸면 RSS가 얼마나 오르는가?
#    ★ 주의: 이 서버에서는 OOM으로 죽을 수 있다. free -h 로 여유를 먼저 확인하라.
#
# 4) 손실 함수를 대칭으로 만들어보라 — 질문→문서 방향뿐 아니라
#    문서→질문 방향(sim.T)도 함께 계산해 평균내는 방식이다.
#    실제 검색 모델 학습에서 자주 쓰인다. 왜 도움이 될까?
