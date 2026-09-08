"""01. 텐서 — PyTorch의 유일한 자료구조

PyTorch에서 다루는 모든 것(입력, 가중치, 그래디언트, 손실값)은 전부 텐서다.
텐서는 그냥 "다차원 배열 + GPU 연산 + 자동미분"이라고 보면 된다.

이 프로젝트와의 연결:
    indexer.py가 만드는 768차원 임베딩 하나가 곧 torch.Size([768]) 텐서다.
    ChromaDB의 cosine 유사도 계산도 아래 matmul 한 줄과 정확히 같은 연산이다.

실행: python torch_lab/01_tensor.py
"""

import torch
import torch.nn.functional as F

torch.set_num_threads(2)      # 2코어 서버 — 기본값이면 스레드 경합으로 오히려 느려진다
torch.manual_seed(42)         # 재현성: 매 실행 같은 난수


# ── 1. 텐서 만들기 ──────────────────────────────────────────────────────────

print("=" * 60)
print("1. 텐서 생성")
print("=" * 60)

# 파이썬 리스트에서
a = torch.tensor([1.0, 2.0, 3.0])
print(f"a           = {a}")
print(f"a.shape     = {a.shape}      ← 1차원, 원소 3개")
print(f"a.dtype     = {a.dtype}      ← 기본은 float32 (4바이트)")
print(f"a.device    = {a.device}     ← GPU가 없으니 cpu")

# 모양만 주고 값은 자동으로
zeros = torch.zeros(2, 3)
randn = torch.randn(2, 3)     # 정규분포 N(0,1)에서 샘플링
print(f"\nzeros.shape = {zeros.shape}")
print(f"randn       =\n{randn}")

# 신경망 가중치는 전부 이런 랜덤 텐서에서 출발한다.


# ── 2. shape 이 전부다 ──────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("2. shape — 디버깅의 90%는 shape 확인이다")
print("=" * 60)

# 임베딩 모델의 실제 데이터 흐름을 shape 으로만 표현해보면:
B, L, H = 8, 128, 768         # 배치 8문장, 각 128토큰, hidden 768차원

token_ids     = torch.randint(0, 32000, (B, L))   # 토크나이저 출력
hidden_states = torch.randn(B, L, H)              # 트랜스포머 출력
sentence_vec  = hidden_states.mean(dim=1)         # 토큰 축(L)을 평균 → 문장 벡터

print(f"token_ids      {tuple(token_ids.shape)}   ← 정수 토큰 ID (dtype={token_ids.dtype})")
print(f"hidden_states  {tuple(hidden_states.shape)}  ← 토큰마다 768차원 벡터")
print(f"sentence_vec   {tuple(sentence_vec.shape)}       ← dim=1(L)을 평균으로 없앰")
print("\n이게 06_encoder.py에서 실제로 할 mean pooling의 뼈대다.")
print("dim=1 을 평균낸다 = '토큰 128개의 벡터를 하나로 뭉갠다'")

# 자주 쓰는 모양 변경
x = torch.randn(4, 6)
print(f"\nx.shape            = {tuple(x.shape)}")
print(f"x.T.shape          = {tuple(x.T.shape)}          ← 전치")
print(f"x.reshape(2,12)    = {tuple(x.reshape(2, 12).shape)}         ← 총 원소 수만 같으면 됨")
print(f"x.unsqueeze(0)     = {tuple(x.unsqueeze(0).shape)}       ← 맨 앞에 크기 1 축 추가")
print(f"x.unsqueeze(-1)    = {tuple(x.unsqueeze(-1).shape)}       ← 맨 뒤에 추가")


# ── 3. 브로드캐스팅 ─────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("3. 브로드캐스팅 — 크기가 달라도 알아서 늘려준다")
print("=" * 60)

mat = torch.ones(3, 4)
row = torch.tensor([10.0, 20.0, 30.0, 40.0])   # shape (4,)

print(f"mat.shape={tuple(mat.shape)}, row.shape={tuple(row.shape)}")
print(f"mat + row =\n{mat + row}")
print("\n(4,) 짜리가 3줄로 자동 복제됐다. 뒤쪽 축부터 맞춰보고")
print("크기가 같거나 한쪽이 1이면 늘려준다. 이 규칙이 브로드캐스팅의 전부.")

# mean pooling에서 attention_mask를 곱할 때 이 규칙을 쓴다:
#   hidden (B, L, H) * mask (B, L, 1) → mask가 H축으로 768번 복제됨
mask = torch.tensor([[1, 1, 1, 0, 0]]).unsqueeze(-1)     # (1, 5, 1)
hid  = torch.ones(1, 5, 4)                               # (1, 5, 4)
print(f"\nhid {tuple(hid.shape)} * mask {tuple(mask.shape)} = {tuple((hid * mask).shape)}")
print("→ 패딩 토큰(mask=0)을 0으로 만들어 평균에서 빼는 데 쓴다.")


# ── 4. 행렬곱과 코사인 유사도 ───────────────────────────────────────────────

print("\n" + "=" * 60)
print("4. 코사인 유사도 — 이 프로젝트 검색의 핵심 연산")
print("=" * 60)

# 768차원 문서 벡터 5개와 질의 벡터 1개 (실제 chroma에 들어있는 것과 같은 모양)
docs  = torch.randn(5, 768)
query = torch.randn(768)

# 방법 A: 정의 그대로 — 내적을 각 노름으로 나눈다
cos_a = (docs @ query) / (docs.norm(dim=1) * query.norm())

# 방법 B: PyTorch 내장
cos_b = F.cosine_similarity(docs, query.unsqueeze(0), dim=1)

# 방법 C: 먼저 정규화(길이를 1로) 해두고 그냥 내적
#         ← 실전에서 쓰는 방식. 문서 벡터를 미리 정규화해두면
#           검색할 때마다 나눗셈을 반복할 필요가 없다.
docs_n  = F.normalize(docs, p=2, dim=1)
query_n = F.normalize(query, p=2, dim=0)
cos_c   = docs_n @ query_n

print(f"A 정의대로  : {cos_a}")
print(f"B 내장 함수 : {cos_b}")
print(f"C 정규화+내적: {cos_c}")
print(f"\n셋이 같은가? {torch.allclose(cos_a, cos_b, atol=1e-6) and torch.allclose(cos_a, cos_c, atol=1e-6)}")

print("\nChromaDB는 이 값을 그대로 주지 않고 'distance = 1 - cosine' 으로 준다:")
print(f"  cosine   {cos_c[0]:.4f}  →  distance {1 - cos_c[0]:.4f}")
print("retriever.py가 distances를 받아 순위를 매기는 게 바로 이 값이다.")

# (5, 768) @ (768,) → (5,)  : 축이 어떻게 사라지는지 보자
print(f"\n{tuple(docs_n.shape)} @ {tuple(query_n.shape)} = {tuple(cos_c.shape)}")
print("맞닿는 768 축이 곱해지며 사라진다. 이게 matmul 규칙의 전부다.")

# 문서 5개 vs 질의 3개를 한 번에 → (3, 5) 유사도 행렬
queries_n = F.normalize(torch.randn(3, 768), dim=1)
sim_matrix = queries_n @ docs_n.T
print(f"\n질의 3개 한꺼번에: {tuple(queries_n.shape)} @ {tuple(docs_n.T.shape)} = {tuple(sim_matrix.shape)}")
print("이 '유사도 행렬'이 07_contrastive.py의 손실 함수에서 그대로 다시 나온다.")


# ── 5. 텐서 ↔ 파이썬/넘파이 ─────────────────────────────────────────────────

print("\n" + "=" * 60)
print("5. 값 꺼내기")
print("=" * 60)

loss = torch.tensor(0.3142)
print(f"loss           = {loss}          ← 0차원 텐서(스칼라)")
print(f"loss.item()    = {loss.item():.4f}              ← 파이썬 float")
print(f"cos_c.tolist() = {[round(v, 3) for v in cos_c.tolist()]}")
print(f"cos_c.numpy()  = {cos_c.numpy().round(3)}   ← 메모리 공유(복사 아님)")

print("\n학습 루프에서 loss를 로그로 남길 땐 반드시 .item()을 쓴다.")
print("텐서 그대로 리스트에 쌓으면 연산 그래프가 통째로 붙어 메모리가 샌다(02에서 설명).")


# ── 연습 과제 ──────────────────────────────────────────────────────────────
#
# 1) docs를 (5, 768)에서 (5, 4)로 줄이고 값을 직접 지정해서,
#    코사인 유사도가 1.0 / 0.0 / -1.0 이 되는 쌍을 각각 만들어보라.
#    (힌트: 같은 방향 / 직교 / 반대 방향)
#
# 2) sim_matrix를 만들 때 docs_n.T 를 빼면 어떤 에러가 나는가?
#    에러 메시지의 숫자가 어느 축을 가리키는지 확인해보라.
#
# 3) mean pooling을 흉내내보자. hidden (2, 5, 4)와 mask (2, 5)를 만들고,
#    mask=1인 토큰만 평균낸 (2, 4) 결과를 구하라.
#    단순히 .mean(dim=1)을 쓰면 왜 틀리는가?
#    → 정답은 06_encoder.py에 있다.
