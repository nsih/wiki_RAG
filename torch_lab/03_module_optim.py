"""03. nn.Module과 옵티마이저 — 표준 학습 루프

02에서는 w, b를 직접 만들고 손으로 갱신했다. 파라미터가 2개니까 가능했다.
실제 모델은 파라미터가 1억 개다. 그걸 손으로 관리할 수는 없다.

PyTorch는 이 문제를 두 도구로 나눠 해결한다:
    nn.Module   — 파라미터를 모아서 들고 있는 그릇 (모델)
    optimizer   — 그 파라미터들을 갱신하는 규칙 (SGD, AdamW, ...)

이 파일에서 나오는 학습 루프가 Stage 3 train_embedder.py의 골격 그대로다.

실행: python torch_lab/03_module_optim.py
"""

import torch
import torch.nn as nn

torch.set_num_threads(2)
torch.manual_seed(42)


# ── 1. nn.Module — 02의 w, b를 그릇에 담기 ──────────────────────────────────

print("=" * 60)
print("1. nn.Module — 모델은 '파라미터를 들고 있는 객체'다")
print("=" * 60)


class LinearModel(nn.Module):
    """y = wx + b 를 nn.Module로 표현. 02와 완전히 같은 모델이다."""

    def __init__(self):
        super().__init__()                # 반드시 먼저 호출 (파라미터 등록 준비)
        self.fc = nn.Linear(1, 1)         # 입력 1차원 → 출력 1차원

    def forward(self, x):
        return self.fc(x)                 # 순전파: 입력을 받아 예측을 낸다


model = LinearModel()

print("model 구조:")
print(model)

print("\nnn.Linear(1, 1) 안에 뭐가 있나:")
for name, p in model.named_parameters():
    print(f"  {name:12s} shape={tuple(p.shape)}  값={p.data.flatten().tolist()}  requires_grad={p.requires_grad}")

print("""
weight와 bias가 자동으로 만들어졌다. 02에서 손으로 만든 w, b와 같은 것이다.
차이는 requires_grad=True 가 기본으로 붙고, model.parameters()로 한꺼번에
꺼낼 수 있다는 점이다. 이게 nn.Module의 존재 이유다.

주의: nn.Linear(in, out) 의 weight shape은 (out, in) 이다. 순서가 뒤집혀 있다.
      y = x @ weight.T + bias 로 계산하기 때문이다.
""")

# forward 는 직접 부르지 않고 model(x) 로 호출한다 (내부 훅이 동작하도록)
x = torch.tensor([[1.0], [2.0]])          # shape (2, 1) — 배치 2개
print(f"model(x) shape: {tuple(model(x).shape)}   ← (배치 2, 출력 1)")


# ── 2. 파라미터를 세는 법 ───────────────────────────────────────────────────

print("\n" + "=" * 60)
print("2. 파라미터 개수 세기 — 메모리 계산의 출발점")
print("=" * 60)


class MLP(nn.Module):
    """조금 더 현실적인 모델: 3층 신경망"""

    def __init__(self, d_in=768, d_hidden=256, d_out=1):
        super().__init__()
        self.layer1 = nn.Linear(d_in, d_hidden)
        self.layer2 = nn.Linear(d_hidden, d_hidden)
        self.layer3 = nn.Linear(d_hidden, d_out)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.act(self.layer1(x))
        x = self.act(self.layer2(x))
        return self.layer3(x)


mlp = MLP()

total = sum(p.numel() for p in mlp.parameters())
trainable = sum(p.numel() for p in mlp.parameters() if p.requires_grad)

print(f"{'파라미터':14s} {'shape':>16s} {'개수':>10s}")
print("-" * 44)
for name, p in mlp.named_parameters():
    print(f"{name:14s} {str(tuple(p.shape)):>16s} {p.numel():>10,d}")
print("-" * 44)
print(f"{'합계':14s} {'':>16s} {total:>10,d}")

print(f"\n전체 {total:,}개 / 학습 대상 {trainable:,}개")
print(f"fp32 기준 가중치 메모리: {total * 4 / 1024**2:.1f} MB")
print(f"AdamW로 학습 시 필요량 : {total * 16 / 1024**2:.1f} MB  (가중치+그래디언트+상태2개)")

print("""
파라미터 1개당 학습 시 16바이트가 필요하다:
    가중치 4B + 그래디언트 4B + AdamW의 exp_avg 4B + exp_avg_sq 4B

PLAN.md의 '110M x 16B = 1,760MB' 계산이 바로 이것이다.
그래서 여유 RAM 1.4GB인 이 서버에서 전체 파인튜닝이 불가능하다.
""")


# ── 3. 층 얼리기 (freeze) — Stage 3의 핵심 기법 ─────────────────────────────

print("=" * 60)
print("3. requires_grad=False — 층을 얼려 메모리를 줄인다")
print("=" * 60)

# layer1을 얼린다 = "이 층은 학습하지 않는다"
for p in mlp.layer1.parameters():
    p.requires_grad = False

frozen = sum(p.numel() for p in mlp.parameters() if not p.requires_grad)
trainable = sum(p.numel() for p in mlp.parameters() if p.requires_grad)

print(f"layer1 동결 후:")
print(f"  동결    {frozen:>10,d}개  ← 그래디언트/옵티마이저 상태 불필요")
print(f"  학습    {trainable:>10,d}개")
print(f"\n학습 시 메모리: {total*4/1024**2:.1f} MB (가중치 전체)"
      f" + {trainable*12/1024**2:.1f} MB (학습분 그래디언트+상태)"
      f" = {(total*4 + trainable*12)/1024**2:.1f} MB")
print(f"전체 학습이었다면 {total*16/1024**2:.1f} MB → "
      f"{(1 - (total*4 + trainable*12)/(total*16))*100:.0f}% 절감")

print("""
가중치는 동결분까지 전부 메모리에 있어야 한다 (순전파에 쓰이므로).
줄어드는 건 그래디언트와 옵티마이저 상태다. 그게 파라미터당 12B 중 대부분이다.

Stage 3에서 이걸 그대로 쓴다:
    for name, p in model.named_parameters():
        p.requires_grad = not is_frozen(name, freeze_below=8)
""")

mlp = MLP()   # 다음 실험을 위해 초기화


# ── 4. 옵티마이저 — 02의 문제를 다시 풀어본다 (예상과 다른 결과가 나온다) ──────────────────────────

print("=" * 60)
print("4. 옵티마이저 — SGD vs AdamW, 통념을 깨는 실험")
print("=" * 60)

# 02와 완전히 같은 문제: y = 3x + 2
x_data = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
y_data = torch.tensor([[5.0], [8.0], [11.0], [14.0]])

loss_fn = nn.MSELoss()          # 02에서 손으로 쓴 ((pred-y)**2).mean() 과 같다


def train(optimizer_name, steps=200, lr=0.01):
    """같은 조건에서 옵티마이저만 바꿔 200스텝 학습"""
    torch.manual_seed(42)                       # 같은 초기값에서 출발
    m = LinearModel()

    if optimizer_name == "SGD":
        opt = torch.optim.SGD(m.parameters(), lr=lr)
    elif optimizer_name == "SGD+momentum":
        opt = torch.optim.SGD(m.parameters(), lr=lr, momentum=0.9)
    elif optimizer_name == "AdamW":
        opt = torch.optim.AdamW(m.parameters(), lr=lr)

    for _ in range(steps):
        loss = loss_fn(m(x_data), y_data)       # 1. 순전파 + 손실
        loss.backward()                         # 2. 역전파
        opt.step()                              # 3. 갱신  ← 02의 'param -= lr*grad'
        opt.zero_grad()                         # 4. 청소  ← 02의 'grad = None'

    w = m.fc.weight.item()
    b = m.fc.bias.item()
    return loss.item(), w, b


print(f"목표: w=3.0, b=2.0   (200스텝, lr=0.01, 동일 초기값)\n")
print(f"{'옵티마이저':16s} {'최종 loss':>12s} {'w':>8s} {'b':>8s}")
print("-" * 48)
for name in ["SGD", "SGD+momentum", "AdamW"]:
    loss_v, w, b = train(name)
    print(f"{name:16s} {loss_v:>12.6f} {w:>8.4f} {b:>8.4f}")

print("""
결과가 예상과 반대다. AdamW가 제일 나쁘다.

이건 버그가 아니라 Adam의 작동 방식 때문이다. 확인해보자.
""")

# ── Adam의 스텝 크기는 lr로 상한이 걸린다 ──
print("[실험] AdamW의 그래디언트 대비 실제 이동량 (lr=0.01)")
torch.manual_seed(42)
m = LinearModel()
opt = torch.optim.AdamW(m.parameters(), lr=0.01)
prev = m.fc.weight.item()
for i in range(5):
    loss = loss_fn(m(x_data), y_data)
    loss.backward()
    g = m.fc.weight.grad.item()
    opt.step()
    opt.zero_grad()
    now = m.fc.weight.item()
    print(f"  step{i}  그래디언트={g:9.3f}  →  실제 이동={now - prev:+.6f}")
    prev = now

print("""
그래디언트가 -39인데 이동량은 +0.0099다. 거의 정확히 lr 값이다.

Adam은 그래디언트를 '자기 자신의 최근 크기'로 나눈다:
    이동량 ≈ lr x (grad / sqrt(grad^2 평균))  ≈  lr x ±1

즉 그래디언트가 -39든 -3이든 한 스텝에 lr만큼만 움직인다.
200스텝 x 0.01 = 최대 2.0밖에 못 간다. 그런데 w는 0.76에서 3.0까지
2.24를 가야 한다. 애초에 도달이 불가능했다.
""")

print("[실험] lr을 올리면 AdamW도 수렴한다")
print(f"{'옵티마이저':16s} {'lr':>7s} {'최종 loss':>12s} {'w':>8s} {'b':>8s}")
print("-" * 56)
for lr in [0.01, 0.05, 0.1, 0.5]:
    loss_v, w, b = train("AdamW", lr=lr)
    print(f"{'AdamW':16s} {lr:>7} {loss_v:>12.6f} {w:>8.4f} {b:>8.4f}")

print("""
정리 — 언제 뭘 쓰나:

  SGD   그래디언트가 크면 크게 움직인다. 이 장난감 문제처럼
        파라미터 2개에 스케일이 균일하면 오히려 빠르다.

  Adam  그래디언트 '크기'를 무시하고 '방향'만 쓴다.
        실제 신경망은 층마다 그래디언트 크기가 수백 배 차이 나는데,
        그때 SGD는 lr 하나로 전부 맞출 수 없어 발산하거나 멈춘다.
        Adam은 파라미터마다 알아서 정규화하므로 그 문제가 없다.

  → 장난감 문제에서는 SGD가, 실제 모델에서는 Adam이 이긴다.
    "Adam이 항상 빠르다"는 틀린 통념이다.

★ 이 성질이 Stage 3의 lr=2e-5 를 설명한다.
  Adam에서는 lr이 사실상 '스텝당 최대 이동량'이다.
  2e-5 = "가중치 하나를 한 스텝에 0.00002 이상 움직이지 마라"
  사전학습으로 잘 맞춰진 가중치를 망가뜨리지 않으려고 의도적으로 작게 주는 것이다.
  처음부터 학습(from scratch)이라면 1e-3 같은 훨씬 큰 값을 쓴다.

  대신 파라미터당 상태 2개(8B)를 들고 있어야 한다 — 2절에서 계산한 그 8바이트다.
  공짜가 아니라 '메모리를 내고 안정성을 산' 거래다.
""")


# ── 5. 표준 학습 루프 ───────────────────────────────────────────────────────

print("=" * 60)
print("5. 표준 학습 루프 — 이 5줄이 전부다")
print("=" * 60)

print("""
    for batch in loader:                  # 04_data.py에서 만든다
        loss = loss_fn(model(x), y)       # (1) 순전파: 예측하고 틀린 정도를 잰다
        loss.backward()                   # (2) 역전파: 그래디언트 계산
        optimizer.step()                  # (3) 갱신:   파라미터를 민다
        optimizer.zero_grad()             # (4) 청소:   그래디언트 비우기

Stage 3 train_embedder.py 와 1:1로 대응한다:

    이 파일                        train_embedder.py
    ------------------------------ ----------------------------------
    model = LinearModel()          model = AutoModel.from_pretrained(...)
    loss_fn = nn.MSELoss()         loss = contrastive_loss(q, d)   # 07에서 직접 구현
    loss_fn(model(x), y)           mean_pool(model(**tok(texts)))  # 06에서 배운다
    opt.step(); opt.zero_grad()    똑같음
    (반복 대상: x_data 4개)         for batch in loader             # 04에서 배운다

즉 남은 건 '데이터를 배치로 공급하는 법(04)'과
'텍스트를 벡터로 만드는 법(06)', '손실 함수(07)' 세 개뿐이다.
학습 루프 자체는 이미 다 배웠다.
""")

# 실제로 model.train() / model.eval() 이 필요한 이유는 05에서 다룬다
print(f"참고: model.training = {LinearModel().training}  (기본은 학습 모드) → 05에서 설명")


# ── 연습 과제 ──────────────────────────────────────────────────────────────
#
# 1) 4절 실험을 SGD의 lr을 0.5로 올려서 돌려보면 어떻게 되는가?
#    AdamW는 lr=0.5에서도 멀쩡한데 SGD는 왜 그렇지 않은가?
#    (힌트: 그래디언트가 -39일 때 SGD의 이동량은 얼마인가? 02의 연습 1번과 이어진다)
#
# 2) MLP의 d_hidden을 256 → 1024로 바꾸면 파라미터가 몇 배 늘어나는가?
#    선형이 아닌 이유를 layer2의 shape으로 설명해보라.
#
# 3) 3절에서 layer1 대신 layer3(마지막 층)을 얼리면 절감량이 어떻게 다른가?
#    Stage 3에서 '상위 층만 학습'하는 게 왜 하위 층을 얼리는 것인지 생각해보라.
#
# 4) 4절 train() 함수에서 opt.zero_grad() 를 지우면 결과가 어떻게 되는가?
#    02의 2절과 연결해서 예측한 뒤 실제로 확인해보라.
