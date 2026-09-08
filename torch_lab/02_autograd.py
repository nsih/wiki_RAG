"""02. autograd — PyTorch가 딥러닝 프레임워크인 이유

신경망 학습은 결국 "손실을 줄이는 방향으로 가중치를 조금씩 옮기는 것"이다.
그 '방향'이 그래디언트(기울기)이고, autograd가 이걸 자동으로 계산해준다.

핵심 개념 3개:
    1. requires_grad=True 인 텐서로 한 연산은 그래프로 기록된다
    2. loss.backward() 를 부르면 그래프를 거꾸로 타며 .grad 를 채운다
    3. .grad 는 누적된다 → 매 스텝 zero_grad() 로 비워야 한다

실행: python torch_lab/02_autograd.py
"""

import torch

torch.set_num_threads(2)
torch.manual_seed(42)


# ── 1. 그래디언트가 뭔지 손으로 확인 ────────────────────────────────────────

print("=" * 60)
print("1. backward() 가 하는 일")
print("=" * 60)

x = torch.tensor(3.0, requires_grad=True)
y = x ** 2                     # y = x²

print(f"x = {x.item()}, y = x² = {y.item()}")
print(f"y.grad_fn = {y.grad_fn}   ← '이 텐서는 거듭제곱으로 만들어졌다'는 기록")

y.backward()                   # dy/dx 를 계산해서 x.grad 에 넣어라

print(f"\nx.grad = {x.grad.item()}")
print("수학적으로 dy/dx = 2x = 2×3 = 6. 정확히 일치한다.")
print("autograd는 미분 공식을 외운 게 아니라, 연산마다 기록해둔 미분 규칙을")
print("연쇄법칙으로 거꾸로 이어붙인 것뿐이다.")


# ── 2. 그래디언트는 '누적'된다 ──────────────────────────────────────────────

print("\n" + "=" * 60)
print("2. zero_grad() 가 왜 필요한가 (가장 흔한 버그)")
print("=" * 60)

w = torch.tensor(2.0, requires_grad=True)

for i in range(3):
    loss = w ** 2              # 매번 똑같은 계산
    loss.backward()
    print(f"  {i+1}번째 backward 후 w.grad = {w.grad.item():5.1f}   (매번 4가 더해진다)")

print("\n비우지 않으면 4 → 8 → 12 로 계속 쌓인다.")

w.grad = None                  # 또는 w.grad.zero_()
loss = w ** 2
loss.backward()
print(f"grad를 비운 뒤 backward → w.grad = {w.grad.item()}   (정상)")

print("\n그래서 학습 루프에는 반드시 optimizer.zero_grad() 가 들어간다.")
print("(누적 자체는 버그가 아니라 기능이다 — 배치를 쪼개 여러 번 backward한 뒤")
print(" 한 번에 step하는 'gradient accumulation'이 이 성질을 이용한다.")
print(" RAM이 부족할 때 배치 크기를 키우는 효과를 내는 기법이다.)")


# ── 3. 옵티마이저 없이 손으로 경사하강 ──────────────────────────────────────

print("\n" + "=" * 60)
print("3. 경사하강 — optimizer 없이 직접")
print("=" * 60)

# 정답: y = 3x + 2 라는 관계를 데이터만 보고 알아맞히기
x_data = torch.tensor([1.0, 2.0, 3.0, 4.0])
y_data = torch.tensor([5.0, 8.0, 11.0, 14.0])      # 3x + 2

w = torch.tensor(0.0, requires_grad=True)          # 아무 값에서나 시작
b = torch.tensor(0.0, requires_grad=True)
lr = 0.01

print(f"목표: w=3.0, b=2.0  /  시작: w={w.item()}, b={b.item()}\n")

for step in range(200):
    y_pred = w * x_data + b                        # 순전파
    loss = ((y_pred - y_data) ** 2).mean()         # MSE 손실

    loss.backward()                                # 역전파 → w.grad, b.grad

    with torch.no_grad():                          # 이 갱신 자체는 미분 대상이 아니다
        w -= lr * w.grad                           # 기울기의 '반대' 방향으로 한 걸음
        b -= lr * b.grad

    w.grad = None                                  # 다음 스텝을 위해 비우기
    b.grad = None

    if step % 40 == 0 or step == 199:
        print(f"  step {step:3d}  loss={loss.item():8.4f}  w={w.item():.4f}  b={b.item():.4f}")

print(f"\n최종: w={w.item():.4f}, b={b.item():.4f}")
print("b가 2.0에 덜 도달했는데, 스텝을 더 돌리면 수렴한다.")
print("(SGD는 이렇게 느리다. 그래서 실전에서는 AdamW를 쓴다 → 03에서)")

print("\n이 4줄이 모든 딥러닝 학습의 전부다:")
print("    loss = ...          # 순전파: 예측하고 얼마나 틀렸는지 잰다")
print("    loss.backward()     # 역전파: 각 파라미터가 손실에 얼마나 기여했나")
print("    param -= lr * grad  # 갱신:   틀린 만큼 반대로 민다")
print("    grad = None         # 청소:   다음 스텝을 위해 비운다")


# ── 4. no_grad() 와 detach() ────────────────────────────────────────────────

print("\n" + "=" * 60)
print("4. 그래프를 끊는 두 가지 방법")
print("=" * 60)

a = torch.tensor([1.0, 2.0], requires_grad=True)

out1 = a * 2
print(f"그냥 연산       : requires_grad={out1.requires_grad}  ← 그래프에 기록됨")

with torch.no_grad():
    out2 = a * 2
print(f"no_grad() 안    : requires_grad={out2.requires_grad}  ← 기록 안 함")

out3 = (a * 2).detach()
print(f"detach()        : requires_grad={out3.requires_grad}  ← 기록에서 떼어냄")

print("""
언제 쓰나:
  no_grad()  — 평가/추론 전체를 감쌀 때. evaluate.py와 reindex.py가 이걸 쓴다.
               그래프를 안 만드니 메모리가 훨씬 덜 들고 속도도 빠르다.
               여유 RAM 1.4GB인 이 서버에서는 선택이 아니라 필수.
  detach()   — 텐서 하나만 떼어낼 때. loss.item() 도 사실상 같은 일을 한다.

흔한 메모리 누수:
    losses.append(loss)          # ✗ 연산 그래프가 통째로 붙어 따라온다
    losses.append(loss.item())   # ✓ 숫자만 뽑는다
""")


# ── 5. 벡터를 미분하면? ─────────────────────────────────────────────────────

print("=" * 60)
print("5. backward() 는 스칼라에서만 부를 수 있다")
print("=" * 60)

v = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
out = v ** 2                                   # shape (3,)

try:
    out.backward()
except RuntimeError as e:
    print(f"out.backward() → RuntimeError:\n  {str(e).splitlines()[0]}")

print("\n'무엇을 무엇으로 미분하라'가 애매하기 때문이다.")
out.sum().backward()                           # 스칼라로 만들어서 미분
print(f"out.sum().backward() → v.grad = {v.grad}   (= 2v)")

print("\n그래서 손실 함수는 항상 .mean() 이나 .sum() 으로 끝나 스칼라를 낸다.")
print("위의 MSE도, 07의 cross_entropy도 전부 마지막에 배치 평균을 낸다.")


# ── 연습 과제 ──────────────────────────────────────────────────────────────
#
# 1) 3번의 lr을 0.1로 올리면? 0.5로 올리면?
#    loss가 nan이 되는 지점을 찾아보라. 이게 '학습률이 너무 크다'의 실체다.
#    (train_embedder.py에서 lr=2e-5 같은 작은 값을 쓰는 이유)
#
# 2) w의 시작값을 100.0으로 바꾸면 몇 스텝만에 수렴하는가?
#
# 3) 3번 루프에서 `w.grad = None` 두 줄을 지우면 어떻게 되는가?
#    loss가 어떻게 변하는지 관찰하고, 왜 그런지 2번 절과 연결해 설명해보라.
