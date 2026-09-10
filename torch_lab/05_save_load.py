"""05. 저장과 로드 — 학습한 것을 잃지 않는 법

03에서 학습한 모델은 스크립트가 끝나면 사라진다.
Stage 3에서 에폭당 15~30분씩 학습할 텐데 그걸 매번 다시 할 수는 없다.

이 파일에서 다루는 것:
    state_dict()        — 모델의 '내용물'만 뽑아낸 dict
    torch.save/load     — 그걸 파일로
    train() / eval()    — 학습 모드와 추론 모드의 차이 (수치로 확인)
    체크포인트          — 학습 재개까지 가능한 형태

실행: python torch_lab/05_save_load.py
"""

import tempfile
from pathlib import Path

import torch
import torch.nn as nn

torch.set_num_threads(2)
torch.manual_seed(42)

TMP = Path(tempfile.mkdtemp(prefix="torch_lab_"))
print(f"임시 저장 경로: {TMP}\n")


class SmallNet(nn.Module):
    """dropout이 있는 작은 모델 — 3절에서 train/eval 차이를 보려고 넣었다."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 8)
        self.drop = nn.Dropout(p=0.5)     # 학습 중 절반을 무작위로 0으로
        self.fc2 = nn.Linear(8, 2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.drop(x)
        return self.fc2(x)


# ── 1. state_dict — 모델의 내용물 ───────────────────────────────────────────

print("=" * 60)
print("1. state_dict() — 가중치만 담은 dict")
print("=" * 60)

model = SmallNet()
sd = model.state_dict()

print(f"타입: {type(sd).__name__}")
print(f"\n{'키':16s} {'shape':>14s} {'원소 수':>8s}")
print("-" * 42)
for k, v in sd.items():
    print(f"{k:16s} {str(tuple(v.shape)):>14s} {v.numel():>8d}")
print("-" * 42)
print(f"{'합계':16s} {'':>14s} {sum(v.numel() for v in sd.values()):>8d}")

print("""
state_dict는 '파라미터 이름 → 텐서' 매핑일 뿐이다.
모델 구조(클래스 정의)는 들어있지 않다. 그래서 로드할 때는
같은 구조의 모델을 먼저 만들고 그 안에 부어넣어야 한다.

Dropout은 학습할 파라미터가 없어서 state_dict에 안 나온다.
""")


# ── 2. 저장과 로드 ──────────────────────────────────────────────────────────

print("=" * 60)
print("2. torch.save / load — 구조가 아니라 '내용물'을 저장한다")
print("=" * 60)

x = torch.randn(3, 4)
model.eval()                              # 추론 모드로 고정 (3절에서 이유 설명)
with torch.no_grad():
    before = model(x)

path = TMP / "model.pt"
torch.save(model.state_dict(), path)      # ★ model 자체가 아니라 state_dict를 저장
print(f"저장: {path.name}  ({path.stat().st_size:,} bytes)")

# 로드 — 같은 구조를 먼저 만들고 부어넣는다
loaded = SmallNet()                       # 초기값은 랜덤 (다른 값)
loaded.eval()
with torch.no_grad():
    random_out = loaded(x)

loaded.load_state_dict(torch.load(path))
loaded.eval()
with torch.no_grad():
    after = loaded(x)

print(f"\n원본 출력          : {before[0].tolist()}")
print(f"로드 전(랜덤 초기값): {random_out[0].tolist()}")
print(f"로드 후            : {after[0].tolist()}")
print(f"\n원본과 완전히 같은가: {torch.equal(before, after)}")

print("""
왜 model 자체를 저장하지 않고 state_dict를 저장하나?
model을 통째로 pickle하면 클래스 정의 경로까지 파일에 박힌다.
나중에 파일을 옮기거나 코드를 리팩터링하면 로드가 깨진다.
state_dict는 순수한 숫자 뭉치라 그런 문제가 없다. 실무 표준이다.
""")


# ── 3. train() vs eval() — 잊으면 조용히 틀리는 부분 ────────────────────────

print("=" * 60)
print("3. model.train() / model.eval() — 수치로 확인")
print("=" * 60)

model.train()                             # 학습 모드
outs = [model(x)[0, 0].item() for _ in range(5)]
print(f"train() 모드에서 같은 입력을 5번:")
print(f"  {[round(v, 4) for v in outs]}")
print(f"  → 매번 다르다. Dropout이 무작위로 절반을 죽이기 때문.")

model.eval()                              # 추론 모드
outs = [model(x)[0, 0].item() for _ in range(5)]
print(f"\neval() 모드에서 같은 입력을 5번:")
print(f"  {[round(v, 4) for v in outs]}")
print(f"  → 항상 같다. Dropout이 꺼진다.")

print(f"\n현재 상태: model.training = {model.training}")

print("""
Dropout과 BatchNorm은 학습 때와 추론 때 '동작이 다른' 층이다.
  Dropout   학습: 무작위로 0으로 만듦 / 추론: 아무것도 안 함
  BatchNorm 학습: 현재 배치 통계 사용 / 추론: 누적 통계 사용

★ eval()을 빼먹으면 에러가 안 나고 그냥 결과가 이상해진다.
  가장 찾기 어려운 버그 중 하나다. 평가 코드에는 반드시 넣는다.

eval() 과 no_grad() 는 다른 것이다. 둘 다 필요하다:
    model.eval()              # 층의 '동작'을 추론 모드로
    with torch.no_grad():     # 연산 그래프를 '기록하지 않음' (메모리 절약)
        out = model(x)
""")


# ── 4. 체크포인트 — 학습을 이어서 하려면 ────────────────────────────────────

print("=" * 60)
print("4. 체크포인트 — 옵티마이저 상태까지 저장해야 재개된다")
print("=" * 60)

m = SmallNet()
opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()
target = torch.randn(3, 2)

for _ in range(10):                       # 10스텝 학습
    loss = loss_fn(m(x), target)
    loss.backward()
    opt.step()
    opt.zero_grad()

# AdamW는 파라미터마다 m, v 를 들고 있다 (04번 문서 참조)
opt_sd = opt.state_dict()
print(f"옵티마이저 state 항목 수: {len(opt_sd['state'])}개 (파라미터 개수와 같다)")
first = next(iter(opt_sd["state"].values()))
print(f"각 항목이 들고 있는 것: {list(first.keys())}")
print("  step      = 몇 번째 스텝인지 (편향 보정 m/(1-β₁^t) 에 쓰인다)")
print("  exp_avg   = m (1차 모멘트)")
print("  exp_avg_sq= v (2차 모멘트)")

ckpt_path = TMP / "checkpoint.pt"
torch.save({
    "epoch": 3,
    "model": m.state_dict(),
    "optimizer": opt.state_dict(),        # ★ 이걸 빼면 재개 시 m, v 가 0으로 리셋된다
    "loss": loss.item(),
}, ckpt_path)
print(f"\n체크포인트 저장: {ckpt_path.stat().st_size:,} bytes")

ckpt = torch.load(ckpt_path)
m2 = SmallNet()
opt2 = torch.optim.AdamW(m2.parameters(), lr=1e-3)
m2.load_state_dict(ckpt["model"])
opt2.load_state_dict(ckpt["optimizer"])
print(f"복원: epoch={ckpt['epoch']}, loss={ckpt['loss']:.6f}")

# 재개 후 한 스텝이 원본과 같은지 확인
def one_step(model_, opt_, seed):
    # ★ 시드를 맞춰야 한다. 이 모델에는 Dropout이 있어서 학습 모드에서는
    #   매 forward마다 무작위 마스크가 다르고, 그러면 그래디언트도 달라진다.
    #   시드를 안 맞추면 체크포인트가 정상인데도 결과가 달라 보인다.
    torch.manual_seed(seed)
    loss_fn(model_(x), target).backward()
    opt_.step()
    opt_.zero_grad()
    return model_.fc1.weight.clone()

w_orig = one_step(m, opt, seed=777)
w_resumed = one_step(m2, opt2, seed=777)
print(f"재개 후 한 스텝이 원본과 동일한가: {torch.equal(w_orig, w_resumed)}")

# 시드를 안 맞추면? — 체크포인트는 멀쩡한데 결과가 달라진다
m3 = SmallNet(); opt3 = torch.optim.AdamW(m3.parameters(), lr=1e-3)
m3.load_state_dict(ckpt["model"]); opt3.load_state_dict(ckpt["optimizer"])
w_diff_seed = one_step(m3, opt3, seed=999)
print(f"시드만 다르게 했을 때        : {torch.equal(w_orig, w_diff_seed)}"
      f"   ← Dropout 마스크가 달라서")

print("""
옵티마이저 상태를 빼고 저장하면 재개했을 때 m, v 가 0에서 다시 시작한다.
그러면 처음 몇 스텝 동안 편향 보정이 크게 걸려 갱신이 튄다.
Stage 3에서 학습이 중간에 죽어도 이어갈 수 있게 이 형태로 저장한다.

★ 위 두 줄이 보여주는 것: '완전한 재개'에는 3가지가 다 필요하다.
    1. 모델 가중치      (model.state_dict)
    2. 옵티마이저 상태  (optimizer.state_dict — m, v, step)
    3. 난수 상태        (Dropout, 셔플 순서)

  1, 2를 제대로 저장해도 3이 다르면 결과가 달라진다.
  엄밀한 재현이 필요하면 torch.get_rng_state() 도 함께 저장한다.
  실무에서는 3까지는 보통 포기하고 1, 2만 저장한다 — 학습 결과에
  큰 영향이 없고, 데이터 순서까지 복원하려면 훨씬 복잡해지기 때문이다.
""")


# ── 5. safetensors — transformers가 쓰는 형식 ───────────────────────────────

print("=" * 60)
print("5. .pt 와 .safetensors")
print("=" * 60)

hf_cache = Path.home() / ".cache/huggingface/hub"
found = list(hf_cache.glob("models--jhgan--ko-sroberta-multitask/snapshots/*/*.safetensors"))
if found:
    f = found[0]
    print(f"우리 임베딩 모델의 가중치 파일:")
    print(f"  {f.name}  ({f.stat().st_size / 1024**2:.1f} MB)")

print("""
torch.save 는 내부적으로 pickle을 쓴다. pickle은 임의의 파이썬 코드를 실행할 수 있어서
신뢰할 수 없는 파일을 로드하면 위험하다.

safetensors 는 그래서 나온 형식이다. 순수한 텐서 데이터만 담아 코드 실행이 불가능하고,
필요한 텐서만 골라 읽을 수 있어 로딩도 빠르다.
HuggingFace 모델은 이제 대부분 이 형식이고, 우리 모델도 model.safetensors 다.

Stage 3에서는 AutoModel.from_pretrained() / save_pretrained() 를 쓰므로
이 변환을 직접 신경 쓸 일은 없다. 다만 저장된 파일이 왜 .safetensors 인지는 알아두자.
""")


# ── 6. 재현성 ───────────────────────────────────────────────────────────────

print("=" * 60)
print("6. 시드 고정 — 같은 결과를 다시 얻으려면")
print("=" * 60)

def make():
    return nn.Linear(3, 2).weight.detach().flatten()[:3].tolist()

torch.manual_seed(0); a = make()
torch.manual_seed(0); b = make()
c = make()                                # 시드 재설정 없이

print(f"seed(0) 후 생성  : {[round(v,4) for v in a]}")
print(f"seed(0) 다시     : {[round(v,4) for v in b]}   같은가: {a == b}")
print(f"시드 재설정 없이 : {[round(v,4) for v in c]}   같은가: {a == c}")

print("""
manual_seed는 '그 시점부터의 난수 순서'를 고정한다.
스크립트 맨 위에 한 번 부르면 전체 실행이 재현된다.

Stage 3에서 시드를 고정하는 이유:
  하이퍼파라미터를 바꿔가며 비교할 때, 초기값이 매번 다르면
  '설정이 좋아서 나아진 건지 운이 좋아서 나아진 건지' 구분할 수 없다.
""")

print(f"\n(임시 파일 정리: rm -rf {TMP})")


# ── 연습 과제 ──────────────────────────────────────────────────────────────
#
# 1) 2절에서 load_state_dict 를 하기 전에 SmallNet의 fc1을 nn.Linear(4, 16)으로
#    바꾸면 어떤 에러가 나는가? 에러 메시지가 무엇을 알려주는가?
#
# 2) 3절에서 model.eval() 없이 평가하면 Recall 같은 지표가 어떻게 될까?
#    Dropout p=0.5 라면 매번 다른 점수가 나올 것이다. 직접 확인해보라.
#
# 3) 4절에서 opt.state_dict() 저장을 빼고 재개하면 첫 스텝의 갱신량이
#    얼마나 달라지는가? (힌트: 04번 문서의 편향 보정 m̂ = m/(1-β₁^t) 에서 t=1)
#
# 4) torch.save(model, path) 로 모델을 통째로 저장한 뒤,
#    SmallNet 클래스 이름을 바꾸고 로드하면 어떻게 되는가?
