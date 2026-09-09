# 01. PyTorch란 무엇인가

> **질문**: PyTorch는 그냥 파이썬에 임포트하는 패키지 같은 건가?

## 한 줄 답

맞지만 절반만 맞다. **파이썬은 조종석이고, 실제 계산 엔진은 C++로 컴파일된 별도 라이브러리다.**

---

## 1. 실측 — 파이썬 코드는 4%뿐

```bash
find venv/lib/python3.12/site-packages/torch -name '*.py'  -printf '%s\n' | awk '{s+=$1} END {print s/1048576" MB"}'
find venv/lib/python3.12/site-packages/torch -name '*.so*' -printf '%s\n' | awk '{s+=$1} END {print s/1048576" MB"}'
```

| 구성 | 크기 | 파일 수 |
|---|---|---|
| 파이썬 코드 (`.py`) | **39.4 MB** | 2,141개 |
| 컴파일된 C++ (`.so`) | **954.1 MB** | 13개 |

가장 큰 파일:

| 파일 | 크기 | 용도 |
|---|---|---|
| `libtorch_cuda.so` | 435 MB | GPU 계산 (이 서버에선 미사용) |
| `libtorch_cpu.so` | 430 MB | **CPU 계산 — 이 프로젝트가 쓰는 것** |
| `libtorch_python.so` | 31 MB | 파이썬 ↔ C++ 연결 |

## 2. 증거 — torch 함수에는 파이썬 소스가 없다

```python
>>> type(torch.matmul)
<class 'builtin_function_or_method'>
>>> inspect.getsource(torch.matmul)
TypeError: ... got builtin_function_or_method
```

파이썬으로 쓰인 함수가 아니라서 **소스를 볼 수 없다.** 이름만 파이썬에 등록돼 있고 실체는 C++ 함수다.

반면 진짜 파이썬인 부분도 있다:

```python
>>> inspect.getsource(nn.Linear.forward)
def forward(self, input: Tensor) -> Tensor:
    return F.linear(input, self.weight, self.bias)   # 다시 C++로 넘긴다
```

**PyTorch의 파이썬 층은 대체로 이런 얇은 껍데기다.**

## 3. 왜 이 구조인가 — 속도 실측

이 서버(2코어)에서 500×500 행렬곱 = 곱셈-덧셈 1.25억 회:

| 방식 | 시간 |
|---|---|
| `a @ b` (워밍업 후 20회 평균) | **4.42 ms** |
| 순수 파이썬 3중 루프 (100×100 환산) | 약 **7,500배** 느림 |

> ⚠️ 첫 측정에서 151 ms가 나왔는데 이건 **워밍업이 섞인 값**이었다.
> 벤치마크는 반드시 몇 번 돌린 뒤 재야 한다.

파이썬은 한 줄씩 해석하며 실행하는 언어라 숫자 계산에 근본적으로 부적합하다. 그래서 역할을 나눈다:

- **파이썬** — 무엇을 계산할지 설계 (모델 구조, 학습 루프, 데이터 흐름)
- **C++/CUDA** — 실제 계산 수행

`a @ b` 한 줄을 쓰면 파이썬은 손을 떼고, C++이 코어를 돌려 계산한 뒤 결과만 돌려준다.

## 4. 계층 구조

```
파이썬 코드        model(x)              <- 내가 쓰는 것
    |
libtorch_python.so 바인딩
    |
libtorch_cpu.so    CPU 계산              <- 이 서버가 쓰는 경로
libtorch_cuda.so   GPU 계산              <- 설치됐지만 죽어 있음
```

## 5. 실무 함의 3가지

**(1) 성능은 "파이썬 for문을 얼마나 안 쓰느냐"로 결정된다.**
텐서 1,000개를 파이썬 반복문으로 하나씩 처리하면 C++의 이점이 사라진다.
항상 **배치로 묶어 한 번에** 던진다.

```python
# 나쁨
for q in queries:
    sim = q @ docs.T

# 좋음 — 질의 3개를 한 번에
sims = queries @ docs.T      # (3, 768) @ (768, N) -> (3, N)
```

> **코드에 `for` 문 안에 텐서 연산이 있으면 일단 의심하라.**
> 대부분 배치 연산 한 줄로 바꿀 수 있고, 그게 PyTorch식 사고방식이다.

**(2) `venv/`가 5.7 GB인 이유.** torch만 1.2 GB이고 그중 435 MB가 쓰지도 않는 CUDA다. → [02번 문서](./02-cuda.md)

**(3) `torch.set_num_threads(2)`가 제어하는 건 파이썬 스레드가 아니라 C++ 쪽 스레드 풀이다.**
그래서 GIL과 무관하게 실제 병렬 계산이 된다. 2코어 서버에서 기본값을 두면 스레드 경합으로 오히려 느려진다.

## 6. "그냥 패키지"라기엔 더 있는 것

numpy 같은 순수 계산 라이브러리와 달리 PyTorch 안에는 이런 게 함께 있다:

- **autograd 엔진** — 연산 그래프를 기록하고 미분을 자동 역전파 (`torch_lab/02_autograd.py`)
- **컴파일러** — `torch.compile`. `torch/bin/ptxas`(36 MB)가 실제 GPU 어셈블러
- **자체 런타임** — 메모리 할당자, 스레드 풀, 커널 스케줄러

즉 **"파이썬으로 조종하는 별도의 계산 엔진"** 에 가깝다.

---

## 이 프로젝트와의 연결

- `torch.set_num_threads(2)`를 모든 학습 스크립트 상단에 넣는 이유가 3-(3)이다.
- Stage 3 `train_embedder.py`에서 배치 크기 8로 묶는 것도 같은 이유 — 하나씩 처리하면 CPU에서 답이 없다.
