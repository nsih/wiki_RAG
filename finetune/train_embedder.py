"""train_embedder.py — 임베딩 모델을 사내 위키 도메인에 파인튜닝한다.

Trainer를 쓰지 않는다. Stage 1에서 배운 것을 그대로 조립하는 게 목적이다.

    02_autograd.py   → loss.backward()
    03_module_optim  → AdamW, 표준 학습 루프 (zero_grad → forward → backward → step)
    04_data.py       → Dataset / DataLoader
    05_save_load.py  → state_dict 저장, train() vs eval()
    06_encoder.py    → 토크나이저 → last_hidden_state → mean pooling → 정규화
    07_contrastive.py→ 유사도 행렬 (B,B), 정답은 대각선, F.cross_entropy

새로운 개념은 하나도 없다. 있다면 두 가지뿐이다:
    ① partial freeze — 하위 층을 얼려 메모리를 줄인다 (2 vCPU / 3.8 GB 서버)
    ② 같은 페이지 마스킹 — 같은 문서에서 나온 negative를 손실에서 뺀다 (PLAN 8절 4항)

실행:
    python finetune/train_embedder.py                    # 기본 (2에폭)
    python finetune/train_embedder.py --epochs 3 --freeze-below 10

★ 실행 전 확인: free -m 으로 여유 RAM 1.7 GB 이상, Streamlit 앱 종료.
"""

import argparse
import json
import math
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

torch.set_num_threads(2)

ROOT = Path(__file__).parent.parent
DATA = ROOT / "finetune" / "data"
BASE = "jhgan/ko-sroberta-multitask"


def rss_mb() -> float:
    """현재 프로세스의 실제 메모리 사용량(MB). ru_maxrss는 최댓값이라 안 줄어든다."""
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1]) / 1024
    return 0.0


# ── 1. 데이터 ──────────────────────────────────────────────────────────────

class PairDataset(Dataset):
    """(질문, 정답 청크, page_id) 를 반환한다. page_id는 손실 마스킹에 쓴다."""

    def __init__(self, path: Path):
        self.rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        return r["question"], r["text"], r["page_id"]


def collate(batch):
    q, d, p = zip(*batch)
    return list(q), list(d), torch.tensor(p)


# ── 2. 인코딩 (06_encoder.py와 동일) ────────────────────────────────────────

def mean_pool(model, tok, texts: list[str], max_len: int) -> torch.Tensor:
    enc = tok(texts, padding=True, truncation=True, max_length=max_len, return_tensors="pt")
    hidden = model(**enc).last_hidden_state                 # (B, L, 768)
    mask = enc["attention_mask"].unsqueeze(-1).float()
    vec = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
    return F.normalize(vec, p=2, dim=1)


# ── 3. 손실 (07_contrastive.py + 같은 페이지 마스킹) ────────────────────────

def contrastive_loss(q: torch.Tensor, d: torch.Tensor, page_ids: torch.Tensor,
                     scale: float, mask_same_page: bool) -> torch.Tensor:
    """in-batch negative contrastive loss.

    유사도 행렬 (B,B)에서 i번 질문의 정답은 i번 문서 — 즉 대각선이 정답이다.
    그래서 F.cross_entropy(sim, arange(B)) 한 줄이면 된다.

    마스킹이 필요한 이유: 같은 페이지에서 나온 다른 청크가 배치에 있으면
    그건 '오답'이 아닌데 오답으로 밀어낸다(false negative). 대각선만 남기고 뺀다.
    """
    sim = (q @ d.T) * scale                                  # (B, B)

    if mask_same_page:
        same = page_ids[:, None] == page_ids[None, :]
        same.fill_diagonal_(False)                           # 정답(대각선)은 살린다
        sim = sim.masked_fill(same, -1e4)

    return F.cross_entropy(sim, torch.arange(len(q)))


# ── 4. partial freeze ──────────────────────────────────────────────────────

def apply_freeze(model, freeze_below: int) -> tuple[int, int]:
    """임베딩 층과 freeze_below 미만의 인코더 층을 얼린다.

    하위 층은 일반적인 한국어 문법·형태를 담당하고 도메인 적응은 주로 상위 층에서
    일어난다. 소규모 데이터에서는 하위 층 동결이 과적합도 막아준다.
    이 서버에서는 메모리 때문에 필수다 (전체 학습 시 AdamW 상태만 880 MB).
    """
    trainable = total = 0
    for name, p in model.named_parameters():
        keep = True
        if name.startswith("embeddings.") or name.startswith("pooler."):
            keep = False                                     # pooler는 mean pooling이라 안 쓴다
        elif ".layer." in name:
            layer = int(name.split(".layer.")[1].split(".")[0])
            keep = layer >= freeze_below
        p.requires_grad = keep
        total += p.numel()
        trainable += p.numel() if keep else 0
    return trainable, total


# ── 5. 학습 중 간이 평가 ───────────────────────────────────────────────────

@torch.no_grad()
def quick_eval(model, tok, eval_rows, corpus, max_len, batch_size=16) -> dict:
    """에폭마다 Recall을 찍는다. 후보 풀이 작아 evaluate.py보다 점수가 높게 나온다.
    절대값이 아니라 **에폭 간 추세**를 보기 위한 것이다. 최종 숫자는 evaluate.py로 낸다."""
    model.eval()
    idx_of = {c["chunk_id"]: i for i, c in enumerate(corpus)}

    D = torch.cat([mean_pool(model, tok, [c["text"] for c in corpus[i:i + batch_size]], max_len)
                   for i in range(0, len(corpus), batch_size)])
    rows = [r for r in eval_rows if r["chunk_id"] in idx_of]
    Q = torch.cat([mean_pool(model, tok, [r["question"] for r in rows[i:i + batch_size]], max_len)
                   for i in range(0, len(rows), batch_size)])

    sim = Q @ D.T
    ranks = []
    for i, r in enumerate(rows):
        gold = [idx_of[g] for g in [r["chunk_id"], *r.get("also_ok", [])] if g in idx_of]
        best = max(sim[i][g].item() for g in gold)
        ranks.append(1 + int((sim[i] > best).sum().item()))

    model.train()
    n = len(ranks)
    return {"R@1": sum(r == 1 for r in ranks) / n * 100,
            "R@5": sum(r <= 5 for r in ranks) / n * 100,
            "MRR@10": sum(1 / r if r <= 10 else 0 for r in ranks) / n * 100}


# ── 6. 진입점 ──────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="임베딩 파인튜닝 (순수 PyTorch 루프)")
    ap.add_argument("--base", default=BASE)
    ap.add_argument("--train", type=Path, default=DATA / "pairs_train.jsonl")
    ap.add_argument("--eval", dest="ev", type=Path, default=DATA / "pairs_eval.jsonl")
    ap.add_argument("--eval-corpus", type=Path, default=DATA / "chunks.jsonl",
                    help="에폭별 간이 평가의 후보 풀. 최종 평가는 evaluate.py로 따로 한다")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "models")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--warmup-ratio", type=float, default=0.1)
    ap.add_argument("--scale", type=float, default=20.0)
    ap.add_argument("--freeze-below", type=int, default=8,
                    help="이 층 미만을 얼린다. 이 서버 8, 20코어 장비로 옮기면 0")
    ap.add_argument("--no-page-mask", action="store_true",
                    help="같은 페이지 마스킹을 끈다 (효과 비교용)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    print(f"메모리 시작: {rss_mb():.0f} MB")
    tok = AutoTokenizer.from_pretrained(args.base)
    model = AutoModel.from_pretrained(args.base)
    model.train()

    trainable, total = apply_freeze(model, args.freeze_below)
    print(f"모델 로드 후: {rss_mb():.0f} MB")
    print(f"학습 대상: {trainable/1e6:.1f}M / {total/1e6:.1f}M ({trainable/total*100:.1f}%) "
          f"— {args.freeze_below}층 미만 동결\n")

    ds = PairDataset(args.train)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                        num_workers=0, collate_fn=collate, drop_last=True)
    eval_rows = [json.loads(l) for l in args.ev.read_text(encoding="utf-8").splitlines() if l.strip()]
    corpus = [json.loads(l) for l in args.eval_corpus.read_text(encoding="utf-8").splitlines() if l.strip()]

    steps_per_epoch = len(loader)
    total_steps = steps_per_epoch * args.epochs
    warmup = max(1, int(total_steps * args.warmup_ratio))

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)

    def lr_lambda(step: int) -> float:
        """10% 워밍업 후 선형 감쇠. 파인튜닝 표준 스케줄이다."""
        if step < warmup:
            return step / warmup
        return max(0.0, (total_steps - step) / max(1, total_steps - warmup))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    print(f"학습 {len(ds)}쌍 / {steps_per_epoch}스텝 x {args.epochs}에폭 = {total_steps}스텝")
    print(f"batch={args.batch_size}, max_len={args.max_len}, lr={args.lr}, "
          f"warmup={warmup}스텝, scale={args.scale}, "
          f"페이지 마스킹={'끔' if args.no_page_mask else '켬'}\n")

    # 학습 전 점수 — 여기서 출발한다
    m0 = quick_eval(model, tok, eval_rows, corpus, args.max_len)
    print(f"[학습 전] R@1 {m0['R@1']:5.1f}  R@5 {m0['R@5']:5.1f}  MRR@10 {m0['MRR@10']:5.1f}"
          f"   (후보 {len(corpus)}청크 기준 간이 측정)\n")

    peak = rss_mb()
    history = [{"epoch": 0, **m0}]
    t_start = time.time()

    for epoch in range(1, args.epochs + 1):
        losses, t0 = [], time.time()

        for step, (q_txt, d_txt, page_ids) in enumerate(loader, 1):
            opt.zero_grad()
            q = mean_pool(model, tok, q_txt, args.max_len)
            d = mean_pool(model, tok, d_txt, args.max_len)
            loss = contrastive_loss(q, d, page_ids, args.scale, not args.no_page_mask)
            loss.backward()
            opt.step()
            sched.step()

            losses.append(loss.item())
            peak = max(peak, rss_mb())
            if step % 10 == 0 or step == steps_per_epoch:
                print(f"  ep{epoch} {step:>3}/{steps_per_epoch}  "
                      f"loss {sum(losses[-10:])/len(losses[-10:]):.4f}  "
                      f"lr {sched.get_last_lr()[0]:.2e}  "
                      f"{(time.time()-t0)/step:.2f}초/스텝  RSS {rss_mb():.0f}MB")

        m = quick_eval(model, tok, eval_rows, corpus, args.max_len)
        history.append({"epoch": epoch, **m})
        print(f"[에폭 {epoch}] loss {sum(losses)/len(losses):.4f}  "
              f"R@1 {m['R@1']:5.1f}  R@5 {m['R@5']:5.1f}  MRR@10 {m['MRR@10']:5.1f}  "
              f"({time.time()-t0:.0f}초)")

        # 마지막 에폭이 최선이라는 보장이 없다. 매 에폭 저장한다.
        ckpt = args.out_dir / f"ft-ep{epoch}"
        ckpt.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(ckpt)
        tok.save_pretrained(ckpt)
        print(f"          저장 → {ckpt}\n")

    print(f"총 {time.time()-t_start:.0f}초 / 최대 RSS {peak:.0f} MB\n")
    print(f"  {'에폭':>4} {'R@1':>6} {'R@5':>6} {'MRR@10':>7}")
    for h in history:
        tag = "학습전" if h["epoch"] == 0 else f"ep{h['epoch']}"
        print(f"  {tag:>4} {h['R@1']:>6.1f} {h['R@5']:>6.1f} {h['MRR@10']:>7.1f}")

    best = max(history[1:], key=lambda h: h["R@5"])
    print(f"\n간이 평가 기준 최선: ep{best['epoch']} (R@5 {best['R@5']:.1f})")
    print("→ 최종 판단은 evaluate.py 로 전량 코퍼스에서 다시 잰다:")
    print(f"   python finetune/evaluate.py --model models/ft-ep{best['epoch']} --label ft-ep{best['epoch']}")


if __name__ == "__main__":
    main()
