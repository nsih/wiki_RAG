"""evaluate.py — 검색 품질을 숫자로 잰다. baseline과 파인튜닝 모델을 같은 절차로 비교한다.

측정 방식은 06_encoder.py에서 손으로 만든 것 그대로다:
    토크나이저 → last_hidden_state (B,L,768) → mean pooling → L2 정규화 → 코사인 유사도
SentenceTransformer를 쓰지 않는 이유는 두 가지다.
    ① Stage 3에서 저장할 체크포인트가 AutoModel 형식이라 그대로 읽힌다
    ② baseline과 파인튜닝 모델이 완전히 같은 코드를 타야 비교가 성립한다

★ 벡터 단독으로 잰다. BM25를 섞으면(하이브리드) 임베딩이 나빠져도 BM25가 가려버려서
   파인튜닝의 효과를 볼 수 없다. 실사용 성능이 아니라 '임베딩이 나아졌는가'를 재는 것이다.

★ 후보 풀은 cap을 적용하지 않은 전량 코퍼스(1,130청크)를 쓴다.
   학습용 318청크로 재면 후보가 3분의 1로 줄어 실사용보다 쉬운 문제가 된다.

★ 페이지별 지표와 macro 평균을 함께 낸다.
   page 33(IP 대장)이 코퍼스의 76%라 micro 평균은 사실상 그 페이지 점수다. (PLAN 8절 4항)

실행:
    python finetune/evaluate.py --model jhgan/ko-sroberta-multitask --label baseline
    python finetune/evaluate.py --model models/ft-ep2 --label ft-ep2
"""

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer

torch.set_num_threads(2)          # 2코어 서버 — 기본값이면 스레드 경합으로 느려진다

ROOT = Path(__file__).parent.parent
DATA = ROOT / "finetune" / "data"
BASE = "jhgan/ko-sroberta-multitask"


# ── 1. 인코딩 (06_encoder.py와 동일) ────────────────────────────────────────

@torch.no_grad()
def encode(texts: list[str], tok, model, batch_size: int, max_len: int,
           desc: str = "") -> torch.Tensor:
    """문장 리스트 → (N, 768) 정규화된 임베딩."""
    out = []
    t0 = time.time()
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        enc = tok(batch, padding=True, truncation=True,
                  max_length=max_len, return_tensors="pt")
        hidden = model(**enc).last_hidden_state          # (B, L, 768)

        # mean pooling — 패딩 토큰은 빼고 평균낸다. 이게 핵심이다.
        mask = enc["attention_mask"].unsqueeze(-1).float()
        vec = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)

        out.append(torch.nn.functional.normalize(vec, p=2, dim=1))
        if desc and len(texts) > batch_size and (i // batch_size) % 10 == 0:
            done = min(i + batch_size, len(texts))
            print(f"\r  {desc} {done}/{len(texts)} ({time.time()-t0:.0f}초)",
                  end="", flush=True)
    if desc:
        print(f"\r  {desc} {len(texts)}/{len(texts)} ({time.time()-t0:.0f}초)")
    return torch.cat(out)


# ── 2. 지표 ────────────────────────────────────────────────────────────────

def rank_of_gold(sim_row: torch.Tensor, gold_idx: list[int]) -> int:
    """정답 청크들 중 가장 높은 순위(1부터). 동점은 불리하게 센다."""
    best = max(sim_row[i].item() for i in gold_idx)
    return 1 + int((sim_row > best).sum().item())


def metrics(ranks: list[int]) -> dict[str, float]:
    n = len(ranks)
    if not n:
        return {"n": 0, "R@1": 0.0, "R@5": 0.0, "MRR@10": 0.0}
    return {
        "n": n,
        "R@1": sum(r == 1 for r in ranks) / n * 100,
        "R@5": sum(r <= 5 for r in ranks) / n * 100,
        "MRR@10": sum(1 / r if r <= 10 else 0 for r in ranks) / n * 100,
    }


# ── 3. 진입점 ──────────────────────────────────────────────────────────────

def load(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def main() -> None:
    ap = argparse.ArgumentParser(description="임베딩 검색 품질 평가 (벡터 단독)")
    ap.add_argument("--model", default=BASE, help="모델 경로 또는 HF 이름")
    ap.add_argument("--label", default="", help="결과표에 남길 이름 (예: baseline)")
    ap.add_argument("--eval", dest="evals", nargs="+", type=Path,
                    default=[DATA / "pairs_eval.jsonl", DATA / "eval_human.jsonl"],
                    help="평가셋 jsonl (여러 개 가능)")
    ap.add_argument("--corpus", type=Path, default=DATA / "chunks_full.jsonl",
                    help="후보 풀. 기본은 cap 없는 전량 코퍼스")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-len", type=int, default=128,
                    help="학습과 같은 값을 써야 조건이 맞는다. 기본 128")
    ap.add_argument("--out", type=Path, default=DATA / "eval_results.jsonl",
                    help="결과를 여기에 한 줄 append 한다")
    args = ap.parse_args()

    corpus = load(args.corpus)
    idx_of = {c["chunk_id"]: i for i, c in enumerate(corpus)}
    page_of = {c["chunk_id"]: c["page_id"] for c in corpus}

    print(f"모델   : {args.model}")
    print(f"후보 풀: {args.corpus.name} ({len(corpus):,}청크)")
    print(f"설정   : max_len={args.max_len}, batch={args.batch_size}, 벡터 단독\n")

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model).eval()

    D = encode([c["text"] for c in corpus], tok, model,
               args.batch_size, args.max_len, desc="코퍼스 인코딩")

    results = []
    for ev_path in args.evals:
        if not ev_path.exists():
            print(f"\n[건너뜀] {ev_path} 없음")
            continue

        rows = load(ev_path)
        # 정답 청크가 후보 풀에 없으면 평가 불가 — 조용히 넘기지 않고 센다
        usable, missing = [], 0
        for r in rows:
            gold = [r["chunk_id"]] + r.get("also_ok", [])
            gold = [g for g in gold if g in idx_of]
            if gold:
                usable.append((r, [idx_of[g] for g in gold]))
            else:
                missing += 1

        print(f"\n[{ev_path.name}] {len(usable)}문항" +
              (f" (정답이 후보에 없어 제외 {missing})" if missing else ""))

        Q = encode([r["question"] for r, _ in usable], tok, model,
                   args.batch_size, args.max_len, desc="질문 인코딩")
        sim = Q @ D.T                                    # (질문 수, 청크 수)

        ranks, by_page = [], defaultdict(list)
        for i, (r, gold_idx) in enumerate(usable):
            rk = rank_of_gold(sim[i], gold_idx)
            ranks.append(rk)
            by_page[page_of.get(r["chunk_id"], r.get("page_id"))].append(rk)

        m = metrics(ranks)
        macro = {k: sum(metrics(v)[k] for v in by_page.values()) / len(by_page)
                 for k in ("R@1", "R@5", "MRR@10")}

        print(f"  micro  R@1 {m['R@1']:5.1f}  R@5 {m['R@5']:5.1f}  MRR@10 {m['MRR@10']:5.1f}")
        print(f"  macro  R@1 {macro['R@1']:5.1f}  R@5 {macro['R@5']:5.1f}  "
              f"MRR@10 {macro['MRR@10']:5.1f}   ← 페이지 평균 (쏠림 보정)")

        print(f"\n  {'page':>5} {'문항':>4} {'R@1':>6} {'R@5':>6} {'MRR@10':>7}  제목")
        title = {c["page_id"]: c["title"] for c in corpus}
        for pid in sorted(by_page):
            pm = metrics(by_page[pid])
            print(f"  {pid:>5} {pm['n']:>4} {pm['R@1']:>6.1f} {pm['R@5']:>6.1f} "
                  f"{pm['MRR@10']:>7.1f}  {title.get(pid,'')[:28]}")

        results.append({
            "label": args.label or args.model,
            "model": str(args.model),
            "eval": ev_path.name,
            "corpus": args.corpus.name,
            "n_corpus": len(corpus),
            "max_len": args.max_len,
            "micro": {k: round(v, 2) for k, v in m.items() if k != "n"},
            "macro": {k: round(v, 2) for k, v in macro.items()},
            "n": m["n"],
            "measured_at": time.strftime("%Y-%m-%d %H:%M"),
        })

    if results:
        with args.out.open("a", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"\n결과 append → {args.out}")
        print("  (baseline과 파인튜닝 결과가 같은 파일에 쌓여 나중에 표로 비교된다)")


if __name__ == "__main__":
    main()
