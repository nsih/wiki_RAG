"""split_pairs.py — pairs.jsonl을 train/eval로 나눈다.

★ 반드시 chunk_id 기준으로 나눈다.
   한 청크에서 질문 2개가 나오는데 이걸 질문 단위로 섞어 나누면,
   같은 청크의 질문 하나는 train에 하나는 eval에 간다.
   그러면 모델은 eval 청크를 학습 때 이미 본 것이 되어 평가가 새어 오염된다.

페이지 단위로 층화(stratify)한다. 그냥 섞으면 청크가 몇 개 안 되는 페이지가
eval에 하나도 안 들어가 그 페이지 성능을 못 재게 된다.

실행:
    python finetune/split_pairs.py                 # 8:2
    python finetune/split_pairs.py --eval-ratio 0.1
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).parent.parent
DATA = ROOT / "finetune" / "data"


def load(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def dump(rows: list[dict], p: Path) -> None:
    with p.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="pairs.jsonl을 chunk_id 기준으로 train/eval 분할")
    ap.add_argument("--in", dest="inp", type=Path, default=DATA / "pairs.jsonl")
    ap.add_argument("--train", type=Path, default=DATA / "pairs_train.jsonl")
    ap.add_argument("--eval", dest="ev", type=Path, default=DATA / "pairs_eval.jsonl")
    ap.add_argument("--eval-ratio", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    pairs = load(args.inp)

    # chunk_id → page_id (청크 단위로 접기)
    page_of: dict[str, int] = {}
    for r in pairs:
        page_of[r["chunk_id"]] = r["page_id"]

    by_page = defaultdict(list)
    for cid, pid in page_of.items():
        by_page[pid].append(cid)

    rng = random.Random(args.seed)
    eval_ids: set[str] = set()
    for pid in sorted(by_page):
        ids = sorted(by_page[pid])
        rng.shuffle(ids)
        # 페이지마다 최소 1개는 eval로 (단, 청크가 1개뿐이면 train에 남긴다)
        k = max(1, round(len(ids) * args.eval_ratio)) if len(ids) > 1 else 0
        eval_ids.update(ids[:k])

    train = [r for r in pairs if r["chunk_id"] not in eval_ids]
    ev = [r for r in pairs if r["chunk_id"] in eval_ids]

    # 누수 검증 — 두 쪽에 같은 chunk_id가 있으면 안 된다
    overlap = {r["chunk_id"] for r in train} & {r["chunk_id"] for r in ev}
    assert not overlap, f"chunk_id 누수 {len(overlap)}건: {list(overlap)[:5]}"

    dump(train, args.train)
    dump(ev, args.ev)

    print(f"입력 {len(pairs):,}쌍 / 청크 {len(page_of):,}개 / 페이지 {len(by_page)}개")
    print(f"  train {len(train):,}쌍 ({len(page_of)-len(eval_ids):,}청크) → {args.train}")
    print(f"  eval  {len(ev):,}쌍 ({len(eval_ids):,}청크) → {args.ev}")
    print("  chunk_id 누수 없음 ✓")

    print(f"\n  {'page_id':>8} {'train':>6} {'eval':>5}  제목")
    title = {r["page_id"]: r["title"] for r in pairs}
    tc = defaultdict(int); ec = defaultdict(int)
    for r in train: tc[r["page_id"]] += 1
    for r in ev:    ec[r["page_id"]] += 1
    for pid in sorted(by_page):
        print(f"  {pid:>8} {tc[pid]:>6} {ec[pid]:>5}  {title[pid]}")


if __name__ == "__main__":
    main()
