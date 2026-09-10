"""prep_eval_human.py — 사람이 직접 쓰는 평가셋(eval_human.jsonl)의 뼈대 생성 + 검증

왜 필요한가:
    합성 질문은 청크 표현을 베끼는 경향이 있어 점수를 부풀린다.
    LLM이 만든 eval에서 Recall@5가 올라도 실사용 개선의 증거가 못 된다.
    → 사람이 쓴 질문 30~50개가 진짜 지표다. (PLAN.md 8절 1항)

두 가지 모드:
    --make   페이지를 골고루 덮는 청크를 뽑아 template.jsonl 을 만든다.
             각 줄의 "question" 을 사람이 채우면 그대로 평가셋이 된다.
    --check  채운 파일이 형식에 맞는지, chunk_id가 실재하는지 검증한다.

실행:
    python finetune/prep_eval_human.py --make -n 40
    # (편집기로 finetune/data/eval_human_template.jsonl 의 question 채우고 저장)
    python finetune/prep_eval_human.py --check finetune/data/eval_human.jsonl
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


# ── 템플릿 생성 ────────────────────────────────────────────────────────────

def make(chunks_path: Path, out: Path, n: int, seed: int) -> None:
    chunks = load(chunks_path)
    by_page = defaultdict(list)
    for c in chunks:
        by_page[c["page_id"]].append(c)

    rng = random.Random(seed)
    pages = sorted(by_page)
    picked: list[dict] = []

    # 페이지를 돌아가며 하나씩 뽑는다 — 큰 페이지가 독식하지 않게
    pools = {pid: rng.sample(by_page[pid], len(by_page[pid])) for pid in pages}
    while len(picked) < n and any(pools.values()):
        for pid in pages:
            if len(picked) >= n:
                break
            if pools[pid]:
                picked.append(pools[pid].pop())

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for c in picked:
            f.write(json.dumps({
                "question": "",                       # ← 여기를 사람이 채운다
                "chunk_id": c["chunk_id"],
                "page_id": c["page_id"],
                "title": c["title"],
                "hint": c["text"][:300],              # 질문 쓸 때 참고용. 평가엔 안 쓴다
            }, ensure_ascii=False) + "\n")

    print(f"템플릿 {len(picked)}줄 생성 → {out}   (페이지 {len(pages)}개 순환 배분)")
    print("\n각 줄의 \"question\" 을 채워라. 이때가 중요하다:")
    print("  · hint 문장을 베껴 쓰지 마라. 베끼면 합성 질문과 똑같아져 의미가 없다.")
    print("  · 실제로 검색창에 칠 말투로 쓴다 (예: '10호관 무선AP 언제 교체했지').")
    print("  · 답이 그 청크에 실제로 있어야 한다. 없으면 그 줄은 지운다.")
    print(f"\n다 채우면: mv {out.name} eval_human.jsonl  후 --check 로 검증")


# ── 검증 ───────────────────────────────────────────────────────────────────

def check(path: Path, chunks_path: Path) -> int:
    rows = load(path)
    valid_ids = {c["chunk_id"] for c in load(chunks_path)}
    all_ids = valid_ids  # chunks.jsonl 은 cap 적용본이라, 밖의 id는 따로 경고

    bad = 0
    seen_q = set()
    for i, r in enumerate(rows, 1):
        q = (r.get("question") or "").strip()
        cid = r.get("chunk_id")
        if not q:
            print(f"  {i:>3}줄 ✗ question 비었음 ({cid})"); bad += 1; continue
        if len(q) < 5:
            print(f"  {i:>3}줄 ✗ question 이 너무 짧다: {q!r}"); bad += 1
        if q in seen_q:
            print(f"  {i:>3}줄 ✗ 중복 질문: {q!r}"); bad += 1
        seen_q.add(q)
        if cid not in all_ids:
            print(f"  {i:>3}줄 ✗ chunks.jsonl 에 없는 chunk_id: {cid}"); bad += 1

    pages = {r.get("page_id") for r in rows}
    print(f"\n{path.name}: {len(rows)}줄 / 페이지 {len(pages)}개 / 문제 {bad}건")
    if len(rows) < 30:
        print("  ⚠ 30문항 미만이다. 지표가 흔들려 개선 판단이 어렵다.")
    if not bad:
        print("  형식 검증 통과 ✓")
    return bad


def main() -> None:
    ap = argparse.ArgumentParser(description="사람 작성 평가셋 뼈대 생성/검증")
    ap.add_argument("--make", action="store_true")
    ap.add_argument("--check", type=Path)
    ap.add_argument("--chunks", type=Path, default=DATA / "chunks.jsonl")
    ap.add_argument("--out", type=Path, default=DATA / "eval_human_template.jsonl")
    ap.add_argument("-n", type=int, default=40)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    if args.make:
        make(args.chunks, args.out, args.n, args.seed)
    elif args.check:
        raise SystemExit(1 if check(args.check, args.chunks) else 0)
    else:
        ap.error("--make 또는 --check 중 하나가 필요하다")


if __name__ == "__main__":
    main()
