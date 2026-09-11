"""fusion_ab.py — BM25와 벡터를 어떻게 합칠지 여러 방식으로 재본다.

왜 필요한가:
    측정해보니 두 방식이 **서로 다른 층위**에서 잘한다.
        BM25   : 정확한 행 찍기가 강하다 (IP·호실·장비번호 같은 식별자)
        벡터   : 맞는 문서 찾기가 강하다 (ft-ep2 페이지 R@5 95% > BM25 90%)
    그런데 현재 앱의 RRF는 1/(60+rank)를 단순 합산할 뿐이라 이 역할 분담을 못 쓴다.
    강한 팔이 약한 팔에 끌려 내려간다.

앱 코드는 건드리지 않는다. 같은 색인과 같은 BM25로 후보 합치는 방식만 바꿔 잰다.

실행:
    python finetune/fusion_ab.py
    python finetune/fusion_ab.py --vec-model models/ft-ep2 --vec-db chroma_db_ft
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from tokenizer import tokenize_ko                      # 색인과 같은 토크나이저

DATA = ROOT / "finetune" / "data"


# ── 1. 후보 생성 ───────────────────────────────────────────────────────────

def rrf(rankings: list[list[str]], weights: list[float], k: int = 60) -> list[str]:
    """가중 RRF. weights=[1,1]이면 retriever.py의 현재 방식과 같다."""
    score: dict[str, float] = {}
    for lst, w in zip(rankings, weights):
        for rank, cid in enumerate(lst, 1):
            score[cid] = score.get(cid, 0.0) + w / (k + rank)
    return [c for c, _ in sorted(score.items(), key=lambda x: -x[1])]


def two_stage(vec_ids: list[str], bm_all: dict[str, float], page_of: dict[str, int],
              n_pages: int, per_page_all: dict[int, list[str]]) -> list[str]:
    """벡터로 문서(페이지)를 좁히고, 그 안에서 BM25로 행을 찍는다.

    벡터가 잘하는 일(어느 문서인가)과 BM25가 잘하는 일(그 안의 어느 행인가)을
    순서대로 시킨다. 둘을 한 번에 합산하는 RRF와 다른 점이다.
    """
    pages: list[int] = []
    for cid in vec_ids:                                  # 벡터 상위 순서대로 페이지 수집
        p = page_of.get(cid)
        if p is not None and p not in pages:
            pages.append(p)
        if len(pages) >= n_pages:
            break
    cands = [c for p in pages for c in per_page_all.get(p, [])]
    return sorted(cands, key=lambda c: -bm_all.get(c, 0.0))


# ── 2. 지표 ────────────────────────────────────────────────────────────────

def evaluate(order: list[str], gold: set[str], gold_pages: set[int],
             page_of: dict[str, int]) -> tuple[int, bool]:
    """(정답 청크 순위, 상위5에 맞는 페이지가 있는가)"""
    rank = 99
    for i, c in enumerate(order[:20], 1):
        if c in gold:
            rank = i
            break
    page_hit = any(page_of.get(c) in gold_pages for c in order[:5])
    return rank, page_hit


def main() -> None:
    ap = argparse.ArgumentParser(description="BM25/벡터 융합 방식 비교")
    ap.add_argument("--vec-db", default="chroma_db_ft")
    ap.add_argument("--vec-model", default="models/ft-ep2")
    ap.add_argument("--base-db", default="chroma_db")
    ap.add_argument("--collection", default="wiki_knowledge")
    ap.add_argument("--evals", nargs="+", default=["eval_human.jsonl", "pairs_eval.jsonl"])
    ap.add_argument("--candidates", type=int, default=20)
    ap.add_argument("--pages", type=int, default=3, help="2단계에서 벡터가 고를 문서 수")
    args = ap.parse_args()

    import chromadb
    from rank_bm25 import BM25Okapi

    src = chromadb.PersistentClient(path=args.base_db).get_collection(args.collection)
    g = src.get(include=["documents", "metadatas"])
    ids, docs, mds = g["ids"], g["documents"], g["metadatas"]
    page_of = {c: m["page_id"] for c, m in zip(ids, mds)}
    per_page: dict[int, list[str]] = {}
    for c, p in page_of.items():
        per_page.setdefault(p, []).append(c)

    bm25 = BM25Okapi([tokenize_ko(d) for d in docs])
    vec_col = chromadb.PersistentClient(path=args.vec_db).get_collection(args.collection)

    for evname in args.evals:
        ev = [json.loads(l) for l in (DATA / evname).read_text(encoding="utf-8").splitlines() if l.strip()]
        print(f"\n=== {evname} ({len(ev)}문항, 후보 {len(ids):,}청크) ===")
        print(f"벡터 팔: {args.vec_model} / {args.vec_db}\n")

        results: dict[str, list] = {}
        for e in ev:
            q = e["question"]
            gold = {e["chunk_id"], *e.get("also_ok", [])}
            gpages = {page_of[c] for c in gold if c in page_of}

            v = vec_col.query(query_texts=[q], n_results=args.candidates, include=[])["ids"][0]
            s = bm25.get_scores(tokenize_ko(q))
            bm_all = {c: float(s[i]) for i, c in enumerate(ids)}
            b = sorted(ids, key=lambda c: -bm_all[c])[:args.candidates]

            variants = {
                "① 벡터 단독": v,
                "② BM25 단독": b,
                "③ RRF 1:1 (현재 앱)": rrf([v, b], [1, 1]),
                "④ RRF 1:2 (BM25 우대)": rrf([v, b], [1, 2]),
                "⑤ RRF 1:3 (BM25 우대)": rrf([v, b], [1, 3]),
                f"⑥ 2단계(벡터→{args.pages}문서→BM25)": two_stage(v, bm_all, page_of, args.pages, per_page),
            }
            for name, order in variants.items():
                results.setdefault(name, []).append(evaluate(order, gold, gpages, page_of))

        print(f"{'방식':<26}{'R@1':>7}{'R@5':>7}{'MRR@10':>8}{'페이지R@5':>10}")
        for name, rows in results.items():
            n = len(rows)
            r1 = sum(r == 1 for r, _ in rows) / n * 100
            r5 = sum(r <= 5 for r, _ in rows) / n * 100
            mrr = sum(1 / r if r <= 10 else 0 for r, _ in rows) / n * 100
            pg = sum(p for _, p in rows) / n * 100
            print(f"{name:<26}{r1:>7.1f}{r5:>7.1f}{mrr:>8.1f}{pg:>10.1f}")


if __name__ == "__main__":
    main()
