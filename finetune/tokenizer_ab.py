"""tokenizer_ab.py — 토크나이저 수정이 검색 품질에 얼마나 영향을 주는지 잰다.

가설: tokenize_ko 가 "7호관"의 숫자를 버려서 BM25가 호관을 구분하지 못한다.
      코퍼스의 77.3%에 '숫자+한글' 토큰이 있으므로 영향이 작지 않을 것이다.

앱 코드(tokenizer.py)는 건드리지 않는다. 수정본을 이 파일 안에 복사해 두고
두 토크나이저로 BM25 색인을 각각 만들어 같은 시험지로 비교한다.

실행:
    python finetune/tokenizer_ab.py
"""

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from tokenizer import tokenize_ko as tokenize_v1, _JOSA   # 현재 앱 (v1)

DATA = ROOT / "finetune" / "data"


# ── 수정본 (v2) ────────────────────────────────────────────────────────────

def tokenize_v2(text: str) -> list[str]:
    """v1과의 차이는 한 가지: '숫자+한글' 덩어리를 먼저 한 토큰으로 잡는다.

        v1: "7호관" → ["7", "호관"] → "7"은 1글자라 삭제 → ["호관"]
        v2: "7호관" → ["7호관"]

    조사 제거는 v1과 같은 규칙을 쓰되, "7호관에서" 같은 토큰에도 적용되도록
    한글로 끝나는 토큰 전체를 대상으로 한다.
    """
    raw = re.findall(r'[0-9]+[가-힣]+|[가-힣]+|[a-zA-Z]+|[0-9]+', text)
    out = []
    for tok in raw:
        if re.match(r'^[0-9]*[가-힣]+$', tok):
            for josa in _JOSA:
                if tok.endswith(josa) and len(tok) > len(josa):
                    tok = tok[:-len(josa)]
                    break
        if len(tok) >= 2:
            out.append(tok.lower())
    return out


def tokenize_v3(text: str) -> list[str]:
    """v2의 문제: "7호관"을 한 토큰으로 만들면 "호관"으로 묻는 질문과 안 맞는다.
    (실측: "몇 호관에 있어" 21위→39위, "몇 단계로 나눠" 1위→20위)

    v3: 숫자+한글 덩어리는 **둘 다** 낸다 — "7호관" → ["7호관", "호관"].
    정확히 물으면 "7호관"이 맞고, 모르고 물으면 "호관"이 맞는다.
    """
    raw = re.findall(r'[0-9]+[가-힣]+|[가-힣]+|[a-zA-Z]+|[0-9]+', text)
    out = []
    for tok in raw:
        if re.match(r'^[0-9]*[가-힣]+$', tok):
            for josa in _JOSA:
                if tok.endswith(josa) and len(tok) > len(josa):
                    tok = tok[:-len(josa)]
                    break
        if len(tok) >= 2:
            out.append(tok.lower())
        m = re.match(r'^([0-9]+)([가-힣]+)$', tok)
        if m and len(m.group(2)) >= 2:
            out.append(m.group(2))            # 한글 부분도 따로 낸다
    return out


# ── 검색 방식 (fusion_ab.py와 동일) ────────────────────────────────────────

def rrf(rankings, weights, k=60):
    score = {}
    for lst, w in zip(rankings, weights):
        for rank, cid in enumerate(lst, 1):
            score[cid] = score.get(cid, 0.0) + w / (k + rank)
    return [c for c, _ in sorted(score.items(), key=lambda x: -x[1])]


def two_stage(vec_ids, bm_all, page_of, per_page, n_pages=3):
    pages = []
    for cid in vec_ids:
        p = page_of.get(cid)
        if p is not None and p not in pages:
            pages.append(p)
        if len(pages) >= n_pages:
            break
    cands = [c for p in pages for c in per_page.get(p, [])]
    return sorted(cands, key=lambda c: -bm_all.get(c, 0.0))


def metrics(ranks):
    n = len(ranks)
    return (sum(r == 1 for r in ranks) / n * 100,
            sum(r <= 5 for r in ranks) / n * 100,
            sum(1 / r if r <= 10 else 0 for r in ranks) / n * 100)


def main() -> None:
    import chromadb
    from rank_bm25 import BM25Okapi

    src = chromadb.PersistentClient(path=str(ROOT / "chroma_db")).get_collection("wiki_knowledge")
    g = src.get(include=["documents", "metadatas"])
    ids, docs, mds = g["ids"], g["documents"], g["metadatas"]
    doc_of = dict(zip(ids, docs))
    page_of = {c: m["page_id"] for c, m in zip(ids, mds)}
    per_page = {}
    for c, p in page_of.items():
        per_page.setdefault(p, []).append(c)

    print(f"코퍼스 {len(ids):,}청크. BM25 색인 2개 구축 중...")
    bm = {"v1(현재)": BM25Okapi([tokenize_v1(d) for d in docs]),
          "v2(합침)": BM25Okapi([tokenize_v2(d) for d in docs]),
          "v3(둘다)": BM25Okapi([tokenize_v3(d) for d in docs])}
    tok = {"v1(현재)": tokenize_v1, "v2(합침)": tokenize_v2, "v3(둘다)": tokenize_v3}

    col_base = chromadb.PersistentClient(path=str(ROOT / "chroma_db")).get_collection("wiki_knowledge")
    col_ft = chromadb.PersistentClient(path=str(ROOT / "chroma_db_ft")).get_collection("wiki_knowledge")

    # ── 1. 토큰화 예시 ──
    print("\n" + "=" * 70)
    print("1. 토큰화 차이")
    print("=" * 70)
    for s in ["7호관 아이피 대역 뭐 써", "제7조에 따라 3층 301호실에서", "2026년 3월 1일자 사무분장표"]:
        print(f"  {s}")
        print(f"    v1: {tokenize_v1(s)}")
        print(f"    v2: {tokenize_v2(s)}")
        print(f"    v3: {tokenize_v3(s)}")

    # ── 2. 호관 변별 ──
    print("\n" + "=" * 70)
    print("2. 호관 변별 — 질의한 호관이 BM25 상위 10개 중 몇 개나 나오나")
    print("=" * 70)
    blds = ["1호관", "6호관", "7호관", "9호관", "별관"]
    print(f"  {'질의':<8}{'v1(현재)':>10}{'v2(합침)':>10}{'v3(둘다)':>10}")
    tot = {n: 0 for n in bm}
    for b in blds:
        q = f"{b} 아이피 대역 뭐 써"
        row = []
        for name in bm:
            s = bm[name].get_scores(tok[name](q))
            top = [ids[i] for i in sorted(range(len(ids)), key=lambda i: -s[i])[:10]]
            n = sum(1 for c in top if f"|{b}|" in doc_of[c])
            tot[name] += n
            row.append(n)
        print(f"  {b:<8}" + "".join(f"{r:>8}/10" for r in row))
    print(f"  {'합계':<8}" + "".join(f"{tot[n]:>8}/50" for n in bm))

    # ── 3. 시험지 평가 ──
    for evname in ["eval_human.jsonl", "pairs_eval.jsonl"]:
        ev = [json.loads(l) for l in (DATA / evname).read_text(encoding="utf-8").splitlines() if l.strip()]
        print("\n" + "=" * 70)
        print(f"3. {evname} ({len(ev)}문항) — 토크나이저별 검색 품질")
        print("=" * 70)

        # 벡터 순위는 토크나이저와 무관하므로 한 번만
        vb = [col_base.query(query_texts=[e["question"]], n_results=20, include=[])["ids"][0] for e in ev]
        vf = [col_ft.query(query_texts=[e["question"]], n_results=20, include=[])["ids"][0] for e in ev]

        results = {}
        for name in bm:
            for i, e in enumerate(ev):
                q = e["question"]
                gold = {e["chunk_id"], *e.get("also_ok", [])}
                s = bm[name].get_scores(tok[name](q))
                bm_all = {c: float(s[j]) for j, c in enumerate(ids)}
                b = sorted(ids, key=lambda c: -bm_all[c])[:20]
                variants = {
                    "BM25 단독": b,
                    "현재앱 (base+RRF)": rrf([vb[i], b], [1, 1]),
                    "ft+2단계": two_stage(vf[i], bm_all, page_of, per_page),
                }
                for vname, order in variants.items():
                    hit = [j + 1 for j, c in enumerate(order[:20]) if c in gold]
                    results.setdefault((vname, name), []).append(min(hit) if hit else 99)

        print(f"  {'방식':<20}{'토크나이저':<10}{'R@1':>7}{'R@5':>7}{'MRR@10':>8}")
        for vname in ["BM25 단독", "현재앱 (base+RRF)", "ft+2단계"]:
            for name in bm:
                r1, r5, mrr = metrics(results[(vname, name)])
                print(f"  {vname:<20}{name:<10}{r1:>7.1f}{r5:>7.1f}{mrr:>8.1f}")
            print()


if __name__ == "__main__":
    main()
