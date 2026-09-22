"""app_config_ab.py — 앱의 실제 검색 구성을 그대로 모사해 저비용 개선안을 비교한다.

지금까지의 평가는 "정답 청크가 상위 5위 안에 오는가"(R@5)로 쟀는데,
실제 앱은 다르다 (app.py:226):

    hybrid_search(..., top_n=2, candidates=20, expand_window=1)

    → RRF 상위 **2개**만 뽑고
    → 각 청크의 앞뒤 1개를 이어붙여(expand_chunks) LLM에 넘긴다

그래서 진짜 물어야 할 것은 **"LLM이 정답 내용을 받았는가"** 다.
정답 청크가 3위여도 2위 청크의 이웃이면 받은 것이고, 2위여도 이웃이 아니면 못 받는다.

이 스크립트는 그 조건을 그대로 계산해서 저비용 변경안들을 비교한다.
앱 코드는 건드리지 않는다.

실행:
    python finetune/app_config_ab.py
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from tokenizer import tokenize_ko

DATA = ROOT / "finetune" / "data"


def rrf(rankings, weights, k=60):
    score = {}
    for lst, w in zip(rankings, weights):
        for rank, cid in enumerate(lst, 1):
            score[cid] = score.get(cid, 0.0) + w / (k + rank)
    return [c for c, _ in sorted(score.items(), key=lambda x: -x[1])]


def expand(cids, all_ids_set, window=1):
    """retriever.expand_chunks 와 같은 규칙: page_{p}_chunk_{i} 의 앞뒤 window개."""
    out = set()
    for cid in cids:
        out.add(cid)
        base, i = cid.rsplit("_chunk_", 1)
        for d in range(1, window + 1):
            for j in (int(i) - d, int(i) + d):
                if j >= 0:
                    nb = f"{base}_chunk_{j}"
                    if nb in all_ids_set:
                        out.add(nb)
    return out


def main() -> None:
    import chromadb
    from rank_bm25 import BM25Okapi

    col = chromadb.PersistentClient(path=str(ROOT / "chroma_db")).get_collection("wiki_knowledge")
    g = col.get(include=["documents"])
    ids, docs = g["ids"], g["documents"]
    id_set = set(ids)
    bm25 = BM25Okapi([tokenize_ko(d) for d in docs])

    ev = [json.loads(l) for l in (DATA / "eval_human.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    ev = [e for e in ev if e["chunk_id"] in id_set]
    print(f"코퍼스 {len(ids):,}청크 / 평가 {len(ev)}문항")
    print("지표 = LLM이 정답 내용을 받았는가 (top_n개 + 각 앞뒤 1청크)\n")

    # 질의별 두 검색기의 순위 목록
    per_q = []
    for e in ev:
        q = e["question"]
        v = col.query(query_texts=[q], n_results=20, include=[])["ids"][0]
        s = bm25.get_scores(tokenize_ko(q))
        b = sorted(range(len(ids)), key=lambda i: -s[i])[:20]
        per_q.append((e, v, [ids[i] for i in b]))

    configs = {
        "현재 앱 (RRF 1:1)":      lambda v, b: rrf([v, b], [1, 1]),
        "RRF 1:3 (BM25 우대)":    lambda v, b: rrf([v, b], [1, 3]),
        "BM25 단독":              lambda v, b: b,
        "벡터 단독":              lambda v, b: v,
    }

    print(f"  {'구성':<22}" + "".join(f"{'top_n='+str(k):>10}" for k in (2, 3, 5)))
    for name, fn in configs.items():
        row = []
        for top_n in (2, 3, 5):
            hit = 0
            for e, v, b in per_q:
                order = fn(v, b)
                got = expand(order[:top_n], id_set, window=1)
                gold = {e["chunk_id"], *e.get("also_ok", [])}
                hit += bool(gold & got)
            row.append(hit / len(ev) * 100)
        mark = "  ← 현재" if name.startswith("현재") else ""
        print(f"  {name:<22}" + "".join(f"{r:>9.1f}%" for r in row) + mark)

    # 컨텍스트 분량 — _CTX_MAX_CHARS = 2500 에 걸리는지
    print(f"\n  {'구성':<22}{'top_n':>7}{'평균 컨텍스트':>14}{'2500자 초과':>12}")
    doc_of = dict(zip(ids, docs))
    for top_n in (2, 3, 5):
        lens = []
        for e, v, b in per_q:
            got = expand(rrf([v, b], [1, 1])[:top_n], id_set, window=1)
            lens.append(sum(len(doc_of[c]) for c in got))
        over = sum(1 for l in lens if l > 2500) / len(lens) * 100
        print(f"  {'현재 앱 (RRF 1:1)':<22}{top_n:>7}{sum(lens)/len(lens):>13.0f}자{over:>11.0f}%")


if __name__ == "__main__":
    main()
