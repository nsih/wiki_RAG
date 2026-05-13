# retriever.py
import logging
from collections import defaultdict

from tokenizer import tokenize_ko

logger = logging.getLogger(__name__)


# ── RRF ─────────────────────────────────────────────────────────────────────

def _rrf(rankings: list[list[str]], k: int = 60, top_n: int = 5) -> list[str]:
    """Reciprocal Rank Fusion — 여러 랭킹 리스트를 하나로 융합한다.

    score(d) = Σ [ 1 / (k + rank_i(d)) ]
    rank_i는 1-based. 리스트에 없는 문서는 해당 검색기 점수 기여 없음.
    """
    scores: dict[str, float] = defaultdict(float)
    for ranking in rankings:
        for rank, chunk_id in enumerate(ranking, start=1):
            scores[chunk_id] += 1.0 / (k + rank)

    sorted_ids = sorted(scores, key=lambda cid: scores[cid], reverse=True)
    return sorted_ids[:top_n]


# ── Hybrid Search ────────────────────────────────────────────────────────────

def hybrid_search(
    collection,
    bm25_index,           # BM25Index | None
    query: str,
    top_n: int = 5,
    candidates: int = 20,
) -> list[dict]:
    """BM25 + 벡터 검색을 RRF로 융합해 상위 top_n개 청크를 반환한다.

    bm25_index가 None이면 벡터 단독 검색으로 자동 폴백한다.
    반환 dict 키: chunk_id, document, metadata, vec_rank, bm25_rank, rrf_score
    """

    # ── 1. 벡터 검색 ────────────────────────────────────────────────────────
    vec_results = collection.query(
        query_texts=[query],
        n_results=min(candidates, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    vec_ids: list[str] = vec_results["ids"][0] if vec_results["ids"] else []
    vec_docs: dict[str, str] = {}
    vec_metas: dict[str, dict] = {}

    for i, cid in enumerate(vec_ids):
        vec_docs[cid] = (vec_results["documents"][0][i] or "")
        vec_metas[cid] = (vec_results["metadatas"][0][i] or {})

    vec_rank_map: dict[str, int] = {cid: r + 1 for r, cid in enumerate(vec_ids)}

    # ── 2. BM25 검색 (폴백 처리 포함) ───────────────────────────────────────
    bm25_ids: list[str] = []

    if bm25_index is not None and bm25_index.chunk_ids:
        try:
            query_tokens = tokenize_ko(query)
            scores = bm25_index.bm25.get_scores(query_tokens)

            # 점수 내림차순 정렬 → 상위 candidates개 추출
            top_indices = sorted(
                range(len(scores)), key=lambda i: scores[i], reverse=True
            )[:candidates]

            bm25_ids = [bm25_index.chunk_ids[i] for i in top_indices
                        if scores[i] > 0]  # 점수 0 이하는 무관 문서 — 제외
        except Exception as e:
            logger.warning(f"BM25 검색 실패, 벡터 단독으로 폴백: {e}")
    else:
        logger.debug("BM25 인덱스 없음 — 벡터 단독 검색")

    bm25_rank_map: dict[str, int] = {cid: r + 1 for r, cid in enumerate(bm25_ids)}

    # ── 3. RRF 융합 ─────────────────────────────────────────────────────────
    rankings = [vec_ids]
    if bm25_ids:
        rankings.append(bm25_ids)

    fused_ids = _rrf(rankings, k=60, top_n=top_n)

    # ── 4. 결과 조립 ─────────────────────────────────────────────────────────
    # BM25에만 있는 청크는 ChromaDB에서 본문·메타데이터를 보완한다
    missing = [cid for cid in fused_ids if cid not in vec_docs]
    if missing:
        try:
            extra = collection.get(ids=missing, include=["documents", "metadatas"])
            for i, cid in enumerate(extra["ids"]):
                vec_docs[cid] = (extra["documents"][i] or "")
                vec_metas[cid] = (extra["metadatas"][i] or {})
        except Exception as e:
            logger.warning(f"BM25 전용 청크 본문 조회 실패: {e}")

    rrf_scores = _rrf(rankings, k=60, top_n=len(fused_ids) + top_n)  # 점수 역산용
    rrf_score_map = {cid: 1.0 / (60 + r + 1) for r, cid in enumerate(rrf_scores)}

    results: list[dict] = []
    for cid in fused_ids:
        results.append({
            "chunk_id": cid,
            "document":  vec_docs.get(cid, ""),
            "metadata":  vec_metas.get(cid, {}),
            "vec_rank":  vec_rank_map.get(cid, -1),   # -1 = 해당 검색기 미포함
            "bm25_rank": bm25_rank_map.get(cid, -1),
            "rrf_score": rrf_score_map.get(cid, 0.0),
        })

    logger.debug(
        f"hybrid_search 완료 — vec={len(vec_ids)} bm25={len(bm25_ids)} "
        f"fused={len(results)}"
    )
    return results