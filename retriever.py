# retriever.py
import logging
from collections import defaultdict

from tokenizer import tokenize_ko

logger = logging.getLogger(__name__)


# ── RRF ─────────────────────────────────────────────────────────────────────

def _rrf(rankings: list[list[str]], k: int = 60,
         top_n: int = 5) -> list[tuple[str, float]]:
    """Reciprocal Rank Fusion — 여러 랭킹 리스트를 융합해 (chunk_id, score) 쌍을 반환.

    score(d) = Σ [ 1 / (k + rank_i(d)) ]    (rank_i는 1-based)

    리스트에 없는 문서는 해당 검색기에서 점수 기여 없음.
    점수까지 함께 반환해 호출측이 디버깅/튜닝 시 실제 RRF 값을 그대로 확인할 수 있다.
    """
    scores: dict[str, float] = defaultdict(float)
    for ranking in rankings:
        for rank, chunk_id in enumerate(ranking, start=1):
            scores[chunk_id] += 1.0 / (k + rank)

    return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n]


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

    # ── 0. 빈 컬렉션 가드 ───────────────────────────────────────────────────
    # 인덱싱이 한 번도 안 된 초기 상태에서 ChromaDB query를 호출하면
    # 내부적으로 거리 계산이 비어 오류가 날 수 있어 미리 분기한다.
    try:
        coll_size = collection.count()
    except Exception as e:
        logger.warning(f"collection.count() 실패 — 0으로 간주: {e}")
        coll_size = 0

    if coll_size == 0:
        logger.debug("빈 컬렉션 — hybrid_search 즉시 종료")
        return []

    # ── 1. 벡터 검색 ────────────────────────────────────────────────────────
    vec_results = collection.query(
        query_texts=[query],
        n_results=min(candidates, coll_size),
        include=["documents", "metadatas", "distances"],
    )

    # ChromaDB는 결과가 없으면 [[]] 형태로 응답하므로 빈 리스트로 평탄화
    vec_ids: list[str] = (vec_results["ids"][0]
                          if vec_results.get("ids") and vec_results["ids"]
                          else [])
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
            # 토큰이 모두 필터링되면(예: 한 글자 질의) BM25 호출을 건너뛴다.
            if query_tokens:
                scores = bm25_index.bm25.get_scores(query_tokens)

                # 점수 내림차순 정렬 → 상위 candidates개 추출
                top_indices = sorted(
                    range(len(scores)), key=lambda i: scores[i], reverse=True
                )[:candidates]

                bm25_ids = [bm25_index.chunk_ids[i] for i in top_indices
                            if scores[i] > 0]  # 0 이하는 무관 문서 — 제외
            else:
                logger.debug("BM25 쿼리 토큰이 비어 — BM25 단계 스킵")
        except Exception as e:
            logger.warning(f"BM25 검색 실패, 벡터 단독으로 폴백: {e}")
    else:
        logger.debug("BM25 인덱스 없음 — 벡터 단독 검색")

    bm25_rank_map: dict[str, int] = {cid: r + 1 for r, cid in enumerate(bm25_ids)}

    # ── 3. RRF 융합 ─────────────────────────────────────────────────────────
    rankings: list[list[str]] = []
    if vec_ids:
        rankings.append(vec_ids)
    if bm25_ids:
        rankings.append(bm25_ids)

    if not rankings:
        # 두 검색기 모두 결과가 없는 극단적 케이스
        logger.debug("벡터/BM25 모두 결과 없음")
        return []

    fused: list[tuple[str, float]] = _rrf(rankings, k=60, top_n=top_n)
    fused_ids = [cid for cid, _ in fused]
    rrf_score_map = dict(fused)  # ← 단일 호출 결과를 그대로 사용 (실제 RRF 점수)

    # ── 4. BM25 전용 청크 본문 보완 ─────────────────────────────────────────
    # BM25에만 있고 벡터 결과엔 없는 청크는 ChromaDB에서 본문·메타데이터를 가져와야 한다.
    missing = [cid for cid in fused_ids if cid not in vec_docs]
    if missing:
        try:
            extra = collection.get(ids=missing, include=["documents", "metadatas"])
            extra_ids = extra.get("ids") or []
            extra_docs = extra.get("documents") or []
            extra_metas = extra.get("metadatas") or []
            for i, cid in enumerate(extra_ids):
                vec_docs[cid] = (extra_docs[i] if i < len(extra_docs) else "") or ""
                vec_metas[cid] = (extra_metas[i] if i < len(extra_metas) else {}) or {}
        except Exception as e:
            logger.warning(f"BM25 전용 청크 본문 조회 실패: {e}")

    # ── 5. 결과 조립 ─────────────────────────────────────────────────────────
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