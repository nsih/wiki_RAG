# retriever.py
import re
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


# 2단계 융합 (벡터 → 페이지 좁히기 → BM25 재정렬)

_PAGE_RE = re.compile(r'^page_(\d+)_chunk_\d+$')


def _page_id(chunk_id: str) -> int | None:
    """'page_{p}_chunk_{i}' 에서 페이지 번호를 뽑는다. 형식이 다르면 None."""
    m = _PAGE_RE.match(chunk_id)
    return int(m.group(1)) if m else None


def _two_stage(vec_ids: list[str], chunk_ids: list[str], bm25_scores,
               n_pages: int, top_n: int) -> list[tuple[str, float]]:
    """벡터로 문서(페이지)를 좁히고, 그 안에서 BM25로 청크를 고른다.

    RRF가 두 점수를 한 번에 합산하는 것과 달리, 두 검색기에 **서로 다른 일을
    순서대로** 시킨다. 측정 결과 둘이 잘하는 층위가 달랐기 때문이다.

        벡터  : 어느 문서인가        (ft-ep2 페이지 R@5 95.0 > BM25 90.0)
        BM25  : 그 문서의 어느 행인가 (청크 R@5 77.5 > 벡터 55.0)

    RRF 1:1에서는 잘하는 쪽이 못하는 쪽에 끌려 내려간다. 역할을 나누면
    eval_human R@5 기준 72.5 → 82.5 (FinetuningDocs/PROGRESS.md 9절).

    Args:
        vec_ids     : 벡터 검색 결과 (순위 순)
        chunk_ids   : BM25 인덱스의 전체 청크 ID (bm25_scores와 1:1 대응)
        bm25_scores : 전체 청크에 대한 BM25 점수 배열
        n_pages     : 벡터가 고를 페이지 수
        top_n       : 반환할 청크 수

    Returns: (chunk_id, bm25_score) 쌍. 페이지를 못 고르면 빈 리스트.
    """
    # 1단계 — 벡터 상위 순서대로 서로 다른 페이지를 n_pages개까지 모은다
    pages: list[int] = []
    for cid in vec_ids:
        p = _page_id(cid)
        if p is not None and p not in pages:
            pages.append(p)
        if len(pages) >= n_pages:
            break

    if not pages:
        # 청크 ID가 'page_..._chunk_...' 형식이 아닌 경우 — 호출측이 RRF로 폴백
        return []

    # 2단계 — 그 페이지들의 모든 청크를 BM25 점수로 재정렬
    #         점수 0인 청크도 남긴다. 후보가 top_n보다 적어지는 것을 막기 위함이며
    #         평가 스크립트(finetune/fusion_ab.py)와 동일한 동작이다.
    page_set = set(pages)
    cands = [(cid, float(bm25_scores[i]))
             for i, cid in enumerate(chunk_ids)
             if _page_id(cid) in page_set]

    cands.sort(key=lambda x: x[1], reverse=True)
    return cands[:top_n]


# 인접 청크 확장

_CHUNK_RE = re.compile(r'^(page_\d+_chunk_)(\d+)$')


def expand_chunks(collection, hits: list[dict], window: int = 1) -> list[dict]:
    """검색된 청크의 앞뒤 window개 청크를 ChromaDB에서 가져와 본문을 확장한다.
    chunk format : 'page_{page_id}_chunk_{i}'
    
    예) window=1 → 청크 #4 + #5(원본) + #6 을 이어붙여 반환
        window=2 → 청크 #3 + #4 + #5(원본) + #6 + #7

    Args:
        collection : ChromaDB 컬렉션
        hits       : hybrid_search() 반환값
        window     : 앞뒤로 확장할 청크 수 (기본 1 → 전후 각 1개)

    Returns: 단순 hit -> 인접 청크를 포함한 확장 텍스트로 교체
    """
    if not hits:
        return hits

    # 수집 대상 인접 ID 계산
    neighbor_ids: set[str] = set()

    for hit in hits:
        m = _CHUNK_RE.match(hit["chunk_id"])
        if not m:
            continue
        prefix, idx = m.group(1), int(m.group(2))
        for delta in range(-window, window + 1):
            if delta == 0:
                continue
            neighbor_ids.add(f"{prefix}{idx + delta}")

    # 이미 hits에 있는 ID는 ChromaDB 재조회 불필요
    existing_ids = {h["chunk_id"]: h["document"] for h in hits}
    to_fetch = list(neighbor_ids - set(existing_ids.keys()))

    # 인접 청크 일괄 조회
    neighbor_docs: dict[str, str] = {}
    if to_fetch:
        try:
            extra = collection.get(ids=to_fetch, include=["documents"])
            for cid, doc in zip(extra.get("ids") or [], extra.get("documents") or []):
                if doc:
                    neighbor_docs[cid] = doc
        except Exception as e:
            logger.warning(f"인접 청크 조회 실패 (원본 청크만 사용): {e}")

    # 본문 확장
    expanded = []
    for hit in hits:
        m = _CHUNK_RE.match(hit["chunk_id"])
        if not m:
            expanded.append(hit)
            continue

        prefix, idx = m.group(1), int(m.group(2))
        parts = []
        for delta in range(-window, window + 1):
            neighbor_id = f"{prefix}{idx + delta}"
            if delta == 0:
                parts.append(hit["document"])
            elif neighbor_id in neighbor_docs:
                parts.append(neighbor_docs[neighbor_id])
            elif neighbor_id in existing_ids:
                # 다른 hit가 이미 가진 청크 재사용 (추가 조회 없이)
                parts.append(existing_ids[neighbor_id])
            # 인덱스 0 미만이거나 마지막 청크 이후는 존재하지 않으므로 조용히 스킵

        merged = "\n".join(filter(None, parts))
        expanded.append({**hit, "document": merged})

    logger.debug(
        f"expand_chunks 완료 — {len(hits)}개 청크 확장 "
        f"(window={window}, 인접 조회={len(to_fetch)}개)"
    )
    return expanded


# Hybrid Search

def hybrid_search(
    collection,
    bm25_index,           # BM25Index | None
    query: str,
    top_n: int = 5,
    candidates: int = 20,
    expand_window: int = 1,
    fusion: str = "rrf",
    two_stage_pages: int = 3,
) -> list[dict]:
    """
    BM25 + 벡터 검색 -> 융합 -> 상위 top_n개 청크를 반환

    Args:
        fusion          : "rrf"       — 두 순위를 1/(60+rank)로 합산 (기존 동작)
                          "two_stage" — 벡터로 페이지 two_stage_pages개를 고른 뒤
                                        그 안에서 BM25 점수로 정렬
        two_stage_pages : fusion="two_stage"일 때 벡터가 고를 페이지 수

    반환 dict 키: chunk_id, document, metadata, vec_rank, bm25_rank,
                  rrf_score, bm25_score, fusion
    """

    # 0. 빈 컬렉션 가드
    try:
        coll_size = collection.count()
    except Exception as e:
        logger.warning(f"collection.count() 실패 — 0으로 간주: {e}")
        coll_size = 0

    if coll_size == 0:
        logger.debug("빈 컬렉션 — hybrid_search 즉시 종료")
        return []

    # 1. 벡터 검색
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

    # 2. BM25 검색 (폴백 처리 포함)
    bm25_ids: list[str] = []
    # 전체 청크 점수 — 2단계 융합이 페이지 내부를 재정렬할 때 쓴다.
    # None이면 BM25를 못 쓴 것이므로 2단계도 불가능하다.
    bm25_scores = None

    if bm25_index is not None and bm25_index.chunk_ids:
        try:
            query_tokens = tokenize_ko(query)
            # 토큰이 모두 필터링되면(예: 한 글자 질의) BM25 호출을 건너뛴다.
            if query_tokens:
                bm25_scores = bm25_index.bm25.get_scores(query_tokens)

                # 점수 내림차순 정렬 → 상위 candidates개 추출
                top_indices = sorted(
                    range(len(bm25_scores)),
                    key=lambda i: bm25_scores[i], reverse=True
                )[:candidates]

                bm25_ids = [bm25_index.chunk_ids[i] for i in top_indices
                            if bm25_scores[i] > 0]  # 0 이하는 무관 문서 — 제외
            else:
                logger.debug("BM25 쿼리 토큰이 비어 — BM25 단계 스킵")
        except Exception as e:
            logger.warning(f"BM25 검색 실패, 벡터 단독으로 폴백: {e}")
            bm25_scores = None
    else:
        logger.debug("BM25 인덱스 없음 — 벡터 단독 검색")

    bm25_rank_map: dict[str, int] = {cid: r + 1 for r, cid in enumerate(bm25_ids)}

    # bm25_scores는 bm25_index.chunk_ids와 같은 순서로 정렬돼 있어야 한다
    # (bm25_store가 tokens/chunk_ids를 항상 같이 갱신한다). 어긋나면 잘못된 청크에
    # 점수를 붙이게 되므로, 쓰지 않고 RRF로 내려간다.
    if bm25_scores is not None and len(bm25_scores) != len(bm25_index.chunk_ids):
        logger.warning(
            f"BM25 점수/ID 길이 불일치({len(bm25_scores)} vs "
            f"{len(bm25_index.chunk_ids)}) — 2단계 융합 비활성화"
        )
        bm25_scores = None

    bm25_score_map: dict[str, float] = {}
    if bm25_scores is not None:
        bm25_score_map = {cid: float(bm25_scores[i])
                          for i, cid in enumerate(bm25_index.chunk_ids)}

    # 3. 융합
    used_fusion = "rrf"
    fused_ids: list[str] = []
    rrf_score_map: dict[str, float] = {}

    # 3-a. 2단계 — 벡터로 페이지를 좁히고 그 안에서 BM25로 정렬
    #      벡터 결과와 BM25 점수가 **둘 다** 있어야 성립한다.
    #      하나라도 없으면 아래 RRF로 조용히 내려간다(= 사실상 단독 검색).
    if fusion == "two_stage" and vec_ids and bm25_scores is not None:
        staged = _two_stage(vec_ids, bm25_index.chunk_ids, bm25_scores,
                            n_pages=two_stage_pages, top_n=top_n)
        if staged:
            used_fusion = "two_stage"
            fused_ids = [cid for cid, _ in staged]
        else:
            logger.debug("2단계에서 페이지를 못 골랐다 — RRF로 폴백")
    elif fusion == "two_stage":
        logger.debug("2단계 조건 미충족(벡터 또는 BM25 결과 없음) — RRF로 폴백")

    # 3-b. RRF — 기본값이자 2단계의 폴백
    if not fused_ids:
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
        rrf_score_map = dict(fused)

    # 4. 벡터 결과 밖 청크의 본문 보완
    # BM25(또는 2단계)로만 뽑힌 청크는 ChromaDB에서 본문·메타데이터를 가져와야 한다.
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

    # 5. 결과 조립
    results: list[dict] = []
    for cid in fused_ids:
        results.append({
            "chunk_id": cid,
            "document":  vec_docs.get(cid, ""),
            "metadata":  vec_metas.get(cid, {}),
            "vec_rank":  vec_rank_map.get(cid, -1),   # -1 = 해당 검색기 미포함
            "bm25_rank": bm25_rank_map.get(cid, -1),
            "rrf_score": rrf_score_map.get(cid, 0.0),   # 2단계에서는 0.0
            "bm25_score": bm25_score_map.get(cid, 0.0),
            "fusion":    used_fusion,
        })

    # 6. 인접 청크 확장
    # expand_window=0 이면 스킵 (기존 동작 유지)
    if expand_window > 0:
        results = expand_chunks(collection, results, window=expand_window)

    logger.debug(
        f"hybrid_search 완료 — fusion={used_fusion} vec={len(vec_ids)} "
        f"bm25={len(bm25_ids)} fused={len(results)} expand_window={expand_window}"
    )
    return results