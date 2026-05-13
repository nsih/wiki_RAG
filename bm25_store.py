# bm25_store.py
import os
import pickle
import logging
from dataclasses import dataclass, field

from rank_bm25 import BM25Okapi

from tokenizer import tokenize_ko

logger = logging.getLogger(__name__)


@dataclass
class BM25Index:
    """BM25 검색에 필요한 상태를 하나로 묶는 컨테이너.

    tokens는 재빌드 시 재토큰화를 생략하기 위한 캐시다.
    bm25는 불변 객체이므로 패치 시 전체 재생성한다.
    """
    bm25: BM25Okapi
    chunk_ids: list[str]          # ChromaDB ID와 1:1 대응 (순서 보장)
    tokens: list[list[str]]       # chunk_ids[i]에 대응하는 토큰 캐시


# ── 빌드 ────────────────────────────────────────────────────────────────────

def build_from_chroma(collection) -> "BM25Index":
    """ChromaDB 컬렉션 전체를 스캔해 BM25Index를 생성한다.

    ChromaDB get()은 limit 미지정 시 전체 반환한다.
    대용량에서는 메모리 사용량에 유의할 것.
    """
    logger.info("BM25 인덱스 빌드 시작 — ChromaDB 전체 스캔")
    result = collection.get(include=["documents"])

    ids: list[str] = result.get("ids") or []
    docs: list[str] = result.get("documents") or []

    if not ids:
        logger.warning("ChromaDB에 문서가 없습니다. 빈 BM25Index를 반환합니다.")
        empty_bm25 = BM25Okapi([[]])
        return BM25Index(bm25=empty_bm25, chunk_ids=[], tokens=[])

    tokens: list[list[str]] = [tokenize_ko(doc) for doc in docs]
    bm25 = BM25Okapi(tokens)

    logger.info(f"BM25 인덱스 빌드 완료 — {len(ids)}개 청크")
    return BM25Index(bm25=bm25, chunk_ids=ids, tokens=tokens)


# ── 저장 / 로드 ──────────────────────────────────────────────────────────────

def save(index: BM25Index, path: str) -> None:
    """os.replace를 이용한 원자적 저장 — 부분 기록 중 앱 읽기 방지."""
    tmp_path = path + ".tmp"
    with open(tmp_path, "wb") as f:
        pickle.dump(index, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp_path, path)
    logger.info(f"BM25 인덱스 저장 완료: {path}")


def load(path: str) -> BM25Index | None:
    """파일이 없거나 손상된 경우 None 반환 — 호출측이 폴백 처리한다."""
    if not os.path.exists(path):
        logger.warning(f"BM25 인덱스 파일 없음: {path} → 벡터 단독 검색으로 폴백")
        return None
    try:
        with open(path, "rb") as f:
            index = pickle.load(f)
        logger.info(f"BM25 인덱스 로드 완료 — {len(index.chunk_ids)}개 청크")
        return index
    except Exception as e:
        logger.error(f"BM25 인덱스 로드 실패: {e} → 폴백")
        return None


# ── 패치 (in-place, 디스크 미반영) ──────────────────────────────────────────

def patch_add(index: BM25Index, new_chunk_ids: list[str], new_documents: list[str]) -> None:
    """PDF 업로드 직후 메모리상 BM25를 즉시 반영한다.

    BM25Okapi는 불변이므로 기존 토큰 캐시에 추가 후 전체 재생성한다.
    디스크에는 기록하지 않으며, 다음 indexer 배치에서 정식 반영된다.
    """
    if not new_chunk_ids:
        return
    try:
        new_tokens = [tokenize_ko(doc) for doc in new_documents]
        index.chunk_ids.extend(new_chunk_ids)
        index.tokens.extend(new_tokens)
        index.bm25 = BM25Okapi(index.tokens)
        logger.info(f"BM25 patch_add 완료 — {len(new_chunk_ids)}개 청크 추가")
    except Exception as e:
        logger.warning(f"BM25 patch_add 실패 (다음 배치에서 복구됨): {e}")
        raise


def patch_remove(index: BM25Index, chunk_ids: list[str]) -> None:
    """업데이트 전 기존 청크를 메모리에서 제거한다.

    chunk_ids 집합에 없는 항목만 남긴 뒤 BM25를 재생성한다.
    """
    if not chunk_ids:
        return
    try:
        remove_set = set(chunk_ids)
        pairs = [(cid, tok) for cid, tok in zip(index.chunk_ids, index.tokens)
                 if cid not in remove_set]

        if not pairs:
            index.chunk_ids.clear()
            index.tokens.clear()
            index.bm25 = BM25Okapi([[]])
            return

        kept_ids, kept_tokens = zip(*pairs)
        index.chunk_ids[:] = list(kept_ids)
        index.tokens[:] = list(kept_tokens)
        index.bm25 = BM25Okapi(index.tokens)
        logger.info(f"BM25 patch_remove 완료 — {len(chunk_ids)}개 청크 제거")
    except Exception as e:
        logger.warning(f"BM25 patch_remove 실패 (다음 배치에서 복구됨): {e}")
        raise