"""export_corpus.py — chroma_db의 청크를 학습용 jsonl로 추출한다.

이 스크립트가 하지 않는 것:
    - Wiki 서버 접속 (불통이어도 상관없다)
    - Chroma 클라이언트 사용 (sqlite를 직접 읽는다)
    - 임베딩 모델 로드 (텍스트만 다룬다)

즉 메모리를 거의 쓰지 않고 운영 DB를 건드릴 위험도 없다(mode=ro).

★ --max-per-page 가 이 스크립트의 핵심이다.
   코퍼스의 76.3%가 page 33(IP 주소 대장) 한 페이지다. 그대로 질문을 생성하면
   질문 예산의 76%를 IP 대장에 쓰게 되고, 학습이 그쪽으로 쏠린다.
   페이지당 상한을 두면 cap 60 기준 최대 비중이 18.9%로 내려간다.
   (근거: FinetuningDocs/PLAN.md 8절 4항)

실행:
    python finetune/export_corpus.py                      # 기본값 (cap 60)
    python finetune/export_corpus.py --max-per-page 0     # 상한 없음 (전량)
"""

import argparse
import json
import random
import sqlite3
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).parent.parent
DEFAULT_DB = ROOT / "chroma_db" / "chroma.sqlite3"
DEFAULT_OUT = ROOT / "finetune" / "data" / "chunks.jsonl"

# embedding_metadata는 (id, key, value) 세로 형태라 key별로 가로로 편다.
_QUERY = """
SELECT e.embedding_id,
       MAX(CASE WHEN m.key = 'chroma:document' THEN m.string_value END) AS document,
       MAX(CASE WHEN m.key = 'title'           THEN m.string_value END) AS title,
       MAX(CASE WHEN m.key = 'path'            THEN m.string_value END) AS path,
       MAX(CASE WHEN m.key = 'page_id'         THEN m.int_value    END) AS page_id
FROM embeddings e
JOIN embedding_metadata m ON m.id = e.id
GROUP BY e.id
ORDER BY e.id
"""


# ── 1. 읽기 ────────────────────────────────────────────────────────────────

def read_chunks(db_path: Path) -> list[dict]:
    """sqlite를 읽기 전용으로 열어 청크를 전부 읽는다."""
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        rows = []
        for cid, doc, title, path, pid in con.execute(_QUERY):
            if not doc:                     # 본문 없는 행은 버린다
                continue
            rows.append({
                "chunk_id": cid,
                "page_id": pid,
                "title": title or "",
                "path": path or "",
                "text": doc,
            })
        return rows
    finally:
        con.close()


# ── 2. 페이지별 상한 ───────────────────────────────────────────────────────

def cap_per_page(rows: list[dict], cap: int, seed: int) -> list[dict]:
    """페이지당 최대 cap개만 남긴다. cap <= 0 이면 그대로 반환.

    어떤 cap개를 고르느냐가 중요하다. 앞에서부터 자르면 문서 앞부분만 남아
    (IP 대장이라면 특정 대역만) 편향이 또 생긴다. 그래서 페이지 안에서
    시드 고정 무작위 표본을 뽑되, 원래 순서를 유지해 되돌려 놓는다.
    """
    if cap <= 0:
        return rows

    by_page = defaultdict(list)
    for i, r in enumerate(rows):
        by_page[r["page_id"]].append(i)

    rng = random.Random(seed)
    keep: set[int] = set()
    for _, idxs in sorted(by_page.items()):
        keep.update(idxs if len(idxs) <= cap else rng.sample(idxs, cap))

    return [r for i, r in enumerate(rows) if i in keep]


# ── 3. 통계 출력 ───────────────────────────────────────────────────────────

def report(rows: list[dict], label: str) -> None:
    n = len(rows)
    by_page = defaultdict(list)
    for r in rows:
        by_page[r["page_id"]].append(r)

    lens = sorted(len(r["text"]) for r in rows)
    print(f"\n[{label}] 청크 {n:,}개 / 페이지 {len(by_page)}개")
    if not n:
        return
    print(f"  본문 길이: 평균 {sum(lens)/n:.0f}자, 중앙 {lens[n//2]}자, "
          f"최소 {lens[0]}자, 최대 {lens[-1]}자")

    top = sorted(by_page.items(), key=lambda kv: -len(kv[1]))[:5]
    print(f"  {'page_id':>8} {'청크':>6} {'비중':>7}  제목")
    for pid, items in top:
        print(f"  {pid:>8} {len(items):>6} {len(items)/n*100:>6.1f}%  {items[0]['title']}")
    if len(by_page) > 5:
        rest = n - sum(len(i) for _, i in top)
        print(f"  {'(나머지)':>8} {rest:>6} {rest/n*100:>6.1f}%")
    print(f"  → 최대 페이지 비중 {len(top[0][1])/n*100:.1f}%")


# ── 4. 진입점 ──────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="chroma_db 청크를 chunks.jsonl로 추출")
    ap.add_argument("--db", type=Path, default=DEFAULT_DB)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--max-per-page", type=int, default=60,
                    help="페이지당 청크 상한 (0이면 무제한). 기본 60")
    ap.add_argument("--min-chars", type=int, default=100,
                    help="이 길이 미만 청크는 버린다. 기본 100")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rows = read_chunks(args.db)
    report(rows, "원본")

    short = [r for r in rows if len(r["text"]) < args.min_chars]
    rows = [r for r in rows if len(r["text"]) >= args.min_chars]
    print(f"\n{args.min_chars}자 미만 {len(short)}개 제외 → {len(rows):,}개")

    rows = cap_per_page(rows, args.max_per_page, args.seed)
    report(rows, f"최종 (cap={args.max_per_page or '무제한'})")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\n저장: {args.out}  ({len(rows):,}줄)")


if __name__ == "__main__":
    main()
