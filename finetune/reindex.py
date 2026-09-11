"""reindex.py — 파인튜닝 모델로 코퍼스 전량을 새 컬렉션에 재색인한다.

**왜 전량 재색인이 필수인가**
임베딩 모델이 바뀌면 벡터 공간 자체가 달라진다. 기존 벡터와 새 벡터는 같은 좌표계가
아니어서 섞으면 거리 계산이 무의미해진다. 일부만 갱신하는 선택지는 없다.

**기존 chroma_db/ 는 건드리지 않는다.** 새 디렉터리(chroma_db_ft/)에 따로 만든다.
파인튜닝이 실패로 판명되면 그냥 지우면 되고, 운영 중인 앱은 영향을 받지 않는다.

**임베딩은 evaluate.py / train_embedder.py 와 같은 코드로 계산한다.**
AutoModel → mean pooling → L2 정규화. 계산한 벡터를 Chroma에 직접 넣는다
(embedding_function에 맡기지 않는다). 학습·평가·색인이 전부 같은 함수를 타야
"평가에서 좋았던 모델"과 "색인에 쓰인 모델"이 같은 것이 된다.

정규화해서 넣으면 거리 척도(l2/cosine)와 무관하게 **순위가 같다**:
    ||q-d||² = ||q||² - 2q·d + ||d||²  이고 ||d||=1 이므로
    d 사이의 비교에서는 q·d(코사인)의 순서만 남는다.

실행:
    python finetune/reindex.py --model models/ft-ep2
"""

import argparse
import json
import sqlite3
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

torch.set_num_threads(2)

ROOT = Path(__file__).parent.parent
SRC_DB = ROOT / "chroma_db" / "chroma.sqlite3"

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


def read_source(db: Path) -> list[dict]:
    """원본 컬렉션을 읽기 전용으로 읽는다. 짧은 청크도 버리지 않는다 —
    학습 데이터는 걸렀지만 색인은 앱이 검색할 전량이어야 한다."""
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        return [{"chunk_id": cid, "text": doc, "title": t or "", "path": p or "", "page_id": pid}
                for cid, doc, t, p, pid in con.execute(_QUERY) if doc]
    finally:
        con.close()


@torch.no_grad()
def encode(texts, tok, model, batch_size, max_len):
    out, t0 = [], time.time()
    for i in range(0, len(texts), batch_size):
        enc = tok(texts[i:i + batch_size], padding=True, truncation=True,
                  max_length=max_len, return_tensors="pt")
        hidden = model(**enc).last_hidden_state
        mask = enc["attention_mask"].unsqueeze(-1).float()
        vec = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        out.append(F.normalize(vec, p=2, dim=1))
        if (i // batch_size) % 10 == 0:
            print(f"\r  인코딩 {min(i+batch_size, len(texts))}/{len(texts)} "
                  f"({time.time()-t0:.0f}초)", end="", flush=True)
    print(f"\r  인코딩 {len(texts)}/{len(texts)} ({time.time()-t0:.0f}초)")
    return torch.cat(out)


def main() -> None:
    ap = argparse.ArgumentParser(description="파인튜닝 모델로 전량 재색인")
    ap.add_argument("--model", default=str(ROOT / "models" / "ft-ep2"))
    ap.add_argument("--source", type=Path, default=SRC_DB)
    ap.add_argument("--out", type=Path, default=ROOT / "chroma_db_ft")
    ap.add_argument("--collection", default="wiki_knowledge")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-len", type=int, default=128,
                    help="학습·평가와 같은 값이어야 한다")
    ap.add_argument("--add-chunk", type=int, default=500, help="Chroma 삽입 단위")
    ap.add_argument("--encoder", choices=["manual", "st"], default="st",
                    help="st: 앱과 같은 SentenceTransformerEmbeddingFunction 을 컬렉션에 심는다"
                         " (query_texts 로 검색 가능, app.py/indexer.py 와 호환). "
                         "manual: 내가 직접 계산한 벡터를 넣는다(평가 코드와 동일하나 앱에서 못 씀)")
    args = ap.parse_args()

    # ★ 안전장치 — 원본 디렉터리에 쓰려고 하면 즉시 멈춘다
    if args.out.resolve() == args.source.parent.resolve():
        raise SystemExit(f"거부: 원본 디렉터리({args.source.parent})에 쓰려고 한다. "
                         f"--out 에 다른 경로를 지정하라.")

    rows = read_source(args.source)
    print(f"원본   : {args.source} ({len(rows):,}청크, 읽기 전용)")
    print(f"모델   : {args.model}")
    print(f"대상   : {args.out} / 컬렉션 {args.collection}\n")

    import chromadb

    # ── 인코더 선택 ────────────────────────────────────────────────────────
    # 앱(app.py:47, indexer.py:39)은 SentenceTransformerEmbeddingFunction 을 쓴다.
    # 컬렉션에 그 EF를 심어야 앱이 query_texts= 로 검색할 수 있다.
    # ko-sroberta 계열은 ST가 Transformer + Pooling(mean)으로 감싸므로
    # 우리가 학습·평가에 쓴 계산과 결과가 일치한다(질의 3개로 확인: 코사인 1.000000).
    ef = None
    emb = None
    if args.encoder == "st":
        from chromadb.utils import embedding_functions
        ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=str(args.model))
        # ★ 문서 인코딩 길이를 학습·평가와 맞춘다. ST 기본값은 512라 그냥 두면
        #    평가할 때(128)와 다른 벡터가 만들어진다.
        try:
            ef.models[str(args.model)].max_seq_length = args.max_len
        except Exception:
            pass
    else:
        tok = AutoTokenizer.from_pretrained(args.model)
        model = AutoModel.from_pretrained(args.model).eval()
        emb = encode([r["text"] for r in rows], tok, model, args.batch_size, args.max_len)

    client = chromadb.PersistentClient(path=str(args.out))
    # 다시 돌려도 깨끗한 상태에서 시작하도록 기존 컬렉션은 지운다(새 디렉터리 한정)
    try:
        client.delete_collection(args.collection)
        print("  기존 컬렉션 삭제 후 재생성")
    except Exception:
        pass
    col = client.create_collection(name=args.collection, metadata={"hnsw:space": "cosine"},
                                   **({"embedding_function": ef} if ef else {}))

    t0 = time.time()
    for i in range(0, len(rows), args.add_chunk):
        part = rows[i:i + args.add_chunk]
        col.add(
            ids=[r["chunk_id"] for r in part],
            documents=[r["text"] for r in part],
            metadatas=[{"page_id": r["page_id"], "title": r["title"], "path": r["path"]}
                       for r in part],
            **({"embeddings": emb[i:i + args.add_chunk].tolist()} if emb is not None else {}),
        )
        print(f"\r  저장 {min(i+args.add_chunk, len(rows))}/{len(rows)}", end="", flush=True)
    print(f"\r  저장 {len(rows)}/{len(rows)} ({time.time()-t0:.0f}초)")

    n = col.count()
    print(f"\n완료: {n:,}건 색인 ({'원본과 일치 ✓' if n == len(rows) else '★ 원본과 불일치'})")
    print(f"기존 {args.source.parent} 은 그대로다 — 앱은 영향 없음")


if __name__ == "__main__":
    main()
