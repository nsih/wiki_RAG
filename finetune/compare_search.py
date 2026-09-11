"""compare_search.py — 같은 질의를 기존/신규 컬렉션에 던져 상위 결과를 나란히 본다.

PLAN 9절의 마지막 게이트다. 지표가 올랐다는 것과 실제 검색 결과가 납득이 된다는 것은
다른 문제다. Recall@5가 올라도 1위에 엉뚱한 문서가 오면 사용자는 나빠졌다고 느낀다.

각 컬렉션은 **자기 모델로 만든 질의 벡터**로 검색해야 한다.
baseline 컬렉션에 파인튜닝 모델의 질의 벡터를 넣으면 좌표계가 달라 무의미하다.
메모리가 빠듯해서 모델을 하나씩 올렸다 내린다.

실행:
    python finetune/compare_search.py
    python finetune/compare_search.py --queries "10호관 무선AP" "물품 대금 언제 줘"
"""

import argparse
import gc
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

torch.set_num_threads(2)
ROOT = Path(__file__).parent.parent

DEFAULT_QUERIES = [
    "10호관 무선AP 언제 교체했지",          # 관리대장 — baseline이 가장 약했던 유형
    "7호관 아이피 대역 뭐 써",              # IP 주소 대장
    "물품 대금 며칠 안에 줘야 해",          # 규정 — baseline도 잘하던 유형
    "랜섬웨어 감염되면 어디로 신고해",      # 가이드라인
    "정보전산원 몇 호관에 있어",            # 일반 안내
]


@torch.no_grad()
def embed(texts: list[str], model_path: str, max_len: int = 128) -> torch.Tensor:
    """학습·평가·색인과 동일한 계산: mean pooling → L2 정규화."""
    tok = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path).eval()
    enc = tok(texts, padding=True, truncation=True, max_length=max_len, return_tensors="pt")
    hidden = model(**enc).last_hidden_state
    mask = enc["attention_mask"].unsqueeze(-1).float()
    vec = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
    del model, tok
    gc.collect()
    return F.normalize(vec, p=2, dim=1)


def search(db: Path, collection: str, qvecs, k: int):
    import chromadb
    col = chromadb.PersistentClient(path=str(db)).get_collection(collection)
    return col.query(query_embeddings=qvecs.tolist(), n_results=k,
                     include=["metadatas", "documents", "distances"])


def main() -> None:
    ap = argparse.ArgumentParser(description="기존/신규 컬렉션 검색 결과 비교")
    ap.add_argument("--queries", nargs="+", default=DEFAULT_QUERIES)
    ap.add_argument("--old-db", type=Path, default=ROOT / "chroma_db")
    ap.add_argument("--new-db", type=Path, default=ROOT / "chroma_db_ft")
    ap.add_argument("--old-model", default="jhgan/ko-sroberta-multitask")
    ap.add_argument("--new-model", default=str(ROOT / "models" / "ft-ep2"))
    ap.add_argument("--collection", default="wiki_knowledge")
    ap.add_argument("-k", type=int, default=3)
    args = ap.parse_args()

    qs = args.queries
    print(f"질의 {len(qs)}개 / 상위 {args.k}개 비교\n")

    old = search(args.old_db, args.collection, embed(qs, args.old_model), args.k)
    new = search(args.new_db, args.collection, embed(qs, args.new_model), args.k)

    for i, q in enumerate(qs):
        print("=" * 72)
        print(f"질의: {q}")
        print("=" * 72)
        for label, res in (("baseline", old), ("ft-ep2  ", new)):
            print(f"  [{label}]")
            for rank, (md, doc, dist) in enumerate(
                    zip(res["metadatas"][i], res["documents"][i], res["distances"][i]), 1):
                body = doc.replace("\n", " ")[:56]
                print(f"    {rank}. (dist {dist:.3f}) {md['title'][:24]:<24} | {body}")
        print()

    print("※ 거리(dist)는 두 컬렉션이 서로 다른 벡터 공간이라 절대값 비교가 무의미하다.")
    print("   순위에 뭐가 올라왔는지만 본다.")


if __name__ == "__main__":
    main()
