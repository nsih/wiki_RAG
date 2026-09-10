"""04. Dataset과 DataLoader — 데이터를 배치로 공급하는 법

03에서 학습 루프를 배웠지만 데이터가 4개뿐이라 매 스텝 전체를 썼다.
실제로는 데이터가 수천~수백만 개다. 그럼 두 가지 문제가 생긴다:

    1. 전부 한 번에 올리면 메모리가 터진다      → 배치로 쪼갠다
    2. 항상 같은 순서로 학습하면 편향이 생긴다  → 매 에폭 섞는다

PyTorch는 이걸 두 클래스로 나눠 해결한다:
    Dataset    — "i번째 데이터 하나를 어떻게 가져오나" (내가 정의)
    DataLoader — 배치 묶기, 셔플, 병렬 로딩 (PyTorch가 제공)

이 파일은 장난감 데이터가 아니라 chroma_db의 실제 위키 청크 1,136개를 읽는다.

실행: python torch_lab/04_data.py
"""

import sqlite3
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader

torch.set_num_threads(2)
torch.manual_seed(42)

CHROMA_DB = Path(__file__).parent.parent / "chroma_db" / "chroma.sqlite3"


# ── 1. Dataset의 최소 형태 ──────────────────────────────────────────────────

print("=" * 60)
print("1. Dataset — 메서드 2개만 구현하면 된다")
print("=" * 60)


class ToyDataset(Dataset):
    """Dataset을 상속해 __len__과 __getitem__만 구현하면 끝이다."""

    def __init__(self, items):
        self.items = items

    def __len__(self):                    # 전체 개수
        return len(self.items)

    def __getitem__(self, i):             # i번째 하나를 반환
        return self.items[i]


toy = ToyDataset(["가", "나", "다", "라", "마"])
print(f"len(toy)  = {len(toy)}        ← __len__ 이 불린다")
print(f"toy[2]    = {toy[2]!r}      ← __getitem__(2) 가 불린다")

print("""
이게 Dataset의 전부다. 상속받아 2개만 구현하면 DataLoader가 알아서 쓴다.
중요한 설계 원칙: __getitem__은 '하나'만 반환한다. 배치 묶기는 DataLoader의 일이다.
""")


# ── 2. 실제 위키 청크를 읽는 Dataset ────────────────────────────────────────

print("=" * 60)
print("2. 실전 — chroma_db에서 위키 청크 1,136개 읽기")
print("=" * 60)

_QUERY = """
SELECT e.embedding_id,
       MAX(CASE WHEN m.key = 'chroma:document' THEN m.string_value END) AS document,
       MAX(CASE WHEN m.key = 'title'           THEN m.string_value END) AS title,
       MAX(CASE WHEN m.key = 'page_id'         THEN m.int_value    END) AS page_id
FROM embeddings e
JOIN embedding_metadata m ON m.id = e.id
GROUP BY e.id
ORDER BY e.id
"""


class WikiChunkDataset(Dataset):
    """chroma_db의 청크를 읽는 Dataset.

    ChromaDB 클라이언트도 임베딩 모델도 쓰지 않는다. sqlite를 읽기 전용으로 열 뿐이다.
    운영 DB를 건드릴 위험이 없고 메모리도 거의 안 쓴다.
    """

    def __init__(self, db_path: Path):
        # 읽기 전용(mode=ro)으로 열어 실수로 쓰는 것을 원천 차단
        con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        try:
            self.rows = [
                {"chunk_id": cid, "document": doc, "title": title, "page_id": pid}
                for cid, doc, title, pid in con.execute(_QUERY)
                if doc                      # 본문 없는 행은 버린다
            ]
        finally:
            con.close()

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        return self.rows[i]


ds = WikiChunkDataset(CHROMA_DB)
print(f"청크 수: {len(ds):,}개")

sample = ds[0]
print(f"\nds[0] 의 형태 (dict):")
for k, v in sample.items():
    shown = f"{v[:50]!r}..." if isinstance(v, str) and len(v) > 50 else repr(v)
    print(f"  {k:10s} = {shown}")

lengths = torch.tensor([len(r["document"]) for r in ds.rows], dtype=torch.float)
print(f"\n본문 길이: 평균 {lengths.mean():.0f}자, 최소 {int(lengths.min())}자, 최대 {int(lengths.max())}자")
print(f"페이지 수: {len({r['page_id'] for r in ds.rows})}개")


# ── 3. DataLoader — 배치로 묶기 ─────────────────────────────────────────────

print("\n" + "=" * 60)
print("3. DataLoader — 배치, 셔플, 그리고 num_workers")
print("=" * 60)

loader = DataLoader(
    ds,
    batch_size=4,
    shuffle=True,      # 매 에폭 순서를 섞는다
    num_workers=0,     # ★ 2코어 서버에서는 0이 정답. 아래 설명 참조
)

print(f"배치 크기 4 → 총 {len(loader)}개 배치 (= ceil({len(ds)}/4))\n")

batch = next(iter(loader))
print(f"배치의 타입: {type(batch).__name__}")
print(f"배치의 키  : {list(batch.keys())}")
print(f"\n각 키가 어떻게 묶였나:")
for k, v in batch.items():
    if isinstance(v, torch.Tensor):
        print(f"  {k:10s} → Tensor {tuple(v.shape)}  {v.tolist()}")
    else:
        print(f"  {k:10s} → {type(v).__name__} (길이 {len(v)})")

print("""
주목할 점: dict 하나가 아니라 '키마다 4개씩 묶인 dict'가 나왔다.
DataLoader의 기본 collate가 알아서 이렇게 바꾼다.
  - 숫자(page_id)는 텐서로 쌓아준다
  - 문자열(document, title)은 텐서로 만들 수 없으니 리스트로 둔다

num_workers 주의:
  기본값 0은 '메인 프로세스에서 로딩'이다. >0 이면 별도 프로세스를 띄운다.
  이 서버는 2코어 / 여유 RAM 1.4GB이므로 워커를 띄우면
  프로세스마다 메모리를 복제해 오히려 손해다. 반드시 0으로 둔다.
""")


# ── 4. collate_fn — 길이가 다른 것을 어떻게 묶나 ────────────────────────────

print("=" * 60)
print("4. collate_fn — 기본 collate가 실패하는 경우")
print("=" * 60)

# 길이가 다른 텐서들을 담은 Dataset
ragged = ToyDataset([torch.ones(3), torch.ones(5), torch.ones(2), torch.ones(4)])

print("길이가 3, 5, 2, 4인 텐서를 배치로 묶으려 하면:")
try:
    next(iter(DataLoader(ragged, batch_size=4)))
except RuntimeError as e:
    print(f"  RuntimeError: {str(e).splitlines()[0]}")

print("""
텐서는 직사각형이어야 한다. (4, ?) 는 존재할 수 없다.
해결책은 짧은 것을 0으로 채워 길이를 맞추는 것 — 이게 '패딩'이다.
""")


def pad_collate(items):
    """짧은 텐서를 0으로 채워 길이를 맞추고, 어디가 실제 데이터인지 mask로 알려준다."""
    max_len = max(len(t) for t in items)
    padded = torch.zeros(len(items), max_len)
    mask = torch.zeros(len(items), max_len, dtype=torch.long)
    for i, t in enumerate(items):
        padded[i, :len(t)] = t
        mask[i, :len(t)] = 1
    return {"input": padded, "mask": mask}


out = next(iter(DataLoader(ragged, batch_size=4, collate_fn=pad_collate)))
print("collate_fn=pad_collate 를 주면:")
print(f"  input {tuple(out['input'].shape)}\n{out['input']}")
print(f"  mask  {tuple(out['mask'].shape)}\n{out['mask']}")

print("""
mask가 왜 필요한가? 평균을 낼 때 패딩 0까지 세면 값이 왜곡된다.

  잘못: input.mean(dim=1)                      → 0으로 채운 칸까지 나눗셈에 포함
  올바름: (input*mask).sum(1) / mask.sum(1)    → 실제 토큰만 센다

★ 이게 06_encoder.py의 mean pooling과 정확히 같은 문제다.
  토크나이저가 문장 길이를 맞추려고 패딩을 넣으므로, 문장 벡터를 만들 때
  반드시 attention_mask로 걸러야 한다. 01_tensor.py 연습 3번의 답이 여기 있다.

실전에서는 pad_collate를 직접 안 짠다. 토크나이저가 padding=True 옵션으로 해준다.
하지만 '왜 mask가 딸려 오는지' 알아야 06에서 헤매지 않는다.
""")


# ── 5. Stage 3에서 쓸 형태 미리보기 ─────────────────────────────────────────

print("=" * 60)
print("5. Stage 3의 PairDataset — 이렇게 생길 것이다")
print("=" * 60)


class PairDataset(Dataset):
    """(질문, 정답 청크) 쌍을 공급하는 Dataset.

    Stage 2의 gen_queries.py가 만든 pairs_train.jsonl을 읽게 된다.
    지금은 데이터가 없으니 청크 본문 앞부분을 가짜 질문으로 대신한다.
    """

    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        return {
            "query": r["title"],           # 실제로는 LLM이 생성한 질문
            "positive": r["document"],     # 그 질문의 정답 청크
        }


pair_ds = PairDataset(ds.rows[:8])
pair_loader = DataLoader(pair_ds, batch_size=4, shuffle=True, num_workers=0)
pb = next(iter(pair_loader))

print(f"배치 키: {list(pb.keys())}")
print(f"query    (문자열 {len(pb['query'])}개): {pb['query']}")
print(f"positive (문자열 {len(pb['positive'])}개): 각 {[len(p) for p in pb['positive']]}자")

print("""
문자열 리스트 2개가 나온다. 여기서 토크나이저를 거치면 텐서가 되고,
그게 모델에 들어간다. Stage 3 학습 루프는 이 모양이다:

    for batch in loader:
        q = mean_pool(model(**tok(batch["query"],    padding=True)))   # 06
        d = mean_pool(model(**tok(batch["positive"], padding=True)))   # 06
        loss = contrastive_loss(q, d)                                  # 07
        loss.backward(); opt.step(); opt.zero_grad()                   # 03 ✓

03에서 배운 루프 4줄이 그대로다. 남은 건 06(텍스트→벡터)과 07(손실)뿐이다.
""")


# ── 연습 과제 ──────────────────────────────────────────────────────────────
#
# 1) 3절에서 shuffle=True를 두 번 실행하면 배치 내용이 달라진다.
#    그런데 이 파일 상단에 manual_seed(42)가 있는데도 왜 달라지지 않는가?
#    (힌트: 스크립트를 다시 실행하면? 같은 실행 안에서 두 번 돌리면?)
#
# 2) WikiChunkDataset의 __init__에서 전체를 self.rows에 올려놨다.
#    데이터가 100GB라면 이 방식이 왜 문제인가? __getitem__에서 그때그때
#    sqlite를 조회하도록 바꾸면 어떤 장단점이 있는가?
#
# 3) 4절의 pad_collate에서 mask를 쓰지 않고 input.mean(dim=1)을 계산해보라.
#    올바른 값 (모두 1.0) 과 얼마나 차이 나는가?
#
# 4) 2절 SQL에서 GROUP BY e.id 를 빼면 어떻게 되는가? 왜 필요한가?
#    (힌트: embedding_metadata는 청크 하나당 4행을 갖는다)
