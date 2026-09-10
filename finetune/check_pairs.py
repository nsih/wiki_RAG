"""check_pairs.py — 합성 질문이 학습에 쓸 만한지 숫자로 검증한다.

`--limit 5` 를 눈으로 본 결과 표에 없는 항목을 묻는 질문이 섞였다
(예: IP 열이 없는 표에 "IP 주소는?"). 눈으로 5개 본 걸로는 규모를 모른다.
그래서 318청크 전량에 대해 네 가지를 측정한다.

    1. 접지(grounding) — 질문 속 구체 토큰(호실·장비명·번호·날짜)이 정답 청크에
       실제로 있는가. 없으면 LLM이 지어낸 것이다.
    2. 라벨 모순      — 같은 질문이 서로 다른 청크에 정답으로 붙어 있는가.
       contrastive 학습에서 이건 서로 반대 방향으로 당기는 신호가 된다.
    3. 베끼기(copy)   — 질문이 청크 문장을 그대로 복사했는가.
       복사가 심하면 "검색"이 아니라 "문자열 매칭"을 학습하고 평가 점수가 부풀려진다.
    4. BM25 자기검색  — 질문을 던져 정답 청크가 상위에 오는가.
       어휘로도 못 찾는 쌍은 학습 신호로도 약하다. (임베딩 모델을 안 쓰므로 가볍다)

★ 이 스크립트는 판정을 내리지 않고 숫자만 낸다. 버릴지 말지는 사람이 정한다.
   --drop 을 주면 문제 있는 쌍을 뺀 파일을 따로 쓴다(원본은 건드리지 않는다).

실행:
    python finetune/check_pairs.py
    python finetune/check_pairs.py --drop finetune/data/pairs_clean.jsonl
"""

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from tokenizer import tokenize_ko          # 색인과 같은 토크나이저를 쓴다

DATA = ROOT / "finetune" / "data"


# ── 1. 접지 검사 ───────────────────────────────────────────────────────────
# 질문을 다른 문서와 구별해주는 건 일반 명사가 아니라 '구체 토큰'이다.
# 그게 정답 청크에 없다면 LLM이 지어낸 것이고, 그 질문으로는 정답을 찾을 수 없다.

_SPECIFIC = [
    re.compile(r"[0-9]+\s*호관"),                 # 10호관
    re.compile(r"[0-9]+\s*호실|[0-9]{3,4}\s*호"),  # 301호
    re.compile(r"[0-9]{1,3}(?:\.[0-9]{1,3}){2,3}"),# 192.168.0.1
    re.compile(r"[A-Za-z][A-Za-z0-9]*(?:-[A-Za-z0-9]+)+"),  # AP-23, Extender-A6T
    re.compile(r"[A-Za-z]{3,}"),                   # iptime
    re.compile(r"제\s*[0-9]+\s*조"),               # 제7조
]

# 날짜는 따로 다룬다. 질문은 "2020년 3월 11일", 문서는 "2020.03.11" 로 쓰는 등
# 표기가 갈려서 문자열 그대로 비교하면 실재하는 날짜를 '지어냈다'고 오판한다.
# (첫 실행에서 실제로 이 오탐이 났다 — 검증기부터 고쳤다.)
_DATE = re.compile(r"([0-9]{4})\s*[.\-/년]\s*([0-9]{1,2})\s*[.\-/월]?\s*([0-9]{1,2})?")


def _date_variants(y: str, m: str, d: str | None) -> list[str]:
    """같은 날짜의 여러 표기를 만든다. 하나라도 문서에 있으면 실재로 친다."""
    mm, dd = f"{int(m):02d}", (f"{int(d):02d}" if d else None)
    heads = [f"{y}.{mm}", f"{y}-{mm}", f"{y}/{mm}", f"{y}년{int(m)}월", f"{y}{mm}"]
    if dd is None:
        return heads
    return [f"{y}.{mm}.{dd}", f"{y}-{mm}-{dd}", f"{y}/{mm}/{dd}",
            f"{y}년{int(m)}월{int(d)}일", f"{y}{mm}{dd}"]


def specific_tokens(text: str) -> list[str]:
    """질문에서 구체 토큰을 뽑는다. 날짜는 원문 표기 그대로 반환한다."""
    out = [m.group(0) for pat in _SPECIFIC for m in pat.finditer(text)]
    out += [m.group(0) for m in _DATE.finditer(text)]
    return out


def _norm(s: str) -> str:
    return re.sub(r"\s+", "", s).lower()


def grounding(question: str, chunk: str) -> tuple[int, int]:
    """(청크에 실재하는 구체 토큰 수, 전체 구체 토큰 수)"""
    body = _norm(chunk)

    hit = tot = 0
    for m in _DATE.finditer(question):
        tot += 1
        if any(_norm(v) in body for v in _date_variants(*m.groups())):
            hit += 1

    # 날짜 부분은 위에서 이미 셌으므로 일반 토큰 검사에서는 뺀다
    masked = _DATE.sub(" ", question)
    for pat in _SPECIFIC:
        for m in pat.finditer(masked):
            tot += 1
            if _norm(m.group(0)) in body:
                hit += 1
    return hit, tot


# ── 1-b. 저품질 청크 ────────────────────────────────────────────────────────
# 목차("개요·········3")처럼 답할 내용이 없는 청크가 있다. 여기서는 LLM이 통째로 지어낸다.
#
# 주의: 처음엔 '|' 비율로 쟀더니 IP 대장의 정상 표 행이 전부 걸렸다(오탐).
#       이 코퍼스는 대부분이 마크다운 표라 '|'는 잡음이 아니라 정상 신호다.
#       빈 셀(||||||) 비율도 안 된다 — IP 대장은 비고 열이 원래 비어 있다.
#       실제로 갈리는 건 ① 목차 점선 ② 내용어(고유 토큰) 수 두 가지였다.
#       318청크로 보정한 결과 아래 기준이 목차류 9개만 정확히 집어낸다.

_DOTS = re.compile(r"[·．.…]{3,}")


def junk_score(text: str) -> tuple[float, int]:
    """(목차 점선 비율, tokenize_ko 기준 고유 내용어 수)"""
    dots = sum(len(m.group(0)) for m in _DOTS.finditer(text)) / max(len(text), 1)
    return dots, len(set(tokenize_ko(text)))


# ── 2. 베끼기 검사 ─────────────────────────────────────────────────────────

def longest_copy(question: str, chunk: str) -> int:
    """질문과 청크의 최장 공통 부분문자열 길이(공백 제거 기준)."""
    a, b = _norm(question), _norm(chunk)
    m = SequenceMatcher(None, a, b, autojunk=False).find_longest_match(0, len(a), 0, len(b))
    return m.size


# ── 3. BM25 자기검색 ───────────────────────────────────────────────────────

def bm25_selfcheck(pairs: list[dict], chunks: list[dict]) -> dict[int, int]:
    """질문을 던졌을 때 정답 청크의 BM25 순위를 구한다. {pair_index: rank}"""
    from rank_bm25 import BM25Okapi

    ids = [c["chunk_id"] for c in chunks]
    pos = {cid: i for i, cid in enumerate(ids)}
    bm25 = BM25Okapi([tokenize_ko(c["text"]) for c in chunks])

    ranks = {}
    for i, p in enumerate(pairs):
        if p["chunk_id"] not in pos:
            continue
        scores = bm25.get_scores(tokenize_ko(p["question"]))
        gold = scores[pos[p["chunk_id"]]]
        # 동점은 불리하게(뒤로) 센다
        ranks[i] = 1 + sum(1 for s in scores if s > gold)
    return ranks


# ── 4. 진입점 ──────────────────────────────────────────────────────────────

def load(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def main() -> None:
    ap = argparse.ArgumentParser(description="합성 질문 품질 검증")
    ap.add_argument("--pairs", type=Path, default=DATA / "pairs.jsonl")
    ap.add_argument("--chunks", type=Path, default=DATA / "chunks.jsonl")
    ap.add_argument("--copy-threshold", type=int, default=20,
                    help="이 길이 이상 연속 복사면 '베낌'으로 센다. 기본 20자")
    ap.add_argument("--no-bm25", action="store_true", help="BM25 검사 생략(빠름)")
    ap.add_argument("--drop", type=Path, help="문제 쌍을 제외한 파일을 여기에 쓴다")
    ap.add_argument("--show", type=int, default=5, help="유형별로 보여줄 예시 수")
    args = ap.parse_args()

    pairs = load(args.pairs)
    chunks = load(args.chunks)
    n = len(pairs)
    if not n:
        raise SystemExit("pairs가 비었다")

    by_chunk = defaultdict(list)
    for p in pairs:
        by_chunk[p["chunk_id"]].append(p)

    print("=" * 68)
    print("0. 기본 통계")
    print("=" * 68)
    covered = len(by_chunk)
    print(f"쌍 {n:,}개 / 질문이 붙은 청크 {covered:,}개 / 전체 청크 {len(chunks):,}개 "
          f"(커버리지 {covered/len(chunks)*100:.1f}%)")
    print(f"청크당 질문 수: {Counter(len(v) for v in by_chunk.values()).most_common()}")
    qlen = sorted(len(p["question"]) for p in pairs)
    print(f"질문 길이: 평균 {sum(qlen)/n:.0f}자, 중앙 {qlen[n//2]}자, "
          f"최소 {qlen[0]}자, 최대 {qlen[-1]}자")
    pc = Counter(p["page_id"] for p in pairs)
    top_pid, top_cnt = pc.most_common(1)[0]
    print(f"페이지 {len(pc)}개 / 최대 비중 page {top_pid} = {top_cnt}쌍 ({top_cnt/n*100:.1f}%)")

    # ── 접지 ──
    print("\n" + "=" * 68)
    print("1. 접지 — 질문 속 구체 토큰이 정답 청크에 실제로 있는가")
    print("=" * 68)
    no_specific, ungrounded, partial = [], [], []
    tot_hit = tot_tok = 0
    for i, p in enumerate(pairs):
        hit, tot = grounding(p["question"], p["text"])
        tot_hit += hit; tot_tok += tot
        if tot == 0:
            no_specific.append(i)
        elif hit == 0:
            ungrounded.append(i)
        elif hit < tot:
            partial.append(i)

    print(f"구체 토큰 총 {tot_tok:,}개 중 청크에 실재 {tot_hit:,}개 "
          f"({tot_hit/max(tot_tok,1)*100:.1f}%)")
    print(f"  전부 실재     : {n - len(no_specific) - len(ungrounded) - len(partial):>5}쌍 "
          f"({(n-len(no_specific)-len(ungrounded)-len(partial))/n*100:>5.1f}%)")
    print(f"  일부만 실재   : {len(partial):>5}쌍 ({len(partial)/n*100:>5.1f}%)  ← 환각 섞임")
    print(f"  하나도 없음   : {len(ungrounded):>5}쌍 ({len(ungrounded)/n*100:>5.1f}%)  ← ★ 위험")
    print(f"  구체 토큰 없음: {len(no_specific):>5}쌍 ({len(no_specific)/n*100:>5.1f}%)  "
          f"← 일반적 질문. 다른 청크와 구분 안 될 수 있다")
    for i in ungrounded[:args.show]:
        print(f"    · [{pairs[i]['chunk_id']}] {pairs[i]['question']}")
        print(f"      지어낸 토큰: {specific_tokens(pairs[i]['question'])}")

    # ── 저품질 청크 ──
    print("\n" + "=" * 68)
    print("1-b. 저품질 청크 — 목차 등 답할 내용이 없는 청크에서 나온 질문")
    print("=" * 68)
    junk_ids = set()
    for c in chunks:
        dots, uniq = junk_score(c["text"])
        if dots > 0.15 or uniq < 15:
            junk_ids.add(c["chunk_id"])
    junk_pairs = [i for i, p_ in enumerate(pairs) if p_["chunk_id"] in junk_ids]
    print(f"저품질 청크 {len(junk_ids)}개 / 전체 {len(chunks)}개 "
          f"({len(junk_ids)/len(chunks)*100:.1f}%)")
    print(f"거기서 나온 쌍 {len(junk_pairs)}개 ({len(junk_pairs)/n*100:.1f}%)")
    print("  ※ 내용이 없는 청크라 LLM이 통째로 지어낸다. export 단계에서 거르는 게 맞다.")
    for i in junk_pairs[:args.show]:
        print(f"    · [{pairs[i]['chunk_id']}] {pairs[i]['question']}")

    # ── 라벨 모순 ──
    print("\n" + "=" * 68)
    print("2. 라벨 모순 — 같은 질문이 다른 청크에 정답으로 붙었는가")
    print("=" * 68)
    qmap = defaultdict(set)
    for p in pairs:
        qmap[_norm(p["question"])].add(p["chunk_id"])
    conflict = {q: cs for q, cs in qmap.items() if len(cs) > 1}
    dup_pairs = sum(len(cs) for cs in conflict.values())
    print(f"서로 다른 청크에 같은 질문: {len(conflict)}종 / 관련 쌍 {dup_pairs}개 "
          f"({dup_pairs/n*100:.1f}%)")
    for q, cs in list(conflict.items())[:args.show]:
        print(f"    · {q[:60]}  →  {sorted(cs)[:4]}")

    # ── 베끼기 ──
    print("\n" + "=" * 68)
    print(f"3. 베끼기 — 청크 문장을 {args.copy_threshold}자 이상 그대로 복사했는가")
    print("=" * 68)
    copies = []
    lens = []
    for i, p in enumerate(pairs):
        c = longest_copy(p["question"], p["text"])
        lens.append(c)
        if c >= args.copy_threshold:
            copies.append((i, c))
    lens_sorted = sorted(lens)
    print(f"최장 연속 복사 길이: 중앙 {lens_sorted[n//2]}자, 상위10% {lens_sorted[int(n*0.9)]}자, "
          f"최대 {lens_sorted[-1]}자")
    print(f"{args.copy_threshold}자 이상 복사: {len(copies)}쌍 ({len(copies)/n*100:.1f}%)")
    for i, c in sorted(copies, key=lambda x: -x[1])[:args.show]:
        print(f"    · ({c}자) {pairs[i]['question']}")

    # ── BM25 ──
    ranks = {}
    if not args.no_bm25:
        print("\n" + "=" * 68)
        print("4. BM25 자기검색 — 질문으로 정답 청크를 어휘만으로 찾을 수 있는가")
        print("=" * 68)
        ranks = bm25_selfcheck(pairs, chunks)
        if ranks:
            r = list(ranks.values())
            top1 = sum(1 for x in r if x == 1)
            top5 = sum(1 for x in r if x <= 5)
            bad = [i for i, x in ranks.items() if x > 20]
            print(f"대상 {len(r):,}쌍 (후보 {len(chunks):,}청크)")
            print(f"  Recall@1  {top1/len(r)*100:>5.1f}%")
            print(f"  Recall@5  {top5/len(r)*100:>5.1f}%")
            print(f"  20위 밖   {len(bad)/len(r)*100:>5.1f}%  ({len(bad)}쌍) ← 어휘로 못 찾는 쌍")
            for i in bad[:args.show]:
                print(f"    · (순위 {ranks[i]}) [{pairs[i]['chunk_id']}] {pairs[i]['question']}")
            print("\n  ※ 이건 BM25 성능 측정이 아니라 '쌍이 성립하는가'의 하한선이다.")
            print("     여기서 못 찾는 쌍은 임베딩 학습 신호로도 약할 가능성이 높다.")

    # ── 제외 파일 ──
    if args.drop:
        bad_idx = set(ungrounded) | set(junk_pairs)
        bad_idx |= {i for i, p in enumerate(pairs)
                    if len(qmap[_norm(p["question"])]) > 1}
        bad_idx |= {i for i, _ in copies}
        if ranks:
            bad_idx |= {i for i, x in ranks.items() if x > 20}
        kept = [p for i, p in enumerate(pairs) if i not in bad_idx]
        with args.drop.open("w", encoding="utf-8") as f:
            for p in kept:
                f.write(json.dumps(p, ensure_ascii=False) + "\n")
        print("\n" + "=" * 68)
        print(f"제외 {len(bad_idx)}쌍 → 남은 {len(kept)}쌍 저장: {args.drop}")
        print(f"  (원본 {args.pairs.name} 은 그대로 둔다)")


if __name__ == "__main__":
    main()
