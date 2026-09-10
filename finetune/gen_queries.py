"""gen_queries.py — LLM으로 (질문, 정답 청크) 쌍을 합성한다.

라벨 데이터가 없으므로 doc2query 방식으로 만든다:
    청크를 LLM에 보여주고 "이 청크로 답할 수 있는 질문"을 쓰게 한다.

호출 규약은 app.py:103-140 을 그대로 따른다 (/no_think 접두사, 단일 user 메시지).
엔드포인트는 .streamlit/secrets.toml 에서 읽는다 — 하드코딩하지 않는다.

★ 중단·재개가 전제다. 91.44도 CPU 추론이라 318청크에 수십 분이 걸린다.
   한 청크 끝날 때마다 한 줄씩 append 하고, 재실행하면 이미 한 chunk_id는 건너뛴다.
   즉 Ctrl-C 로 끊고 다시 돌려도 손실이 없다.

실행:
    python finetune/gen_queries.py --limit 5     # 먼저 5개만 눈으로 확인 (필수 게이트)
    python finetune/gen_queries.py               # 전량 (--resume 이 기본)
"""

import argparse
import json
import re
import sys
import time
import tomllib
from pathlib import Path

import requests

ROOT = Path(__file__).parent.parent
DEFAULT_IN = ROOT / "finetune" / "data" / "chunks.jsonl"
DEFAULT_OUT = ROOT / "finetune" / "data" / "pairs.jsonl"
SECRETS = ROOT / ".streamlit" / "secrets.toml"


# ── 1. 설정 ────────────────────────────────────────────────────────────────

def load_worker() -> tuple[str, str]:
    """secrets.toml에서 (엔드포인트, 모델명)을 읽는다."""
    with SECRETS.open("rb") as f:
        s = tomllib.load(f)
    ip = s["AI_WORKER_IP"]
    port = s["AI_WORKER_PORT"]
    return f"http://{ip}:{port}/v1/chat/completions", s.get("AI_MODEL_NAME", "")


PROMPT = """/no_think

아래는 대학교 정보전산원 내부 위키의 문서 일부다.
이 내용으로 **답할 수 있는** 한국어 질문을 정확히 2개 만들어라.

규칙:
- 실제 교직원이 검색창에 칠 법한 자연스러운 질문으로 쓴다.
- 문서에 나온 고유한 단어(장비명, 호관/호실, 부서명, 규정 이름, IP 대역, 날짜)를
  최소 하나는 질문에 넣는다. 그래야 다른 문서와 구분되는 질문이 된다.
- "이 문서", "위 표", "해당 내용" 같은 지시어는 쓰지 않는다. 질문만 읽고도 뜻이 통해야 한다.
- 답이 문서에 없는 질문은 만들지 않는다.
- 질문 2개를 각각 한 줄씩, 번호나 기호 없이 그대로 출력한다. 다른 말은 쓰지 않는다.

[문서 제목]
{title}

[문서 내용]
{text}
"""


# ── 2. 응답 파싱 ───────────────────────────────────────────────────────────

_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)
_LEAD = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s*")


def parse_questions(raw: str) -> list[str]:
    """모델 출력에서 질문 줄만 추려낸다. 형식이 깨지면 빈 리스트."""
    raw = _THINK.sub("", raw or "")
    out = []
    for line in raw.splitlines():
        q = _LEAD.sub("", line.strip()).strip().strip('"').strip("*").strip()
        if len(q) < 6 or len(q) > 200:      # 너무 짧거나 긴 줄은 잡소리
            continue
        # 질문 판별: 처음엔 ?/까/요 로 끝나는 줄만 받았는데 너무 좁았다.
        # "무엇인가", "어떤 기기인지" 같은 정상 질문이 통째로 버려져
        # 청크 2개가 계속 실패했다(page_33_chunk_623 등).
        # 한국어 평서문은 대개 '다'로 끝나므로, 그 반대로 판별한다.
        tail = q.rstrip(" .。!·")          # 마침표를 떼고 어미를 본다
        if not q.endswith("?") and (tail.endswith("다") or tail.endswith(":")
                                    or tail.endswith("음") or tail.endswith("함")):
            continue                        # 평서문·머리말 줄 제외
        if q not in out:
            out.append(q)
    return out[:2]


# ── 3. LLM 호출 ────────────────────────────────────────────────────────────

def ask(endpoint: str, model: str, prompt: str, temperature: float,
        timeout: tuple[int, int], retries: int) -> str:
    payload = {
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "temperature": temperature,
        "max_tokens": 512,
    }
    if model:
        payload["model"] = model

    for attempt in range(retries + 1):
        try:
            res = requests.post(endpoint, json=payload,
                                headers={"Content-Type": "application/json"},
                                timeout=timeout)
            if res.status_code == 200:
                return res.json()["choices"][0]["message"]["content"]
            print(f"    HTTP {res.status_code}", file=sys.stderr)
        except Exception as e:
            print(f"    호출 실패: {e}", file=sys.stderr)
        if attempt < retries:
            time.sleep(2 * (attempt + 1))
    return ""


# ── 4. 진입점 ──────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="청크에서 질문을 합성해 pairs.jsonl 생성")
    ap.add_argument("--in", dest="inp", type=Path, default=DEFAULT_IN)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--limit", type=int, default=0, help="이번 실행에서 처리할 청크 수 (0=전부)")
    ap.add_argument("--no-resume", action="store_true",
                    help="기존 출력을 무시하고 처음부터 (덮어쓰지 않고 이어붙이므로 주의)")
    ap.add_argument("--temperature", type=float, default=0.3)
    ap.add_argument("--max-chars", type=int, default=1200, help="청크 본문을 이 길이로 자른다")
    ap.add_argument("--retries", type=int, default=2)
    args = ap.parse_args()

    endpoint, model = load_worker()
    chunks = [json.loads(l) for l in args.inp.read_text(encoding="utf-8").splitlines() if l.strip()]

    # 재개: 이미 질문을 만든 chunk_id는 건너뛴다
    done: set[str] = set()
    if args.out.exists() and not args.no_resume:
        for l in args.out.read_text(encoding="utf-8").splitlines():
            if l.strip():
                done.add(json.loads(l)["chunk_id"])

    todo = [c for c in chunks if c["chunk_id"] not in done]
    if args.limit:
        todo = todo[:args.limit]

    print(f"엔드포인트 {endpoint} / 모델 {model or '(서버 기본)'}")
    print(f"전체 {len(chunks)}청크, 완료 {len(done)}, 이번 실행 {len(todo)}\n")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    ok = skipped = 0
    t0 = time.time()

    # 한 청크 끝날 때마다 flush — 언제 끊겨도 여기까지는 남는다
    with args.out.open("a", encoding="utf-8") as f:
        for i, c in enumerate(todo, 1):
            prompt = PROMPT.format(title=c["title"], text=c["text"][:args.max_chars])
            raw = ask(endpoint, model, prompt, args.temperature, (10, 180), args.retries)
            qs = parse_questions(raw)

            if not qs:
                skipped += 1
                print(f"[{i}/{len(todo)}] {c['chunk_id']}  ✗ 파싱 실패(스킵)")
                continue

            for q in qs:
                f.write(json.dumps({
                    "chunk_id": c["chunk_id"],
                    "page_id": c["page_id"],
                    "title": c["title"],
                    "question": q,
                    "text": c["text"],
                }, ensure_ascii=False) + "\n")
            f.flush()
            ok += 1

            el = time.time() - t0
            eta = el / i * (len(todo) - i)
            print(f"[{i}/{len(todo)}] {c['chunk_id']}  질문 {len(qs)}개  "
                  f"({el/i:.1f}초/청크, 남은 시간 ~{eta/60:.0f}분)")
            for q in qs:
                print(f"      · {q}")

    print(f"\n완료: {ok}청크 성공 / {skipped}청크 스킵 / {time.time()-t0:.0f}초")
    print(f"저장: {args.out}")


if __name__ == "__main__":
    main()
