# tokenizer.py
import re

# 끝조사 제거 대상 목록 (빈도 높은 한국어 조사)
_JOSA = ("을", "를", "이", "가", "은", "는", "의", "에", "과", "와", "도", "로", "으로", "에서")

def tokenize_ko(text: str) -> list[str]:
    """한국어/영문/숫자 덩어리를 추출하고 끝조사·단음절 토큰 제거

    형태소 분석기 없이 정규식 + 조사 스트리핑 휴리스틱으로 처리.
    indexer.py와 app.py가 동일 함수를 공유해 BM25 인덱스 일관성을 보장한다.
    """
    # 한글 음절 덩어리 / 영문 단어 / 숫자열 추출
    raw_tokens: list[str] = re.findall(r'[가-힣]+|[a-zA-Z]+|[0-9]+', text)

    result: list[str] = []
    for tok in raw_tokens:
        # 한글 토큰에 한해 끝조사 제거 시도
        if re.match(r'^[가-힣]+$', tok):
            for josa in _JOSA:
                if tok.endswith(josa) and len(tok) > len(josa):
                    tok = tok[: -len(josa)]
                    break  # 조사는 최대 1회만 제거

        # 2글자 미만 토큰 제거 (노이즈 억제)
        if len(tok) >= 2:
            result.append(tok.lower())  # 영문 소문자 정규화

    return result