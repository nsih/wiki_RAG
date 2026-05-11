"""
indexer.py와 app.py가 동일한 청킹 로직을 공유하도록 분리한 모듈.
두 곳에서 청크 크기/오버랩이 어긋나면 같은 page_id의 청크가 서로 다른 길이로 색인되어 검색 품질이 나빠지므로 반드시 함께 사용한다.
"""


def chunk_text(text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> list:
    """슬라이딩 윈도우로 텍스트를 분할한다.

    Args:
        text: 분할 대상 텍스트
        chunk_size: 청크당 최대 글자 수 (기본 500)
        chunk_overlap: 청크 간 중복 글자 수 (기본 50)

    Returns:
        청크 문자열 리스트. 빈 입력에는 빈 리스트.

    Raises:
        ValueError: chunk_overlap이 chunk_size 이상인 경우
    """
    if not text:
        return []

    if chunk_overlap >= chunk_size:
        raise ValueError(
            f"chunk_overlap({chunk_overlap})은 chunk_size({chunk_size})보다 작아야 합니다."
        )

    step = chunk_size - chunk_overlap
    return [text[i:i + chunk_size] for i in range(0, len(text), step)]