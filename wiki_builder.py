import requests
import logging
from typing import Optional, Tuple
import fitz  # PyMuPDF
import pymupdf4llm
from io import BytesIO

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(funcName)s] %(message)s')
logger = logging.getLogger(__name__)


# ── 상수 ────────────────────────────────────────────────────────────────────

# indexer.py가 'pdf' 태그를 기준으로 재색인 대상에서 제외한다.
# 'auto'는 사용자가 수동 작성한 문서와 구분하기 위한 마커.
_TAG_AUTO_CREATED = ["auto", "pdf"]
_TAG_AUTO_UPDATED_EXTRA = ["pdf", "updated"]  # update 시 기존 태그와 합치는 보장 집합

_REQUEST_TIMEOUT_QUERY = 30
_REQUEST_TIMEOUT_MUTATION = 60


class WikiBuilderError(Exception):
    pass


# ── URL 정규화 ──────────────────────────────────────────────────────────────

def _fix_graphql_url(url: str) -> str:
    url = url.rstrip('/')
    if not url.endswith('/graphql'):
        url += '/graphql'
    return url


# ── GraphQL 공통 헬퍼 ────────────────────────────────────────────────────────

def _post_graphql(endpoint: str, api_token: str, query: str,
                  variables: dict, timeout: int) -> dict:
    """GraphQL POST 호출 공통화. requests 예외는 WikiBuilderError로 래핑한다."""
    headers = {"Authorization": f"Bearer {api_token}", "Content-Type": "application/json"}
    try:
        response = requests.post(
            endpoint, headers=headers,
            json={"query": query, "variables": variables},
            timeout=timeout
        )
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        raise WikiBuilderError(f"Wiki.js 통신 오류: {str(e)}")


def _handle_graphql_errors(errors: list, *, raise_on_other: bool) -> bool:
    """GraphQL errors 배열을 처리한다.

    - 권한/인증 관련 메시지는 즉시 WikiBuilderError를 던진다.
    - 그 외 에러는 raise_on_other에 따라 분기:
        True  → WikiBuilderError 발생 (mutation 경로)
        False → False 반환 (query 경로 — PageNotFound를 '없음'으로 취급)

    Returns:
        에러가 있었는지 여부. (False면 정상)
    """
    if not errors:
        return False

    perm_keywords = ("forbidden", "unauthorized", "permission")
    for err in errors:
        msg = (err.get("message") or "").lower()
        if any(k in msg for k in perm_keywords):
            raise WikiBuilderError(f"권한 부족: {err.get('message')}")

    if raise_on_other:
        # mutation 경로: 권한 외 에러도 실패로 간주해 원인을 그대로 노출
        msgs = "; ".join(e.get("message", "") for e in errors)
        raise WikiBuilderError(f"GraphQL 오류: {msgs}")

    # query 경로: 권한 외 에러(보통 PageNotFound)는 호출측에서 '없음'으로 처리
    logger.debug(f"GraphQL 비치명적 에러: {errors}")
    return True


# ── PDF 추출 ────────────────────────────────────────────────────────────────

import re

def _merge_page_breaks(md_text: str) -> str:
    page_sep = re.compile(r'\n{0,2}(?<!\|)-{3,}(?!\|)\n{0,2}')
    
    segments = page_sep.split(md_text)
    merged_parts = []
    
    for i, seg in enumerate(segments):
        if i == 0:
            merged_parts.append(seg)
            continue
        
        prev_tail = merged_parts[-1].rstrip()
        cur_head = seg.lstrip()
        
        if prev_tail.endswith('|') and cur_head.startswith('|'):
            merged_parts[-1] = prev_tail + '\n' + cur_head
        elif prev_tail and prev_tail[-1] not in '.。!?|#\n':
            merged_parts[-1] = prev_tail + cur_head
        else:
            merged_parts[-1] = prev_tail + '\n\n' + cur_head
    
    result = ''.join(merged_parts)
    result = re.sub(r'([가-힣a-zA-Z])-\n([가-힣a-zA-Z])', r'\1\2', result)
    
    lines = result.split('\n')
    out = []
    for line in lines:
        stripped = line.strip()
        if re.match(r'^\|(?:[-:]+\|)+$', stripped):
            prev = next((l.strip() for l in reversed(out) if l.strip()), '')
            if re.match(r'^\|\s*\d+\s*\|', prev):  # 직전이 데이터 행이면 제거
                continue
        out.append(line)

    result = '\n'.join(out)
    result = re.sub(r'\n{3,}', '\n\n', result)
    result = re.sub(r'(\|[^\n]+)\n\n+(\|)', r'\1\n\2', result)

    return result


def extract_text_from_pdf(pdf_file_obj: BytesIO) -> str:
    try:
        pdf_file_obj.seek(0)
        doc = fitz.open(stream=pdf_file_obj.read(), filetype="pdf")
        md_text = pymupdf4llm.to_markdown(doc)
        return _merge_page_breaks(md_text)  # ← 추가
    except Exception as e:
        logger.error(f"PDF 추출 실패: {e}")
        raise WikiBuilderError(f"PDF 파싱 에러: {str(e)}")


# ── 페이지 조회 ──────────────────────────────────────────────────────────────

def check_page_exists(wiki_url: str, api_token: str, target_path: str,
                      locale: str = "ko") -> Tuple[bool, Optional[int]]:
    """path로 페이지 존재 여부를 단건 조회."""
    wiki_url = _fix_graphql_url(wiki_url)
    query = """
    query ($path: String!, $locale: String!) {
      pages {
        singleByPath(path: $path, locale: $locale) {
          id
        }
      }
    }
    """
    variables = {"path": target_path, "locale": locale}

    data = _post_graphql(wiki_url, api_token, query, variables, _REQUEST_TIMEOUT_QUERY)

    # 에러 분기 — query 경로이므로 PageNotFound 류는 '없음'으로 폴백
    if _handle_graphql_errors(data.get("errors", []), raise_on_other=False):
        return False, None

    page = (data.get("data") or {}).get("pages", {}).get("singleByPath")
    if page and page.get("id"):
        return True, int(page["id"])
    return False, None


def fetch_page_tags(wiki_url: str, api_token: str, page_id: int) -> list:
    """페이지의 기존 태그 목록을 가져옵니다. 업데이트 시 태그 보존용.

    Note: pages.single은 tags를 [Tag] 객체 배열로 반환하므로 tag 필드를 꺼내야 함.
    (pages.list는 [String] 문자열 배열 - 다름)

    실패해도 update 자체를 막지 않기 위해 빈 리스트를 반환한다.
    """
    wiki_url = _fix_graphql_url(wiki_url)
    query = """
    query ($id: Int!) {
      pages {
        single(id: $id) {
          tags { tag }
        }
      }
    }
    """
    try:
        data = _post_graphql(wiki_url, api_token, query,
                             {"id": page_id}, _REQUEST_TIMEOUT_QUERY)
        if data.get("errors"):
            logger.warning(f"태그 조회 GraphQL 에러 (page_id={page_id}): {data['errors']}")
            return []

        tags_data = ((data.get("data") or {})
                     .get("pages", {}).get("single", {}) or {}).get("tags", [])
        return [t["tag"] for t in tags_data if t.get("tag")]
    except WikiBuilderError as e:
        logger.warning(f"태그 조회 실패 (page_id={page_id}): {e}")
        return []


# ── 페이지 생성/업데이트 ─────────────────────────────────────────────────────

def create_wikijs_page(wiki_url: str, api_token: str, title: str,
                       content: str, path: str) -> int:
    wiki_url = _fix_graphql_url(wiki_url)
    query = """
    mutation CreatePage($content: String!, $path: String!, $title: String!, $tags: [String]!) {
      pages {
        create(content: $content, description: "AI 자동 생성", editor: "markdown", isPublished: true, 
               isPrivate: false, locale: "ko", path: $path, tags: $tags, title: $title) {
          responseResult { succeeded message }
          page { id }
        }
      }
    }
    """
    variables = {
        "content": content,
        "path": path,
        "title": title,
        "tags": _TAG_AUTO_CREATED,
    }
    return _execute_mutation(wiki_url, api_token, query, variables, "create")


def update_wikijs_page(wiki_url: str, api_token: str, page_id: int,
                       title: str, content: str, path: str) -> int:
    wiki_url = _fix_graphql_url(wiki_url)

    # 기존 태그 보존 + 'pdf' 태그 보장 (사용자가 수동으로 추가한 태그도 유지)
    existing_tags = fetch_page_tags(wiki_url, api_token, page_id)
    merged_tags = list(set(existing_tags + _TAG_AUTO_UPDATED_EXTRA))

    query = """
    mutation UpdatePage($id: Int!, $content: String!, $path: String!, $title: String!, $tags: [String]!) {
      pages {
        update(id: $id, content: $content, description: "AI 자동 업데이트", editor: "markdown", 
               isPublished: true, isPrivate: false, locale: "ko", path: $path, tags: $tags, title: $title) {
          responseResult { succeeded message }
          page { id }
        }
      }
    }
    """
    variables = {
        "id": page_id,
        "content": content,
        "path": path,
        "title": title,
        "tags": merged_tags,
    }
    return _execute_mutation(wiki_url, api_token, query, variables, "update")


def _execute_mutation(endpoint: str, api_token: str, query: str,
                      variables: dict, m_type: str) -> int:
    """create/update 공통 처리.

    실패 경로 우선순위:
      1) GraphQL errors 배열 (data가 null인 케이스 포함)
      2) responseResult.succeeded == false
      3) page.id 누락
    어느 단계에서 실패했는지 원인을 명확히 분리해 메시지에 노출한다.
    """
    data = _post_graphql(endpoint, api_token, query, variables, _REQUEST_TIMEOUT_MUTATION)

    # (1) GraphQL 레벨 에러 — data가 null이거나 부분 실패한 경우를 잡는다.
    # mutation은 PageNotFound 같은 '정상 분기' 에러가 없으므로 모두 실패로 처리.
    _handle_graphql_errors(data.get("errors", []), raise_on_other=True)

    # (2) mutation 본문에서 succeeded 확인
    result = ((data.get("data") or {}).get("pages") or {}).get(m_type) or {}
    response_result = result.get("responseResult") or {}

    if not response_result.get("succeeded"):
        msg = response_result.get("message") or "응답에 succeeded/message가 없습니다."
        raise WikiBuilderError(f"Wiki.js {m_type} 실패: {msg}")

    # (3) page.id 추출 — 성공 응답이지만 id가 없는 비정상 케이스 방어
    page = result.get("page") or {}
    page_id = page.get("id")
    if page_id is None:
        raise WikiBuilderError(f"Wiki.js {m_type} 응답에 page.id가 없습니다.")
    return int(page_id)