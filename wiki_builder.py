import requests
import json
import logging
from typing import Optional, Dict, Any, Tuple
import fitz  # PyMuPDF
import pymupdf4llm
from io import BytesIO

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(funcName)s] %(message)s')
logger = logging.getLogger(__name__)

class WikiBuilderError(Exception):
    pass

def _fix_graphql_url(url: str) -> str:
    url = url.rstrip('/')
    if not url.endswith('/graphql'):
        url += '/graphql'
    return url

def extract_text_from_pdf(pdf_file_obj: BytesIO) -> str:
    """PyMuPDF를 활용해 PDF의 표와 레이아웃을 마크다운 형태로 1차 추출합니다."""
    try:
        pdf_file_obj.seek(0)
        doc = fitz.open(stream=pdf_file_obj.read(), filetype="pdf")
        md_text = pymupdf4llm.to_markdown(doc)
        return md_text
    except Exception as e:
        logger.error(f"PDF 추출 실패: {e}")
        raise WikiBuilderError(f"PDF 파싱 에러: {str(e)}")

def refine_text_with_ollama(raw_text: str, model_name: str = "qwen2.5:7b", endpoint: str = "http://localhost:11434/api/generate"):
    """청킹(Chunking)을 통해 누락 없이 마크다운을 정제하며, 표 구조를 100% 보존합니다."""
    system_prompt = (
        "당신은 PDF Text를 마크다운(Markdown) 포맷으로 변환하는 전문 테크니컬 라이터입니다. "
        "사용자가 제공한 텍스트의 내용을 단 한 글자도 누락하거나 요약하지 마십시오. "
        "특히 입력 데이터에 이미 표(Table) 형태가 포함되어 있다면 그 구조와 데이터를 절대 변경하지 마십시오. "
        "본문 외의 인사말이나 부연 설명은 절대 출력하지 마십시오."
        "문서를 작성할 때 절대 블록을 만들지 마십시오"
    )
    
    paragraphs = raw_text.split('\n\n')
    chunks = []
    current_chunk = ""
    
    for p in paragraphs:
        if len(current_chunk) + len(p) < 1500: 
            current_chunk += p + "\n\n"
        else:
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
            current_chunk = p + "\n\n"
    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    for idx, chunk in enumerate(chunks):
        payload = {
            "model": model_name,
            "prompt": f"다음 텍스트를 마크다운으로 작성하라. 단, 작성시 '```markdown'을 절대 포함하지 말것. :\n\n{chunk}",
            "system": system_prompt,
            "stream": True,
            "options": {"temperature": 0.1, "top_p": 0.9}
        }
        
        try:
            response = requests.post(endpoint, headers={"Content-Type": "application/json"},
                                     json=payload, timeout=300, stream=True)
            response.raise_for_status()
            
            for line in response.iter_lines():
                if line:
                    data = json.loads(line.decode('utf-8'))
                    response_piece = data.get("response", "")
                    if response_piece:
                        yield response_piece
                    if data.get("done"):
                        break
            yield "\n\n"
            
        except Exception as e:
            logger.error(f"Ollama 오류 (Chunk {idx+1}/{len(chunks)}): {str(e)}")
            yield f"\n\n**[오류: 일부 구간 변환 실패]**\n\n"

def check_page_exists(wiki_url: str, api_token: str, target_path: str, locale: str = "ko") -> Tuple[bool, Optional[int]]:
    #path로 페이지 존재 여부를 단건 조회
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
    headers = {"Authorization": f"Bearer {api_token}", "Content-Type": "application/json"}
    
    try:
        response = requests.post(
            wiki_url, headers=headers,
            json={"query": query, "variables": variables},
            timeout=30
        )
        response.raise_for_status()
        data = response.json()
        
        # GraphQL 에러 분기
        if data.get("errors"):
            for err in data["errors"]:
                msg = (err.get("message") or "").lower()
                if "forbidden" in msg or "unauthorized" in msg or "permission" in msg:
                    raise WikiBuilderError(f"권한 부족: {err.get('message')}")
            # 권한 외 에러 (보통 PageNotFound) → 페이지 없음으로 간주
            logger.debug(f"singleByPath 에러를 페이지 없음으로 처리: {data['errors']}")
            return False, None
        
        page = data.get("data", {}).get("pages", {}).get("singleByPath")
        if page and page.get("id"):
            return True, int(page["id"])
        return False, None
    except WikiBuilderError:
        raise
    except Exception as e:
        raise WikiBuilderError(f"조회 실패: {str(e)}")

def fetch_page_tags(wiki_url: str, api_token: str, page_id: int) -> list:
    """페이지의 기존 태그 목록을 가져옵니다. 업데이트 시 태그 보존용.
    
    Note: pages.single은 tags를 [Tag] 객체 배열로 반환하므로 tag 필드를 꺼내야 함.
    (pages.list는 [String] 문자열 배열 - 다름)
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
    headers = {"Authorization": f"Bearer {api_token}", "Content-Type": "application/json"}
    try:
        response = requests.post(wiki_url, headers=headers, 
                                 json={"query": query, "variables": {"id": page_id}}, 
                                 timeout=30)
        response.raise_for_status()
        tags_data = response.json().get("data", {}).get("pages", {}).get("single", {}).get("tags", [])
        return [t["tag"] for t in tags_data if t.get("tag")]
    except Exception as e:
        logger.warning(f"태그 조회 실패 (page_id={page_id}): {e}")
        return []

def create_wikijs_page(wiki_url: str, api_token: str, title: str, content: str, path: str) -> int:
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
        "tags": ["auto", "pdf"]  # 'pdf' 태그로 indexer가 스킵 판단
    }
    return _execute_mutation(wiki_url, api_token, query, variables, "create")

def update_wikijs_page(wiki_url: str, api_token: str, page_id: int, title: str, content: str, path: str) -> int:
    wiki_url = _fix_graphql_url(wiki_url)
    
    # 기존 태그 보존 + 'pdf' 태그 보장 (사용자가 수동으로 추가한 태그도 유지)
    existing_tags = fetch_page_tags(wiki_url, api_token, page_id)
    merged_tags = list(set(existing_tags + ["pdf", "updated"]))
    
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
        "tags": merged_tags
    }
    return _execute_mutation(wiki_url, api_token, query, variables, "update")

def _execute_mutation(endpoint, api_token, query, variables, m_type) -> int:
    headers = {"Authorization": f"Bearer {api_token}", "Content-Type": "application/json"}
    try:
        response = requests.post(endpoint, headers=headers, json={"query": query, "variables": variables}, timeout=60)
        response.raise_for_status()
        data = response.json()
        result = data.get("data", {}).get("pages", {}).get(m_type, {})
        if not result.get("responseResult", {}).get("succeeded"):
            raise WikiBuilderError(result.get("responseResult", {}).get("message"))
        return result.get("page", {}).get("id")
    except Exception as e:
        raise WikiBuilderError(f"Wiki.js 통신 오류: {str(e)}")