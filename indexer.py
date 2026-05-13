import requests
import json
import re
import tomllib
import bm25_store


from pathlib import Path
import chromadb
from markdown import markdown
from bs4 import BeautifulSoup
from chromadb.utils import embedding_functions

from chunker import chunk_text


# 설정 로드 (secrets.toml)

_secrets_path = Path(__file__).parent / ".streamlit" / "secrets.toml"
with open(_secrets_path, "rb") as f:
    _secrets = tomllib.load(f)

WIKI_URL = f"{_secrets['WIKI_BASE_URL']}/graphql"
API_TOKEN = _secrets["WIKI_API_TOKEN"]
CHROMA_PATH = _secrets.get("CHROMA_PATH", "./chroma_db")
COLLECTION_NAME = _secrets.get("COLLECTION_NAME", "wiki_knowledge")

headers = {
    "Authorization": f"Bearer {API_TOKEN}",
    "Content-Type": "application/json"
}


# 벡터 DB 및 임베딩 설정

chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)

sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="jhgan/ko-sroberta-multitask"
)

collection = chroma_client.get_or_create_collection(
    name=COLLECTION_NAME,
    embedding_function=sentence_transformer_ef
)


# Wiki.js API

def fetch_page_list():
    query = """
    query {
      pages {
        list(locale: "ko") {
          id
          path
          title
          tags
        }
      }
    }
    """
    try:
        response = requests.post(WIKI_URL, json={'query': query}, headers=headers)
        if response.status_code == 200:
            data = response.json()
            return data.get('data', {}).get('pages', {}).get('list', [])
        return []
    except Exception as e:
        print(f"목록 수집 에러: {e}")
        return []

def fetch_page_content(page_id):
    query = """
    query ($id: Int!) {
      pages {
        single(id: $id) {
          content
        }
      }
    }
    """
    try:
        variables = {"id": page_id}
        response = requests.post(WIKI_URL, json={'query': query, 'variables': variables}, headers=headers)
        if response.status_code == 200:
            data = response.json()
            return data.get('data', {}).get('pages', {}).get('single', {}).get('content', '')
        return ""
    except Exception as e:
        print(f"본문 수집 에러 (ID {page_id}): {e}")
        return ""



# 데이터 정제

def is_noise_line(line):
    line = line.strip()
    if not line:
        return True
    
    # 한글, 영문, 숫자를 제외한 특수문자 개수 계산
    special_chars = re.sub(r'[a-zA-Z0-9가-힣\s]', '', line)
    
    # 줄 전체 길이 대비 특수문자 비중이 30% 이상이면 노이즈(아스키아트)로 간주
    if len(line) > 5 and (len(special_chars) / len(line)) > 0.3:
        return True
    return False

def clean_markdown(md_content):
    content = re.sub(r'```.*?```', '', md_content, flags=re.DOTALL)
    
    # HTML 변환 및 텍스트 추출
    html = markdown(content)
    soup = BeautifulSoup(html, "html.parser")
    raw_text = soup.get_text()
    
    # 줄 단위로 분석하여 아스키아트 라인 삭제
    lines = raw_text.split('\n')
    filtered_lines = [line for line in lines if not is_noise_line(line)]
    
    # 연속 공백 정리
    clean_text = " ".join(filtered_lines)
    return re.sub(r'\s+', ' ', clean_text).strip()



# 벡터 DB 저장

def save_to_vector_db(page, chunks):
    if not chunks:
        return

    ids = [f"page_{page['id']}_chunk_{i}" for i in range(len(chunks))]
    metadatas = [{
        "page_id": page['id'],
        "title": page['title'],
        "path": page['path']
    } for _ in range(len(chunks))]
    
    collection.add(
        ids=ids,
        documents=chunks,
        metadatas=metadatas
    )
    print(f"   (3) ChromaDB 저장 완료: {len(chunks)}개 청크 색인됨")


def collect_existing_page_ids() -> set:
    """ChromaDB에 현재 색인된 모든 page_id를 수집
    Wiki.js에서 삭제된 페이지의 잔여 청크를 정리하기 위해 사용
    """
    try:
        all_items = collection.get(include=["metadatas"])
    except Exception as e:
        print(f"-> ChromaDB 메타데이터 조회 실패: {e}")
        return set()

    existing_ids = set()
    metadatas = all_items.get("metadatas") if all_items else None
    if not metadatas:
        return existing_ids

    for m in metadatas:
        if m and "page_id" in m:
            existing_ids.add(m["page_id"])
    return existing_ids


# 메인 인덱싱 파이프라인

def run_full_indexing():
    print("=== Wiki RAG 인덱싱 시작 (Noise Filtering 활성화) ===")

    # 1. 페이지 목록 수집
    pages = fetch_page_list()
    if not pages:
        print("pages null")
        return

    # 2. PDF / 비-PDF 분류
    pdf_pages = [p for p in pages if 'pdf' in (p.get('tags') or [])]
    non_pdf_pages = [p for p in pages if 'pdf' not in (p.get('tags') or [])]

    wiki_page_ids = {p['id'] for p in pages}
    non_pdf_ids = {p['id'] for p in non_pdf_pages}

    print(f"-> 총 {len(pages)}개 문서 수집 "
          f"(재색인 대상: {len(non_pdf_pages)}개 / PDF 임베딩 보존: {len(pdf_pages)}개)\n")

    # 3. 선택적 청크 정리
    #    (a) 재색인 대상 페이지의 기존 청크 삭제
    #    (b) Wiki.js에서 삭제된 페이지의 잔여 청크 정리 (PDF였든 아니든)
    existing_ids_in_db = collect_existing_page_ids()
    orphan_ids = existing_ids_in_db - wiki_page_ids   # Wiki에 없는 page_id
    to_delete = non_pdf_ids | orphan_ids

    if to_delete:
        try:
            collection.delete(where={"page_id": {"$in": list(to_delete)}})
            print(f"-> 청크 정리 완료: 재색인 대상 {len(non_pdf_ids)}개 페이지 + "
                  f"Wiki 삭제분 {len(orphan_ids)}개 페이지\n")
        except Exception as e:
            print(f"-> 청크 정리 중 오류: {e}\n")
    else:
        print("-> 정리할 청크가 없습니다.\n")

    # 4. 비-PDF 페이지 재색인
    for page in non_pdf_pages:
        print(f"[{page['title']}] 처리 중...")
        
        raw_content = fetch_page_content(page['id'])
        
        if raw_content:
            # 정제 및 분할
            clean_text = clean_markdown(raw_content)
            
            # 본문이 필터링 후 너무 짧아지면 스킵
            if len(clean_text) < 10:
                print(f"   [정보] 필터링 후 유의미한 텍스트가 부족하여 스킵합니다.")
                print(f"--- 처리 완료 ---\n")
                continue

            chunks = chunk_text(clean_text)
            print(f"   (1) 정제 완료 (글자 수: {len(clean_text)})")
            print(f"   (2) 청킹 완료 (조각 수: {len(chunks)})")
            
            # 벡터 DB 저장
            save_to_vector_db(page, chunks)
        else:
            print(f"   [경고] 본문을 가져오지 못했습니다.")
        
        print(f"--- 처리 완료 ---\n")

    # ── BM25 인덱스 빌드 및 저장 ──────────────────────────────────────────
    bm25_path = _secrets.get("BM25_PATH", "./bm25_index.pkl")
    print("=== BM25 인덱스 빌드 중... ===")
    try:
        bm25_idx = bm25_store.build_from_chroma(collection)
        bm25_store.save(bm25_idx, bm25_path)
        print(f"=== BM25 인덱스 저장 완료: {bm25_path} ({len(bm25_idx.chunk_ids)}개 청크) ===")
    except Exception as e:
        print(f"=== [경고] BM25 빌드 실패 (다음 실행 시 재시도): {e} ===")

    if pdf_pages:
        print(f"=== PDF 임베딩 문서 {len(pdf_pages)}개는 보존되었습니다. ===")
    print("=== 모든 인덱싱 공정이 성공적으로 종료되었습니다. ===")

if __name__ == "__main__":
    run_full_indexing()