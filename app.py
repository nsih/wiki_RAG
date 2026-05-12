import streamlit as st
import chromadb
import requests
import json
import re
import logging
from chromadb.utils import embedding_functions
from io import BytesIO

# 코어 모듈 임포트
import wiki_builder
from chunker import chunk_text

logger = logging.getLogger(__name__)


# 세션 상태 초기화 콜백 (메뉴 전환 시 호출)

def reset_generation_state():
    if 'generation_config' in st.session_state:
        del st.session_state.generation_config
    if 'raw_text' in st.session_state:
        del st.session_state.raw_text
    if 'uploaded_file_buffer' in st.session_state:
        del st.session_state.uploaded_file_buffer

# 설정 값 (st.secrets에서 로드)

# Directory
CHROMA_PATH = st.secrets.get("CHROMA_PATH", "./chroma_db")
COLLECTION_NAME = st.secrets.get("COLLECTION_NAME", "wiki_knowledge")

# Wiki.js API
WIKI_BASE_URL = st.secrets["WIKI_BASE_URL"]
WIKI_URL = f"{WIKI_BASE_URL}/graphql"
WIKI_API_TOKEN = st.secrets["WIKI_API_TOKEN"]

# LM Studio (Search AI 모드 전용)
AI_WORKER_IP = st.secrets["AI_WORKER_IP"]
AI_WORKER_PORT = st.secrets.get("AI_WORKER_PORT", 1234)
AI_WORKER_ENDPOINT = f"http://{AI_WORKER_IP}:{AI_WORKER_PORT}/v1/chat/completions"
AI_MODEL_NAME = st.secrets.get("AI_MODEL_NAME", "gemma-3n-e4b-it-text")

# 헬퍼 함수
@st.cache_resource
def load_vectordb():
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="jhgan/ko-sroberta-multitask")
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    return client.get_or_create_collection(name=COLLECTION_NAME, embedding_function=ef)

def call_llm(messages, context):
    url = AI_WORKER_ENDPOINT

    SYSTEM_PROMPT = (
        "당신은 사내 지식 기반 챗봇입니다. "
        "답변은 반드시 한국어로, 제공되는 참고 문서를 바탕으로 "
        "객관적이고 명확하게 답변하십시오."
    )

    # 이전 대화 내역 포맷팅 (메모리 최적화를 위해 최근 4턴 유지)
    recent_history = messages[:-1][-4:] if len(messages) > 1 else []
    formatted_messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    for msg in recent_history:
        formatted_messages.append({
            "role": msg["role"],
            "content": msg["content"]
        })

    # 마지막 '현재 질문'에 RAG 검색 컨텍스트 결합
    last_msg = messages[-1]["content"]
    augmented_prompt = f"[참고 문서]\n{context}\n\n[질문]\n{last_msg}"
    formatted_messages.append({"role": "user", "content": augmented_prompt})

    payload = {
        "model": AI_MODEL_NAME,
        "messages": formatted_messages,
        "stream": False,
        "temperature": 0.2,
        "max_tokens": 1024
    }

    try:
        # 내장 그래픽 환경에서의 E4B 연산 지연을 고려해 타임아웃 여유
        res = requests.post(
            url, json=payload,
            headers={"Content-Type": "application/json"},
            timeout=120
        )
        if res.status_code == 200:
            return res.json()["choices"][0]["message"]["content"]
        else:
            return f"LM Studio 응답 오류: {res.status_code} - {res.text}"
    except Exception as e:
        return f"LM Studio 연산 서버({AI_WORKER_IP}:{AI_WORKER_PORT}) 통신 실패: {e}"

def search_similar_titles(collection, query_title: str, threshold: float = 0.2):
    results = collection.query(query_texts=[query_title], n_results=3, include=["metadatas", "distances"])
    similar = []
    seen = set()
    if results['ids'] and results['ids'][0]:
        for i in range(len(results['ids'][0])):
            meta = results['metadatas'][0][i]
            dist = results['distances'][0][i]
            if dist <= threshold and meta is not None and 'path' in meta and meta['path'] not in seen:
                similar.append({"title": meta['title'], "path": meta['path'], "distance": dist})
                seen.add(meta['path'])
    return similar

def update_vector_db(collection, page_id: int, title: str, path: str, content: str):
    """페이지의 기존 청크를 삭제하고 새 내용으로 재색인합니다.
    indexer.py와 동일한 chunker.chunk_text를 사용해 청크 일관성을 보장합니다.
    """
    # 기존 청크 삭제 (없거나 실패해도 진행 가능)
    try:
        collection.delete(where={"page_id": page_id})
    except Exception as e:
        logger.warning(f"기존 청크 삭제 실패 (page_id={page_id}): {e}")

    chunks = chunk_text(content)
    if not chunks:
        return 0

    ids = [f"page_{page_id}_chunk_{i}" for i in range(len(chunks))]
    metas = [{"page_id": page_id, "title": title, "path": path} for _ in range(len(chunks))]
    collection.add(ids=ids, documents=chunks, metadatas=metas)
    return len(chunks)

@st.dialog("⚠️ 중복 감지")
def overwrite_confirm_dialog(similar_docs, original_title, final_path, is_exact=False):
    if is_exact: st.error(f"동일 경로(`{final_path}`)가 이미 존재합니다.")
    else: st.warning("유사한 문서가 발견되었습니다.")
    for doc in similar_docs:
        st.write(f"- **{doc['title']}** ({doc['path']}) / 유사도: {max(0, 1 - doc['distance']):.1%}")
    st.markdown("---")
    if st.button("덮어쓰기 (Update)", use_container_width=True):
        st.session_state.generation_config['action'] = 'update'
        st.session_state.generation_config['path'] = similar_docs[0]['path']
        st.rerun()
    if st.button("신규 생성", use_container_width=True):
        st.session_state.generation_config['action'] = 'create'
        st.rerun()

# 메인 UI
st.set_page_config(page_title="CSU WIKI AI", layout="centered")
try:
    collection = load_vectordb()
except Exception as e:
    st.error(f"DB 로드 실패: {e}")
    st.stop()

app_mode = st.sidebar.radio(
    "모드 선택",
    ["Search AI", "PDF -> Wiki Data"],
    on_change=reset_generation_state
)

if app_mode == "Search AI":
    st.title("🏫 CSU wiki AI")
    if "messages" not in st.session_state: st.session_state.messages = []
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]): st.markdown(msg["content"])

    if prompt := st.chat_input("질문하세요"):
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("assistant"):
            res = collection.query(query_texts=[prompt], n_results=3)

            if not res['documents'] or not res['documents'][0]:
                ctx = "검색된 관련 문서가 없습니다. 이전 대화 문맥을 참고하여 답변하세요."
                titles = set()
            else:
                valid_docs = [doc for doc in res['documents'][0] if doc is not None]
                if valid_docs:
                    ctx = "\n---\n".join(valid_docs)
                else:
                    ctx = "검색된 관련 문서가 없습니다. 이전 대화 문맥을 참고하여 답변하세요."

                valid_metas = [m for m in res['metadatas'][0] if m is not None]
                titles = set([m.get('title', '제목 없음') for m in valid_metas])

            ans = call_llm(st.session_state.messages, ctx)

            if titles:
                ans += "\n\n**[출처]**\n" + "\n".join([f"- {t}" for t in titles])

            st.markdown(ans)
            st.session_state.messages.append({"role": "assistant", "content": ans})

elif app_mode == "PDF -> Wiki Data":
    st.title("📄 PDF -> Wiki Data")

    if 'generation_config' not in st.session_state:
        with st.form("upload_form"):
            file = st.file_uploader("PDF 선택", type=["pdf"])
            dept = st.selectbox("부서", ["정보전산원", "교무처", "학생처", "기획처"])
            title = st.text_input("문서 제목")
            if st.form_submit_button("시작"):
                if file and title:
                    safe_title = re.sub(r'[^\w가-힣-]', '', re.sub(r'[\s/]+', '-', title.strip()))
                    final_path = f"{dept}/{safe_title}"

                    with st.spinner("PDF 파싱 및 검사 중..."):
                        st.session_state.raw_text = wiki_builder.extract_text_from_pdf(BytesIO(file.getvalue()))
                        is_exists, existing_id = wiki_builder.check_page_exists(WIKI_URL, WIKI_API_TOKEN, final_path)
                        similar = search_similar_titles(collection, title)

                    # 공통: 작업 설정 초안 (action은 분기에서 결정)
                    base_config = {
                        'action': 'create',
                        'path': final_path,
                        'page_id': None,
                        'title': title
                    }

                    if is_exists:
                        st.session_state.generation_config = base_config
                        overwrite_confirm_dialog(
                            [{'title': title, 'path': final_path, 'distance': 0}],
                            title, final_path, True
                        )
                    elif similar:
                        st.session_state.generation_config = base_config
                        overwrite_confirm_dialog(similar, title, final_path)
                    else:
                        st.session_state.generation_config = base_config
                        st.rerun()
    else:
        config = st.session_state.generation_config
        st.info(f"🚀 처리 중 (대상: `{config['path']}`)")

        try:
            # PyMuPDF 추출 원본을 그대로 사용
            refined_md = st.session_state.raw_text
            with st.expander("📄 추출된 마크다운 미리보기", expanded=False):
                st.markdown(refined_md)
            st.success(f"✅ 추출 완료 (길이: {len(refined_md):,}자)")

            with st.spinner("Wiki.js 전송 중..."):
                if config['action'] == 'update':
                    _, existing_id = wiki_builder.check_page_exists(WIKI_URL, WIKI_API_TOKEN, config['path'])
                    page_id = wiki_builder.update_wikijs_page(
                        WIKI_URL, WIKI_API_TOKEN, existing_id,
                        config['title'], refined_md, config['path']
                    )
                else:
                    page_id = wiki_builder.create_wikijs_page(
                        WIKI_URL, WIKI_API_TOKEN,
                        config['title'], refined_md, config['path']
                    )
                st.success(f"✅ 위키 반영 완료 (ID: {page_id})")

            with st.spinner("RAG 엔진 동기화 중..."):
                cnt = update_vector_db(collection, page_id, config['title'], config['path'], refined_md)
                st.success(f"✅ 인덱싱 완료 ({cnt}개 청크)")

            st.button("새로운 작업 시작", on_click=reset_generation_state, type="primary")

        except Exception as e:
            st.error(f"오류: {e}")
            st.button("초기화 및 돌아가기", on_click=reset_generation_state)