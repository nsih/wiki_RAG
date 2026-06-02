import streamlit as st
import chromadb
import requests
import re
import logging
import threading
import os
import datetime
from chromadb.utils import embedding_functions
from io import BytesIO

# 코어 모듈 임포트
import wiki_builder
import bm25_store
from retriever import hybrid_search
from chunker import chunk_text

logger = logging.getLogger(__name__)

# BM25 in-place 패치 보호용 락 (멀티스레드 동시 업로드 방어)
_bm25_lock = threading.Lock()

# BM25 메모리 패치 시각을 재시작 후에도 유지하기 위한 사이드카 파일
# bm25_index.pkl → bm25_index.pkl.patched
_PATCH_TIME_FILE = str(st.secrets.get("BM25_PATH", "./bm25_index.pkl")) + ".patched"


# ── 세션 상태 초기화 콜백 (메뉴 전환 시 호출) ────────────────────────────────

def reset_generation_state():
    for k in ('generation_config', 'raw_text', 'uploaded_file_buffer', 'pending_check'):
        st.session_state.pop(k, None)


# ── 설정 값 (st.secrets에서 로드) ─────────────────────────────────────────────

CHROMA_PATH = st.secrets.get("CHROMA_PATH", "./chroma_db")
COLLECTION_NAME = st.secrets.get("COLLECTION_NAME", "wiki_knowledge")

WIKI_BASE_URL = st.secrets["WIKI_BASE_URL"]
WIKI_URL = f"{WIKI_BASE_URL}/graphql"
WIKI_API_TOKEN = st.secrets["WIKI_API_TOKEN"]

AI_WORKER_IP = st.secrets["AI_WORKER_IP"]
AI_WORKER_PORT = st.secrets.get("AI_WORKER_PORT", 1234)
AI_WORKER_ENDPOINT = f"http://{AI_WORKER_IP}:{AI_WORKER_PORT}/v1/chat/completions"
AI_MODEL_NAME = st.secrets.get("AI_MODEL_NAME", "")


# ── 헬퍼 함수 ────────────────────────────────────────────────────────────────

@st.cache_resource
def load_vectordb():
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="jhgan/ko-sroberta-multitask"
    )
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    return client.get_or_create_collection(name=COLLECTION_NAME, embedding_function=ef)


@st.cache_resource
def load_bm25_index():
    bm25_path = st.secrets.get("BM25_PATH", "./bm25_index.pkl")
    return bm25_store.load(bm25_path)


def _save_patch_time() -> None:
    try:
        with open(_PATCH_TIME_FILE, "w") as f:
            f.write(datetime.datetime.now().isoformat())
    except Exception as e:
        logger.warning(f"패치 시각 파일 저장 실패: {e}")


def _load_patch_time() -> datetime.datetime | None:
    try:
        with open(_PATCH_TIME_FILE) as f:
            return datetime.datetime.fromisoformat(f.read().strip())
    except Exception:
        return None


def render_bm25_status(placeholder) -> None:
    bm25_path = st.secrets.get("BM25_PATH", "./bm25_index.pkl")

    if not os.path.exists(bm25_path):
        placeholder.caption("⚠️ BM25 인덱스 없음 — 벡터 단독 검색 중")
        return

    mtime = datetime.datetime.fromtimestamp(os.path.getmtime(bm25_path))

    # 세션 → 사이드카 파일 순으로 패치 시각 확인
    last_patch = st.session_state.get("bm25_last_patch") or _load_patch_time()

    if last_patch and last_patch > mtime:
        # 패치가 더 최신 → 메모리 패치 상태
        placeholder.caption(f"인덱스 최종 갱신 (메모리): {last_patch:%Y-%m-%d %H:%M}")
    else:
        # indexer 배치가 더 최신이거나 패치 없음 → pkl mtime 기준
        placeholder.caption(f"인덱스 최종 갱신: {mtime:%Y-%m-%d %H:%M}")


def _get_loaded_model_id() -> str:
    """LM Studio /v1/models 에서 현재 로드된 모델 ID를 조회한다.
    실패 시 secrets의 AI_MODEL_NAME 또는 '알 수 없음'을 반환한다.
    """
    try:
        res = requests.get(
            f"http://{AI_WORKER_IP}:{AI_WORKER_PORT}/v1/models",
            timeout=5
        )
        if res.status_code == 200:
            models = res.json().get("data", [])
            if models:
                return models[0].get("id", AI_MODEL_NAME or "알 수 없음")
    except Exception:
        pass
    return AI_MODEL_NAME or "알 수 없음"


def call_llm(messages, context):
    SYSTEM_PROMPT = (
        "/no_think "
        "당신은 RAG 챗봇입니다. "
        "답변은 한국어로 제공되는 참고 문서를 바탕으로 정확하고 명료하게 답변해주세요. "
    )

    # 이전 대화 내역 포맷팅 (최근 5개 메시지)
    recent_history = messages[:-1][-5:] if len(messages) > 1 else []

    # 시스템 프롬프트를 첫 번째 user 메시지 앞에 인라인으로 병합
    formatted_messages = []
    for i, msg in enumerate(recent_history):
        content = msg["content"]
        if i == 0 and msg["role"] == "user":
            content = f"{SYSTEM_PROMPT}\n\n{content}"
        formatted_messages.append({"role": msg["role"], "content": content})

    # 마지막 현재 질문에 RAG 검색 컨텍스트 결합
    last_msg = messages[-1]["content"]
    augmented_prompt = f"[참고 문서]\n{context}\n\n[질문]\n{last_msg}"

    # 대화 이력이 없으면(첫 턴) 시스템 프롬프트를 현재 질문에 병합
    if not formatted_messages:
        augmented_prompt = f"{SYSTEM_PROMPT}\n\n{augmented_prompt}"

    formatted_messages.append({"role": "user", "content": augmented_prompt})

    payload = {
        "messages": formatted_messages,
        "stream": False,
        "temperature": 0.2,
        "max_tokens": 4096,
    }
    # AI_MODEL_NAME이 비어 있으면 LM Studio가 현재 로드된 모델을 사용하도록 필드 생략
    if AI_MODEL_NAME:
        payload["model"] = AI_MODEL_NAME

    try:
        res = requests.post(
            AI_WORKER_ENDPOINT,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=600,
        )
        if res.status_code == 200:
            return res.json()["choices"][0]["message"]["content"]
        else:
            return f"LM Studio 응답 오류: {res.status_code} - {res.text}"
    except Exception as e:
        return f"LM Studio 연산 서버({AI_WORKER_IP}:{AI_WORKER_PORT}) 통신 실패: {e}"


def search_similar_titles(collection, query_title: str, threshold: float = 0.2):
    results = collection.query(
        query_texts=[query_title], n_results=3,
        include=["metadatas", "distances"]
    )
    similar = []
    seen = set()
    if results['ids'] and results['ids'][0]:
        for i in range(len(results['ids'][0])):
            meta = results['metadatas'][0][i]
            dist = results['distances'][0][i]
            if (dist <= threshold and meta is not None
                    and 'path' in meta and meta['path'] not in seen):
                similar.append({"title": meta['title'], "path": meta['path'], "distance": dist})
                seen.add(meta['path'])
    return similar


def update_vector_db(collection, page_id: int, title: str, path: str, content: str):
    """페이지의 기존 청크를 삭제하고 새 내용으로 재색인한다.
    indexer.py와 동일한 chunker.chunk_text를 사용해 청크 일관성을 보장한다.
    """
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


# ── 메인 UI ──────────────────────────────────────────────────────────────────

st.set_page_config(page_title="CSU WIKI AI", layout="centered")

try:
    collection = load_vectordb()
    bm25_index = load_bm25_index()
except Exception as e:
    st.error(f"DB 로드 실패: {e}")
    st.stop()

app_mode = st.sidebar.radio(
    "모드 선택",
    ["Search AI", "PDF -> Wiki Data"],
    on_change=reset_generation_state,
)

# 사이드바 — 현재 로드된 모델 표시
with st.sidebar:
    try:
        model_id = _get_loaded_model_id()
        st.caption(f"🤖 모델: {model_id}")
    except Exception:
        pass

# 사이드바 — BM25 인덱스 갱신 시각
# empty() 플레이스홀더로 선언해 BM25 패치 완료 후 재호출 시 즉시 갱신
bm25_status_placeholder = st.sidebar.empty()
render_bm25_status(bm25_status_placeholder)


# ── Search AI 모드 ────────────────────────────────────────────────────────────

if app_mode == "Search AI":
    st.title("🏫 CSU wiki AI")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("질문하세요"):
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("assistant"):
            hits = hybrid_search(
                collection, bm25_index, prompt,
                top_n=2, candidates=20, expand_window=1,
            )

            if not hits:
                ctx = "검색된 관련 문서가 없습니다. 이전 대화 문맥을 참고하여 답변하세요."
                titles = set()
            else:
                ctx = "\n---\n".join(h["document"] for h in hits if h["document"].strip())
                titles = {h["metadata"].get("title", "제목 없음") for h in hits if h["metadata"]}

            ans = call_llm(st.session_state.messages, ctx)

            if titles:
                ans += "\n\n**[출처]**\n" + "\n".join([f"- {t}" for t in titles])

            st.markdown(ans)
            st.session_state.messages.append({"role": "assistant", "content": ans})


# ── PDF → Wiki Data 모드 ──────────────────────────────────────────────────────

elif app_mode == "PDF -> Wiki Data":
    st.title("📄 PDF -> Wiki Data")

    if 'generation_config' not in st.session_state:

        if 'pending_check' not in st.session_state:
            # ── 1단계: form — 데이터 수집 및 중복 검사 ──────────────────────
            with st.form("upload_form"):
                file = st.file_uploader("PDF 선택", type=["pdf"])
                dept = st.selectbox("부서", ["정보전산원", "교무처", "학생처", "기획처"])
                title = st.text_input("문서 제목")
                if st.form_submit_button("시작"):
                    if file and title:
                        safe_title = re.sub(r'[^\w가-힣-]', '',
                                            re.sub(r'[\s/]+', '-', title.strip()))
                        final_path = f"{dept}/{safe_title}"

                        with st.spinner("PDF 파싱 및 검사 중..."):
                            st.session_state.raw_text = wiki_builder.extract_text_from_pdf(
                                BytesIO(file.getvalue())
                            )
                            is_exists, _ = wiki_builder.check_page_exists(
                                WIKI_URL, WIKI_API_TOKEN, final_path
                            )
                            similar = search_similar_titles(collection, title)

                        st.session_state.pending_check = {
                            'is_exists': is_exists,
                            'similar': similar,
                            'title': title,
                            'final_path': final_path,
                        }
                        st.rerun()

        else:
            # ── 2단계: inline confirmation UI ───────────────────────────────
            pending = st.session_state.pending_check
            base_config = {
                'action': 'create',
                'path': pending['final_path'],
                'page_id': None,
                'title': pending['title'],
            }

            if pending['is_exists']:
                # 동일 경로 존재 — 덮어쓰기 / 취소
                st.error(f"동일 경로(`{pending['final_path']}`)가 이미 존재합니다.")
                st.write(f"- **{pending['title']}** ({pending['final_path']})")
                st.markdown("---")

                col1, col2 = st.columns(2)
                with col1:
                    if st.button("덮어쓰기 (Update)", type="primary",
                                 use_container_width=True, key="cf_overwrite_exact"):
                        base_config['action'] = 'update'
                        st.session_state.generation_config = base_config
                        st.session_state.pop('pending_check', None)
                        st.rerun()
                with col2:
                    if st.button("취소", use_container_width=True, key="cf_cancel_exact"):
                        reset_generation_state()
                        st.rerun()

            elif pending['similar']:
                # 유사 문서 발견 — 덮어쓰기 / 신규 생성 / 취소
                st.warning("⚠️ 유사한 문서가 발견되었습니다.")
                for doc in pending['similar']:
                    st.write(
                        f"- **{doc['title']}** ({doc['path']}) / "
                        f"유사도: {max(0, 1 - doc['distance']):.1%}"
                    )
                st.markdown("---")

                col1, col2, col3 = st.columns(3)
                with col1:
                    if st.button("덮어쓰기 (Update)", type="primary",
                                 use_container_width=True, key="cf_overwrite_sim"):
                        base_config['action'] = 'update'
                        base_config['path'] = pending['similar'][0]['path']
                        st.session_state.generation_config = base_config
                        st.session_state.pop('pending_check', None)
                        st.rerun()
                with col2:
                    if st.button("신규 생성", use_container_width=True, key="cf_create_sim"):
                        base_config['action'] = 'create'
                        st.session_state.generation_config = base_config
                        st.session_state.pop('pending_check', None)
                        st.rerun()
                with col3:
                    if st.button("취소", use_container_width=True, key="cf_cancel_sim"):
                        reset_generation_state()
                        st.rerun()

            else:
                # 중복 없음 — 바로 처리 단계로
                st.session_state.generation_config = base_config
                del st.session_state.pending_check
                st.rerun()

    else:
        # ── 3단계: Wiki.js 반영 및 RAG 인덱싱 ──────────────────────────────
        config = st.session_state.generation_config
        st.info(f"🚀 처리 중 (대상: `{config['path']}`)")
        
        try:
            refined_md = st.session_state.raw_text
            with st.expander("📄 추출된 마크다운 미리보기", expanded=False):
                st.markdown(refined_md)
            st.success(f"✅ 추출 완료 (길이: {len(refined_md):,}자)")

            with st.spinner("Wiki.js 전송 중..."):
                if config['action'] == 'update':
                    _, existing_id = wiki_builder.check_page_exists(
                        WIKI_URL, WIKI_API_TOKEN, config['path']
                    )
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
                cnt = update_vector_db(
                    collection, page_id, config['title'], config['path'], refined_md
                )
                st.success(f"✅ 인덱싱 완료 ({cnt}개 청크)")

            # BM25 인덱스 메모리 패치 (디스크 미반영 — 다음 indexer 배치에서 정식 반영)
            # _bm25_lock으로 보호 — 동시 업로드 시 race condition 방지
            try:
                new_chunk_ids = [f"page_{page_id}_chunk_{i}" for i in range(cnt)]
                with _bm25_lock:
                    if config['action'] == 'update':
                        old_chunk_ids = [
                            cid for cid in bm25_index.chunk_ids
                            if cid.startswith(f"page_{page_id}_chunk_")
                        ] if bm25_index else []
                        if old_chunk_ids:
                            bm25_store.patch_remove(bm25_index, old_chunk_ids)
                    if bm25_index is not None:
                        bm25_store.patch_add(bm25_index, new_chunk_ids, chunk_text(refined_md))
                        st.session_state["bm25_last_patch"] = datetime.datetime.now()
                        _save_patch_time()                      # ← 재시작 후에도 복원
                        render_bm25_status(bm25_status_placeholder)  # ← 사이드바 즉시 갱신
            except Exception as e:
                logger.warning(f"BM25 패치 실패 (다음 indexer 배치에서 복구됨): {e}")

            st.button("새로운 작업 시작", on_click=reset_generation_state, type="primary")

        except Exception as e:
            st.error(f"오류: {e}")
            st.button("초기화 및 돌아가기", on_click=reset_generation_state)