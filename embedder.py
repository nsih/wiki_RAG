# embedder.py
"""임베딩 함수 생성을 app.py / indexer.py / finetune 스크립트가 공유한다.

**왜 따로 뺐나** — 색인을 만든 모델과 검색할 때 쓰는 모델이 조금이라도 다르면
벡터 공간이 어긋나 거리 계산이 무의미해진다. 모델 이름뿐 아니라
`max_seq_length`(문서를 몇 토큰에서 자르는가)까지 같아야 한다.
파인튜닝 모델은 max_len=128로 학습·평가·색인했는데 SentenceTransformer 기본값은
512라, 그냥 두면 같은 모델인데도 다른 벡터가 나온다.

설정은 secrets.toml에서 읽는다(app.py, indexer.py 각자 로드):

    EMBED_MODEL       = "./models/ft-ep2"   # 또는 "jhgan/ko-sroberta-multitask"
    EMBED_MAX_SEQ_LEN = 128                 # 0 또는 미지정 = 모델 기본값
"""

import logging

from chromadb.utils import embedding_functions

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "jhgan/ko-sroberta-multitask"


def make_embedding_function(model_name: str = DEFAULT_MODEL, max_seq_len: int = 0):
    """ChromaDB 컬렉션에 심을 SentenceTransformer 임베딩 함수를 만든다.

    Args:
        model_name  : HF 모델 ID 또는 로컬 경로(예: "./models/ft-ep2")
        max_seq_len : 토큰 상한. 0이면 모델 기본값을 그대로 쓴다.

    파인튜닝 모델 디렉터리에는 sentence-transformers 설정(modules.json)이 없어
    ST가 Transformer + mean pooling 구성을 자동으로 만든다. 이는 학습·평가에 쓴
    수동 계산과 일치한다(finetune/reindex.py 주석 참조 — 코사인 1.000000 확인).
    """
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=model_name)

    if max_seq_len:
        # 내부 SentenceTransformer 인스턴스에 직접 꽂는다. ST/chromadb 버전에 따라
        # 접근 경로가 달라질 수 있어 실패해도 앱을 멈추지는 않되, 벡터가 색인과
        # 어긋날 수 있으므로 경고를 남긴다.
        model = getattr(ef, "_model", None) or getattr(ef, "models", {}).get(model_name)
        if model is not None:
            model.max_seq_length = max_seq_len
        else:
            logger.warning(
                f"max_seq_length={max_seq_len} 적용 실패 — 모델 기본값 사용. "
                f"색인과 다른 벡터가 나올 수 있다."
            )

    return ef
