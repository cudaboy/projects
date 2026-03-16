import torch
from transformers import BertTokenizerFast

# LangChain 관련 모듈 (파이프라인 조립용)
from langchain_core.language_models.llms import LLM
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# 파트너 패키지로 분리된 HuggingFace & Chroma DB 모듈
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# 직접 만든 트랜스포머 아키텍처 뼈대 불러오기
from model import Transformer
from utils import setup_logger

logger = setup_logger()

# =====================================================================
# 1. 전역 설정값 정의
# =====================================================================
# GPU가 있으면 사용하고, 맥북(MPS)이나 일반 PC(CPU) 환경에 맞춰 자동 설정합니다.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

# MLflow 연동을 빼고 도커 배포를 가볍게 하기 위해, 
# 실험에서 찾았던 최적의 하이퍼파라미터를 하드코딩(고정) 합니다.
BEST_DROPOUT = 0.0750 

def initialize_rag_system():
    """
    서버가 켜질 때 한 번만 실행되어 DB, 토크나이저, 모델, RAG 체인을 메모리에 올립니다.
    """
    # =====================================================================
    # 2. 벡터 DB 로드 및 검색기(Retriever) 설정
    # =====================================================================
    logger.info("📚 로컬 벡터 DB(Chroma)를 불러오는 중...")
    
    # 문장을 숫자로 바꿔주는 한국어 임베딩 모델을 가져옵니다.
    embeddings = HuggingFaceEmbeddings(model_name="jhgan/ko-sroberta-multitask")
    
    # 미리 만들어둔 'rag_db' 폴더에서 데이터를 읽어옵니다. (매번 새로 임베딩하지 않아 빠름)
    vectorstore = Chroma(persist_directory="./rag_db", embedding_function=embeddings)
    
    # 가장 유사한 문서 3개(k=3)를 찾아주는 검색기를 만듭니다.
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # =====================================================================
    # 3. 파이토치 모델(Transformer) 및 토크나이저 로드
    # =====================================================================
    logger.info("🤖 챗봇 뇌(가중치)를 이식하는 중...")
    
    # 허깅페이스에서 기준이 되는 한국어 사전을 가져옵니다.
    tokenizer = BertTokenizerFast.from_pretrained("klue/bert-base") 
    
    # 우리가 만든 Transformer 클래스에 옵션값을 넣어 빈 껍데기를 만듭니다.
    local_model = Transformer(
        vocab_size=tokenizer.vocab_size, 
        num_layers=2, 
        dff=512, 
        d_model=256, 
        num_heads=8, 
        dropout=BEST_DROPOUT
    ).to(DEVICE)
    
    # 파인튜닝으로 똑똑해진 뇌(가중치) 파일을 껍데기에 덮어씌우고 평가(eval) 모드로 바꿉니다.
    local_model.load_state_dict(torch.load("transformer_best.pth", map_location=DEVICE))
    local_model.eval()

    # =====================================================================
    # 4. LangChain Custom LLM 래핑 및 토크나이저 오류(##) 해결
    # =====================================================================
    # LangChain 파이프라인에 우리의 파이토치 모델을 끼워넣기 위해 포장(Wrap)합니다.
    class CustomKoreanTransformer(LLM):
        @property
        def _llm_type(self) -> str:
            return "custom_pytorch_transformer"

        def _call(self, prompt: str, stop=None) -> str:
            # 질문을 숫자로 바꿔서 모델에 넣어줍니다.
            input_ids = tokenizer.encode(prompt, return_tensors='pt').to(DEVICE)
            start_token = tokenizer.cls_token_id
            dec_inputs = torch.tensor([[start_token]]).to(DEVICE)
            
            # 💡 [핵심 해결책] 예측한 단어 번호(ID)들을 담아둘 빈 리스트
            answer_ids = [] 
            
            with torch.no_grad():
                for i in range(40): # 최대 40단어까지 예측
                    predictions = local_model(input_ids, dec_inputs)
                    next_word_id = predictions[:, -1, :].argmax(dim=-1).item()
                    
                    # 문장이 끝났다는 신호(SEP 토큰)를 받으면 멈춥니다.
                    if next_word_id == tokenizer.sep_token_id:
                        break
                        
                    # 💡 예측한 숫자(ID)를 리스트에 차곡차곡 모아둡니다.
                    answer_ids.append(next_word_id)
                    
                    # 다음 단어 예측을 위해 방금 예측한 단어를 입력에 이어붙입니다.
                    dec_inputs = torch.cat([dec_inputs, torch.tensor([[next_word_id]]).to(DEVICE)], dim=-1)
            
            # 💡 반복문이 끝나고 모아둔 숫자들을 한꺼번에 문자로 해독합니다!
            # skip_special_tokens=True 옵션이 '##' 기호를 알아서 예쁘게 지워줍니다.
            return tokenizer.decode(answer_ids, skip_special_tokens=True).strip()

    llm = CustomKoreanTransformer()

    # =====================================================================
    # 5. LangChain RAG 체인(Chain) 조립
    # =====================================================================
    # 챗봇에게 역할을 부여하는 프롬프트(지시문) 템플릿입니다.
    template = """주어진 문맥(Context)을 바탕으로 질문(Question)에 자연스럽게 답하세요.
    Context: {context}

    Question: {question}
    Answer:"""
    prompt_template = PromptTemplate.from_template(template)

    # 검색된 문서 여러 개를 하나의 긴 텍스트로 합쳐주는 함수입니다.
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # LCEL (LangChain Expression Language) 문법으로 조립 라인을 구축합니다.
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt_template
        | llm
        | StrOutputParser()
    )
    
    logger.info("✅ RAG 시스템 초기화 완료!")
    return rag_chain