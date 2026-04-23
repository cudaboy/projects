"""
===============================================================================
[File Role]
이 파일(database.py)은 BioLinker 프로젝트의 '하이브리드 데이터베이스 구축 및 관리'를 담당합니다.

[상세 설명]
1. 목적: data_parser.py에서 정제된 문헌 텍스트와 관계 데이터를 각각 알맞은 DB에 적재하여,
   이후 에이전트(LangGraph)가 정확하고 풍부한 컨텍스트를 검색할 수 있는 기반을 마련합니다.
2. 기능 (Hybrid Search System):
   - Vector DB (Chroma): 논문 초록 텍스트를 OpenAI 임베딩으로 벡터화하여 저장. 
     -> 의미론적 유사도 기반의 유연한 문헌 검색(Semantic Search) 지원
   - Graph DB (NetworkX): 질병-유전자-약물 간의 관계를 노드(Node)와 엣지(Edge)로 구성된 지식 그래프로 구축.
     -> 개체 간의 명시적이고 팩트에 기반한 타겟 기전(MoA) 추적 지원
===============================================================================
"""

import os
import logging
import pandas as pd
import networkx as nx
from pathlib import Path
from tqdm import tqdm

# LangChain 기반 Vector DB 구축 도구
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

# config.py에서 전역 설정 불러오기
try:
    from biolinker import config
except ImportError:
    import config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class BioDatabaseManager:
    def __init__(self):
        """
        하이브리드 DB 매니저 초기화.
        환경 변수 및 경로가 정상적으로 설정되어 있는지 확인합니다.
        """
        # GPU 환경이 지원된다면 'cuda'로 변경하여 연산 속도를 대폭 향상시킬 수 있습니다.
        model_kwargs = {'device': 'cuda'} 
        encode_kwargs = {
            'normalize_embeddings': True, # 코사인 유사도 연산 최적화
            'batch_size': 4, # 기본값(32)에서 4 또는 8로 대폭 낮춰 GPU 메모리 부하를 방지
        }
        
        self.embeddings = HuggingFaceEmbeddings(
            model_name=config.EMBEDDING_MODEL,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )
        logging.info(f"✅ 임베딩 모델 로드 완료: {config.EMBEDDING_MODEL}")
        self.chroma_persist_dir = str(config.CHROMA_DB_DIR)
        self.knowledge_graph_path = config.KNOWLEDGE_GRAPH_PATH

    def build_vector_db(self, parsed_docs_csv: Path):
        """
        정제된 문헌 데이터를 읽어 임베딩하고 Chroma Vector DB에 적재합니다.
        """
        if not parsed_docs_csv.exists():
            logging.error(f"문헌 데이터가 없습니다: {parsed_docs_csv}")
            return

        logging.info("Vector DB(Chroma) 구축을 시작합니다...")
        df = pd.read_csv(parsed_docs_csv)
        
        # LangChain Document 객체 리스트 생성
        documents = []
        for _, row in df.iterrows():
            doc = Document(
                page_content=str(row.get('text', '')),
                metadata={
                    "doc_id": str(row.get('doc_id', 'unknown')),
                    "title": str(row.get('title', ''))
                }
            )
            documents.append(doc)
        
        # 빈 DB를 먼저 만든 뒤 잘게 쪼개어 넣으며 진행률을 표시
        vector_db = Chroma(
            embedding_function=self.embeddings,
            persist_directory=self.chroma_persist_dir,
            collection_name=config.CHROMA_COLLECTION_NAME
        )

        # 진행률 바 적용
        insert_batch_size = 10 # 10개씩 끊어서 처리
        
        for i in tqdm(range(0, len(documents), insert_batch_size), desc="임베딩 진행률", colour="green"):
            batch_docs = documents[i : i + insert_batch_size]
            vector_db.add_documents(batch_docs)

        logging.info(f"✅ Vector DB 구축 완료: 총 {len(documents)}개 문헌 임베딩됨. (경로: {self.chroma_persist_dir})")
        return vector_db

    def build_knowledge_graph(self, parsed_relations_csv: Path):
        """
        추출된 개체-관계 데이터를 바탕으로 NetworkX 기반의 지식 그래프(Knowledge Graph)를 구축합니다.
        """
        if not parsed_relations_csv.exists():
            logging.error(f"관계 데이터가 없습니다: {parsed_relations_csv}")
            return

        logging.info("Knowledge Graph 구축을 시작합니다...")
        df = pd.read_csv(parsed_relations_csv)
        
        # 방향성 그래프(DiGraph) 초기화
        G = nx.DiGraph()
        
        for _, row in df.iterrows():
            subj = str(row.get('subject_name'))
            obj = str(row.get('object_name'))
            rel_type = str(row.get('relation_type'))
            
            # 노드 속성 추가 (개체 타입)
            G.add_node(subj, entity_type=str(row.get('subject_type')))
            G.add_node(obj, entity_type=str(row.get('object_type')))
            
            # 엣지 속성 추가 (관계 타입 및 출처 논문 ID)
            G.add_edge(subj, obj, relation=rel_type, doc_id=str(row.get('doc_id')))

        # 구축된 그래프를 GML 포맷으로 로컬에 저장 (이후 Streamlit 시각화 등에 활용)
        nx.write_gml(G, self.knowledge_graph_path)
        logging.info(f"✅ Knowledge Graph 구축 완료: {G.number_of_nodes()}개 노드, {G.number_of_edges()}개 엣지 생성. (저장: {self.knowledge_graph_path})")
        return G

    def get_vector_retriever(self):
        """
        에이전트(LangGraph)에서 문헌 검색을 위해 호출할 Retriever 인터페이스를 반환합니다.
        """
        vector_db = Chroma(
            persist_directory=self.chroma_persist_dir,
            embedding_function=self.embeddings,
            collection_name=config.CHROMA_COLLECTION_NAME
        )
        # config에 설정된 K값 만큼 유사도 높은 문서를 반환하도록 설정
        return vector_db.as_retriever(search_kwargs={"k": config.RETRIEVER_K})

    def load_knowledge_graph(self):
        """
        저장된 지식 그래프를 메모리에 로드하여 에이전트의 Graph 탐색용으로 반환합니다.
        """
        if self.knowledge_graph_path.exists():
            return nx.read_gml(self.knowledge_graph_path)
        else:
            logging.warning("저장된 지식 그래프가 없습니다. 새로 구축이 필요합니다.")
            return nx.DiGraph()

# 단독 실행 시 테스트 로직 (데이터 구축용)
if __name__ == "__main__":
    manager = BioDatabaseManager()
    
    # data_parser.py가 생성한 CSV 경로를 예상하여 입력
    docs_csv_path = config.PROCESSED_DATA_DIR / "parsed_documents.csv"
    relations_csv_path = config.PARSED_CSV_PATH
    
    # 1. Vector DB 구축
    manager.build_vector_db(docs_csv_path)
    
    # 2. Knowledge Graph 구축
    manager.build_knowledge_graph(relations_csv_path)