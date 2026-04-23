"""
===============================================================================
[File Role]
이 파일(agents.py)은 BioLinker 프로젝트의 '멀티에이전트 두뇌(LLM Nodes)' 역할을 담당합니다.

[상세 설명]
1. 목적: LangGraph 워크플로우(workflow.py)에서 호출될 개별 에이전트들의 핵심 로직(검색, 추론, 합성)을 정의합니다.
2. 주요 에이전트(Nodes):
   - Router Agent: 사용자의 질문 의도를 분석하여 Vector DB를 뒤질지, Graph DB를 뒤질지, 혹은 둘 다 필요한지 분기합니다.
   - Vector Retrieval Agent: database.py의 Vector DB를 활용해 관련 논문 텍스트(Context)를 검색합니다.
   - Graph Retrieval Agent: 질문에서 핵심 개체(Entity)를 추출한 뒤, Graph DB에서 기전(MoA) 연결 경로를 탐색합니다.
   - Synthesizer Agent: 검색된 하이브리드 데이터를 종합하여, 임상 연구원에게 제공할 최종 답변과 근거를 합성합니다.
===============================================================================
"""

import logging
from typing import List, Dict, Any, Tuple
import networkx as nx

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser

# 전역 설정 및 데이터베이스 매니저 호출
try:
    from biolinker import config
    from biolinker.database import BioDatabaseManager
except ImportError:
    import config
    from database import BioDatabaseManager

class BioAgentManager:
    def __init__(self, db_manager: BioDatabaseManager) -> None:
        """
        에이전트 구동에 필요한 LLM과 데이터베이스 연결을 초기화합니다.
        """
        self.db_manager = db_manager
        # LLM 객체는 api.py에서 사용자 인증 후 동적으로 주입될 수 있도록 None으로 초기화
        self.llm: Any = None 
        
        # 하이브리드 검색기(Retriever) 로드
        self.vector_retriever = self.db_manager.get_vector_retriever()
        self.knowledge_graph: nx.DiGraph = self.db_manager.load_knowledge_graph()

    # ---------------------------------------------------------
    # 1. Router Agent (라우터 에이전트)
    # ---------------------------------------------------------
    def route_query(self, question: str) -> str:
        """
        사용자의 질문을 분석하여 탐색할 데이터베이스 경로를 결정합니다.
        (vector, graph, both, irrelevant 중 하나를 반환)
        """
        if not self.llm:
            raise ValueError("LLM이 초기화되지 않았습니다. api.py에서 주입되었는지 확인하세요.")

        prompt = ChatPromptTemplate.from_messages([
            ("system", 
             "당신은 제약 R&D 임상이행 파트의 인텐트 라우터(Intent Router)입니다.\n"
             "사용자의 질문이 논문 문헌 검색이 필요한지, 질병-유전자-약물 간의 관계(Graph) 탐색이 필요한지, 혹은 둘 다 필요한지 판단하세요.\n"
             "결과는 반드시 'vector', 'graph', 'both', 'irrelevant' 중 하나의 단어만 소문자로 출력하세요.\n"
             "💡 주의: 가상의 약물이나 신약 후보 물질(예: FakeDrug)에 대한 임상적/가설적 질문도 반드시 의학 질문으로 간주하여 'vector'나 'both'로 분류하세요. 절대 'irrelevant'로 차단하지 마세요.\n"
             "의학, 약학, 생물학과 전혀 무관한 일상 대화일 때만 'irrelevant'를 출력하세요."
            ),
            ("user", f"질문: {question}")
        ])
        
        chain = prompt | self.llm | StrOutputParser()
        decision: str = chain.invoke({}).strip().lower()
        
        if decision not in ['vector', 'graph', 'both', 'irrelevant']:
            return 'both'
        return decision

    # ---------------------------------------------------------
    # 2. Vector DB Retriever Agent
    # ---------------------------------------------------------
    def retrieve_vector_context(self, question: str) -> List[Any]:
        """
        의학 특화 임베딩을 통해 질문과 의미론적으로 가장 유사한 논문 초록을 검색합니다.
        """
        try:
            docs = self.vector_retriever.invoke(question)
            return docs
        except Exception as e:
            logging.error(f"Vector DB 검색 오류: {e}")
            return []

    # ---------------------------------------------------------
    # 3. Simple Graph Retriever Agent (멀티홉 제거 및 매칭 고도화 버전)
    # ---------------------------------------------------------
    def retrieve_graph_context(self, question: str) -> Tuple[str, List[str]]:
        """
        Knowledge Graph에서 질문 텍스트와 매칭되는 단일 노드의 직접 연결(1-hop) 정보만 단순 추출합니다.
        """
        current_logs = []
        
        if not self.knowledge_graph or len(self.knowledge_graph.nodes) == 0:
            current_logs.append("⚠️ [Graph DB] 그래프 데이터베이스가 비어있어 탐색을 건너뜁니다.")
            return "그래프 데이터베이스가 비어있습니다.", current_logs

        question_lower = question.lower()
        raw_matched_nodes = []
        
        # [Step 1] 질문 내에 그래프 노드 이름이 문자열로 포함되어 있는지 1차 매칭
        for node in self.knowledge_graph.nodes():
            node_str = str(node).lower()
            if len(node_str) > 1 and node_str in question_lower:
                raw_matched_nodes.append(str(node))
                
        # [Step 2] 노이즈 제거 필터링 (예: 'BRCA1'이 매칭되면 종속 단어인 'RC', 'CA' 등은 탈락시킴)
        matched_nodes = set()
        for n in raw_matched_nodes:
            is_substring = False
            for other in raw_matched_nodes:
                if n != other and n.lower() in other.lower():
                    is_substring = True
                    break
            if not is_substring:
                matched_nodes.add(n)

        # 매칭된 결과가 없을 경우
        if not matched_nodes:
            current_logs.append("⚠️ [Graph DB] 질문과 일치하는 명시적 의료 keyword(약물/질환 등)를 찾지 못했습니다.")
            return "질문과 직접적으로 연관된 그래프 관계 정보가 없습니다.", current_logs

        # 매칭 성공 로그 (사용자 친화적)
        current_logs.append(f"📍 [Graph DB] 질문에서 핵심 keyword 검색 완료: {', '.join(list(matched_nodes))}")

        found_edges: List[str] = []
        preview_edges: List[str] = []
        
        # [Step 3] 1-hop 직접 연결된 이웃 노드만 검색
        for node in matched_nodes:
            for neighbor in self.knowledge_graph.successors(node):
                edge_data: Dict[str, Any] = self.knowledge_graph.get_edge_data(node, neighbor)
                relation: str = edge_data.get('relation', '연관됨')
                doc_id: str = edge_data.get('doc_id', '출처미상')
                
                # 합성용 데이터 저장
                found_edges.append(f"[{node}] --({relation})--> [{neighbor}] (출처 논문 ID: {doc_id})")
                
                # UI 로그 미리보기용 데이터 저장 (최대 2개까지만 표출)
                if len(preview_edges) < 2:
                    preview_edges.append(f"[{node}] → [{neighbor}]")

        unique_edges = list(set(found_edges))
        
        if unique_edges:
            log_msg = f"🕸️ [Graph DB] 지식 그래프 탐색 성공: keyword 간 연관된 {len(unique_edges)}개의 관계 정보를 추출했습니다."
            if preview_edges:
                log_msg += f" (예시: {', '.join(preview_edges)} ...)"
            current_logs.append(log_msg)
            
            return "\n".join(sorted(unique_edges)), current_logs
        else:
            current_logs.append(f"⚠️ [Graph DB] '{', '.join(list(matched_nodes))}' keyword는 찾았으나, 연결된 관계 정보가 없습니다.")
            return "keyword 간 연관 관계가 그래프에 존재하지 않습니다.", current_logs

    # ---------------------------------------------------------
    # 4. Synthesizer Agent (종합 및 답변 생성 에이전트)
    # ---------------------------------------------------------
    def synthesize_answer(self, question: str, vector_docs: List[Any], graph_context: str) -> str:
        """
        검색된 하이브리드 컨텍스트(Vector 논문 + Graph 관계망)를 종합하여 
        연구원 맞춤형 최종 답변 리포트를 생성합니다.
        """
        if not self.llm:
            raise ValueError("LLM이 초기화되지 않았습니다.")

        # [안전 장치 완화] 두 DB 모두 비어있을 때만 에러 문구 반환 (하나라도 있으면 답변 시도)
        is_vector_empty = not vector_docs
        is_graph_empty = (not graph_context or "없습니다" in graph_context or "비어있습니다" in graph_context)
        
        if is_vector_empty and is_graph_empty:
            return "❌ 질문과 관련된 임상 논문 근거 및 그래프 데이터를 당사의 데이터베이스에서 찾을 수 없습니다. (데이터 보유 범위 초과)"

        # 벡터 문서 내용 합치기 및 출처 포맷팅
        formatted_docs: List[str] = []
        for i, doc in enumerate(vector_docs, 1):
            title: str = doc.metadata.get('title', '제목 없음')
            doc_id: str = doc.metadata.get('doc_id', '알 수 없는 ID')
            formatted_docs.append(f"[문헌 {i}]\n- 제목: {title}\n- 출처 ID: {doc_id}\n- 내용: {doc.page_content}")
        
        vector_text: str = "\n\n".join(formatted_docs)
        if not vector_text.strip():
            vector_text = "Vector DB에서 검색된 문헌 정보가 없습니다."

        # 프롬프트 구성
        system_prompt = """
        당신은 데이터 기반 임상 연구원입니다.
        주어진 [Vector DB 검색 결과]와 [Graph DB 개체 관계] 데이터를 종합하여 사용자의 질문에 전문적이고 논리적인 리포트 형태로 답변하세요.
        
        [필수 지시사항 - 할루시네이션 방지 가이드레일]
        1. 제공된 데이터 외의 외부 지식을 활용하여 허위 내용을 지어내지 마세요.
        2. 답변의 근거를 제시할 때는 '문헌 1에서'와 같이 애매하게 표현하지 말고, 제공된 문헌의 **[논문 제목]**이나 **[출처 ID]**를 괄호로 명시하여 신뢰성을 증명하세요.
        3. [가장 중요] 검색된 [Vector DB] 문헌이나 [Graph DB] 정보 내용 중에 사용자가 질문한 **핵심 질환명이나 약물명이 전혀 포함되어 있지 않다면**, 절대 모델이 가진 사전 지식으로 답변을 지어내지 마세요. 이 경우 무조건 "근거 데이터가 부족합니다. 제공된 문헌과 데이터베이스 검색 결과에서는 해당 정보를 찾을 수 없습니다." 라고만 답변하세요.
        4. Graph DB에서 제공된 1-hop 기전 정보들을 활용하여 약물과 질환 간의 인과 흐름(예: A -> B)을 논리적으로 연결하여 설명하세요.
        """

        user_prompt = f"""
        [사용자 질문]: {question}

        ---
        [Vector DB 검색 결과 (논문 문헌)]
        {vector_text}

        ---
        [Graph DB 개체 관계 검색 결과]
        {graph_context}
        """

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("user", user_prompt)
        ])
        
        chain = prompt | self.llm | StrOutputParser()
        try:
            final_answer: str = chain.invoke({})
            return final_answer
        except Exception as e:
            logging.error(f"Synthesizer 처리 중 오류 발생: {e}")
            return "응답을 합성하는 도중 오류가 발생했습니다."