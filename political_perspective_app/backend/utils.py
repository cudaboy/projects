# utils.py : 유저의 질문과 AI의 답변을 로컬 JSONL 파일에 기록

from database import SessionLocal
from models import UserQuery, AnalysisResult

def save_to_history(question: str, analysis_result: dict):
    """
    유저의 질문과 AI의 심층 분석 결과를 SQLite 데이터베이스에 안전하게 저장합니다.
    """
    # DB 세션 생성
    db = SessionLocal()
    
    try:
        # 1. UserQuery(질문) 레코드 생성 및 저장
        db_query = UserQuery(keyword=question)
        db.add(db_query)
        db.commit()
        db.refresh(db_query) # 데이터베이스에서 자동 생성된 id값을 가져오기 위해 새로고침

        # 2. AnalysisResult(분석 결과) 레코드 생성
        # 리스트 형태인 카테고리를 쉼표로 구분된 문자열로 변환 (예: "사회, 경제")
        categories_str = ", ".join(analysis_result.get('key_categories', []))
        
        # 앞서 저장한 질문의 id(query_id)를 외래키로 사용하여 결과 저장
        db_result = AnalysisResult(
            query_id=db_query.id,
            categories=categories_str,
            
            # 진보 진영 데이터
            prog_tone=analysis_result['progressive']['core_tone'],
            prog_analysis=analysis_result['progressive']['detailed_analysis'],
            
            # 중도 진영 데이터
            cent_tone=analysis_result['centrist']['core_tone'],
            cent_analysis=analysis_result['centrist']['detailed_analysis'],
            
            # 보수 진영 데이터
            cons_tone=analysis_result['conservative']['core_tone'],
            cons_analysis=analysis_result['conservative']['detailed_analysis']
        )
        
        db.add(db_result)
        db.commit()
        print(f"✅ DB 저장 완료: '{question}'")
        
    except Exception as e:
        # 에러 발생 시 변경사항 롤백(취소)
        db.rollback()
        print(f"🚨 DB 저장 중 오류 발생: {e}")
        
    finally:
        # 작업이 끝나면 항상 세션 종료
        db.close()