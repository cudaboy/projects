# schemas.py : Pydantic을 사용하여 LLM의 출력 구조를 강제
# LLM이 항상 issue_target, key_categories, progressive, conservative, centrist라는 키를 가진 JSON 형태로
# 답변을 주도록 구조를 명확히 잡아주는 역할

from pydantic import BaseModel, Field
from typing import List

class PerspectiveDetail(BaseModel):
    """각 정치 성향별 세부 분석 결과를 담는 모델입니다."""
    core_tone: str = Field(
        description="해당 진영의 핵심 주장을 생방송 토론에서 발언하듯 강력하고 단호한 구어체로 작성한 1~2줄 요약 (예: '~해야 합니다!', '~는 어불성설입니다.')"
    )
    detailed_analysis: str = Field(
        description="기조를 바탕으로 한 구체적인 논거. 딱딱한 문어체가 아닌, 실제 패널이 토론하는 것처럼 확신에 찬 구어체(~습니다, ~합니다, ~아닙니까?)를 사용하여 청중을 설득하듯 작성할 것."
    )

class ComprehensiveAnalysis(BaseModel):
    """LLM이 최종적으로 반환해야 하는 전체 JSON 구조를 정의한 모델입니다."""
    issue_target: str = Field(description="분석 대상 이슈 (입력된 키워드 그대로)")
    key_categories: List[str] = Field(description="주요 연관 범주 리스트 (예: ['4. 복지', '3. 분배', '2. 경제'])")
    progressive: PerspectiveDetail = Field(description="진보 논객의 생방송 토론 발언")
    conservative: PerspectiveDetail = Field(description="보수 논객의 생방송 토론 발언")
    centrist: PerspectiveDetail = Field(description="중도 논객의 객관적이고 실용적인 분석 발언")