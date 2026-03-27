import json
import httpx
import pandas as pd
import io
import os
import re
from bs4 import BeautifulSoup, SoupStrainer
from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager

# FinanceDataReader 임포트 (종목 코드 검색용)
import FinanceDataReader as fdr

# LangChain 1.x+ 에 맞춘 명시적 임포트
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.tools import tool

# 🌟 DuckDuckGo 검색 라이브러리 추가
from langchain_community.tools import DuckDuckGoSearchRun

# ==========================================
# 0. 종목 리스트 사전 로드 (서버 기동 시 1회 실행)
# ==========================================
try:
    krx_listing = fdr.StockListing('KRX')
    print("✅ KRX 종목 리스트 로드 완료")
except Exception as e:
    print(f"❌ 종목 리스트 로드 실패: {e}")
    krx_listing = pd.DataFrame()


# ==========================================
# 1. 종목 코드 검색 도구 (FinanceDataReader 기반 - 정밀도 향상)
# ==========================================
@tool
def get_code(company_name: str) -> str:
    """
    FinanceDataReader를 활용하여 회사 이름에 해당하는 6자리 주식 종목 코드를 반환한다.
    반드시 6자리 숫자로 된 코드값만 결과로 제공하여 다른 도구들이 즉시 사용할 수 있게 한다.
    """
    if krx_listing.empty:
        return "오류: 서버에 종목 리스트가 로드되지 않았습니다."

    try:
        exact_match = krx_listing[krx_listing['Name'] == company_name]
        
        if not exact_match.empty:
            code = exact_match.iloc[0]['Code']
        else:
            partial_match = krx_listing[krx_listing['Name'].str.contains(company_name, na=False)]
            if not partial_match.empty:
                code = partial_match.iloc[0]['Code']
            else:
                return f"'{company_name}'에 해당하는 종목 코드를 찾을 수 없습니다."
        
        if re.match(r'^\d{6}$', str(code)):
            return str(code)
        else:
            return f"'{company_name}'의 코드를 찾았으나 형식({code})이 올바르지 않습니다."
            
    except Exception as e:
        return f"종목 코드 검색 중 오류 발생: {str(e)}"

# ==========================================
# 2. 재무제표 크롤링 도구 (FnGuide 기반 - 속도/안정성 최상)
# ==========================================
@tool
def finance_report(company_code: str) -> str:
    """
    6자리 종목 코드를 입력받아 에프앤가이드(FnGuide)에서 
    기업의 주요 재무제표(Financial Highlight) 데이터를 가져온다.
    """
    if not re.match(r'^\d{6}$', str(company_code)):
        return f"잘못된 종목 코드 형식입니다: {company_code}. 6자리 숫자를 입력하세요."

    try:
        fnguide_url = f"https://comp.fnguide.com/SVO2/ASP/SVD_Main.asp?pGB=1&gicode=A{company_code}&cID=&MenuYn=Y&ReportGB=&NewMenuID=101&stkGb=701"
        
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        
        response = httpx.get(fnguide_url, headers=headers, timeout=10.0)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        table_div = soup.find('div', id='highlight_D_A')
        if not table_div:
            return "재무제표(Financial Highlight) 데이터를 찾을 수 없습니다."
            
        html_table = table_div.find('table')
        
        df_list = pd.read_html(io.StringIO(str(html_table)))
        if not df_list:
            return "테이블 파싱에 실패했습니다."
            
        df = df_list[0]
        
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = ['_'.join(str(c) for c in col if c and 'Unnamed' not in str(c)).strip() for col in df.columns]
        
        df = df.fillna("")
        return df.to_json(orient="records", force_ascii=False)
        
    except Exception as e:
        return f"FnGuide 재무 데이터 수집 중 오류 발생: {str(e)}"

# ==========================================
# 3. 최신 뉴스 검색 도구 (Analyst 에이전트용 - 네이버 + DuckDuckGo Fallback)
# ==========================================
@tool
def get_news(company_name: str) -> str:
    """
    회사명으로 최신 뉴스를 검색하여 기사 본문 내용이나 요약본을 하나의 텍스트로 병합하여 반환한다.
    """
    # 환경 변수에서 네이버 키를 가져옵니다. (graph.py에서 주입 예정)
    client_id = os.getenv("NAVER_CLIENT_ID")
    client_secret = os.getenv("NAVER_CLIENT_SECRET")
    
    # 🌟 1. 네이버 API 키가 유효한 경우 -> 기존 네이버 뉴스 크롤링 로직 실행
    if client_id and client_secret:
        url = "https://openapi.naver.com/v1/search/news.json"
        headers = {
            "X-Naver-Client-Id": client_id,
            "X-Naver-Client-Secret": client_secret
        }
        params = {'query': company_name, 'display': 10, 'start': 1, "sort": "date"} 
        
        try:
            response = httpx.get(url, headers=headers, params=params)
            response.raise_for_status()
            
            items = response.json().get('items', [])
            total_naver_url = [item['link'] for item in items if 'naver.com' in item['link']]
            
            bs4_kwargs = {'parse_only': SoupStrainer("div", id="newsct_article")}
            news_contents = []
            
            # 1단계: 네이버 포털 기사 원문 크롤링 시도
            for link in total_naver_url:
                try:
                    loader = WebBaseLoader(link, bs_kwargs=bs4_kwargs)
                    docs = loader.load()
                    if docs:
                        news_contents.append(docs[0].page_content.strip())
                except Exception:
                    continue 
            
            # 🌟 2단계(추가됨): 크롤링된 본문이 없다면 API가 제공한 요약본(description) 긁어오기
            if not news_contents:
                for item in items:
                    # HTML 태그(<b> 등)를 깔끔하게 제거
                    clean_text = re.sub(r'<[^>]+>', '', item.get('description', ''))
                    if clean_text:
                        news_contents.append(clean_text)
                    
            if news_contents:
                return " ".join(news_contents)
            else:
                return f"{company_name}에 대한 최근 뉴스 검색 결과가 없습니다."
                
        except Exception as e:
            return f"네이버 뉴스 데이터를 가져오는 중 오류 발생: {str(e)}"
            
    # 🌟 2. 네이버 API 키가 없는 경우 -> Fallback으로 DuckDuckGo 무료 검색 엔진 사용
    else:
        try:
            search = DuckDuckGoSearchRun()
            # DuckDuckGo는 본문 전체 크롤링 대신 검색 결과 요약(snippet)들을 제공합니다.
            query = f"{company_name} 주식 최근 경제 뉴스 핵심 요약"
            result = search.invoke(query)
            return f"[DuckDuckGo 검색 결과]\n{result}"
        except Exception as e:
            return f"무료 검색(DuckDuckGo) 중 오류 발생: {str(e)}"

# ==========================================
# 4. 주가 차트 데이터 API 호출 도구 (Trader 에이전트용)
# ==========================================
@tool
def get_data(company_code: str, sdate: str, edate: str) -> str:
    """
    종목 코드와 날짜를 입력받아 시가, 고가, 저가, 종가, 거래량 등의 과거 차트 데이터를 가져온다.
    (sdate/edate 형식: YYYYMMDD)
    """
    if not re.match(r'^\d{6}$', str(company_code)):
        return f"잘못된 종목 코드입니다: {company_code}"

    stock_url = f"https://m.stock.naver.com/front-api/external/chart/domestic/info?symbol={company_code}&requestType=1&startTime={sdate}&endTime={edate}&timeframe=day"
    try:
        response = httpx.get(stock_url)
        return response.text.strip()
    except Exception as e:
        return f"주가 데이터를 가져오는 중 오류 발생: {str(e)}"