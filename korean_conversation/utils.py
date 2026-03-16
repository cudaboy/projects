import logging
import os
from datetime import datetime

def setup_logger():
    """
    로그 폴더를 생성하고, 파일 및 콘솔(터미널 화면)에 동시에 로그를 출력하도록 로거(Logger)를 설정합니다.
    """
    # =====================================================================
    # 1. 로그 저장 폴더 생성
    # =====================================================================
    LOG_DIR = "./logs"
    # 만약 현재 프로젝트 폴더 안에 'logs'라는 폴더가 없다면 새로 만듭니다.
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)

    # =====================================================================
    # 2. 날짜별 로그 파일 이름 지정
    # =====================================================================
    # 오늘 날짜를 기준으로 파일 이름을 만듭니다. (예: chatbot_2026-03-16.log)
    # 날짜가 바뀌면 새로운 파일이 알아서 생성됩니다.
    log_filename = datetime.now().strftime("chatbot_%Y-%m-%d.log")
    log_path = os.path.join(LOG_DIR, log_filename)

    # =====================================================================
    # 3. 로거(Logger) 객체 생성 및 중복 방지
    # =====================================================================
    # "ChatbotLogger"라는 이름표를 가진 로거를 가져옵니다.
    logger = logging.getLogger("ChatbotLogger")
    
    # 로거가 여러 번 호출되어 로그가 2줄, 3줄씩 중복으로 찍히는 것을 방지합니다.
    if not logger.handlers:
        # INFO 수준 이상의 정보(INFO, WARNING, ERROR 등)만 기록하도록 설정합니다.
        logger.setLevel(logging.INFO)
        
        # 로그가 찍히는 양식을 예쁘게 지정합니다. (예: 2026-03-16 10:00:00,000 [INFO] 서버 시작됨)
        formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
        
        # -----------------------------------------------------------------
        # 3-1. 파일 핸들러 (File Handler)
        # -----------------------------------------------------------------
        # 한글이 깨지지 않도록 encoding='utf-8'을 지정하여 텍스트 파일에 기록합니다.
        file_handler = logging.FileHandler(log_path, encoding='utf-8')
        file_handler.setFormatter(formatter)
        
        # -----------------------------------------------------------------
        # 3-2. 스트림 핸들러 (Stream Handler)
        # -----------------------------------------------------------------
        # 터미널(콘솔) 화면에도 실시간으로 로그를 띄워줍니다.
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        
        # -----------------------------------------------------------------
        # 3-3. 로거에 핸들러 부착
        # -----------------------------------------------------------------
        # 만든 두 개의 핸들러를 로거에 달아줍니다. 이제 한 번만 로깅하면 파일과 화면 양쪽에 동시에 써집니다.
        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)
        
    # 설정이 완료된 로거 객체를 반환합니다.
    return logger