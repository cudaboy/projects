import logging
import os
from datetime import datetime

def setup_logger():
    """
    로그 폴더를 생성하고, 파일 및 콘솔에 동시에 로그를 출력하도록 로거를 설정합니다.
    """
    LOG_DIR = "./logs"
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)

    log_filename = datetime.now().strftime("chatbot_%Y-%m-%d.log")
    log_path = os.path.join(LOG_DIR, log_filename)

    logger = logging.getLogger("ChatbotLogger")
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
        
        file_handler = logging.FileHandler(log_path, encoding='utf-8')
        file_handler.setFormatter(formatter)
        
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)
        
    return logger