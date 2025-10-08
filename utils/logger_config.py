import logging
import sys

def setup_logger():
    logger = logging.getLogger()
    
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        
        file_handler = logging.FileHandler("app.log")
        file_handler.setLevel(logging.INFO)
        
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        
        log_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(log_format)
        console_handler.setFormatter(log_format)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)