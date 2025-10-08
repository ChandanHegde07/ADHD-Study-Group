import json
import os
import time

RATE_LIMIT_LOG_FILE = "rate_limit_log.json" 
REQUEST_LIMIT = 5
TIME_WINDOW = 60

def _load_rate_limit_log():
    if os.path.exists(RATE_LIMIT_LOG_FILE):
        try:
            with open(RATE_LIMIT_LOG_FILE, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return {}
    return {}

def _save_rate_limit_log(log_data):
    with open(RATE_LIMIT_LOG_FILE, "w") as f:
        json.dump(log_data, f)

def check_and_log_request(username: str) -> bool:
    log_data = _load_rate_limit_log()
    current_time = time.time()

    user_timestamps = log_data.get(username, [])

    recent_timestamps = [t for t in user_timestamps if current_time - t < TIME_WINDOW]
    
    if len(recent_timestamps) >= REQUEST_LIMIT:
        log_data[username] = recent_timestamps
        _save_rate_limit_log(log_data)
        return False  
    
    recent_timestamps.append(current_time)
    log_data[username] = recent_timestamps
    
    _save_rate_limit_log(log_data)
    
    return True 