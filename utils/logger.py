"""
日誌記錄工具模組。

提供全專案統一的 Logger 建立函式，
同時將日誌輸出到終端機（stdout）與檔案，
方便訓練過程中即時監控，也留有可查閱的紀錄檔。

所屬位置：Salary Predictive Analytics Engine/utils/logger.py
被以下模組引用：main.py、catboost_main.py、salary_server.py
"""

import logging
import sys
from pathlib import Path


def setup_logger(name: str = "salary_training", log_file: str | Path = "outputs/logs/training.log") -> logging.Logger:
    """
    建立同時輸出到終端機與檔案的 Logger。

    若同名 Logger 已存在，會先清除既有的 handlers，
    避免重複初始化時日誌被輸出多次。
    日誌目錄不存在時會自動建立。

    參數：
        name      - Logger 名稱，預設為 'salary_training'。
        log_file  - 日誌檔案路徑，預設為 'outputs/logs/training.log'。

    回傳：
        設定完成的 logging.Logger 物件。
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # 清除既有 handlers，避免重複呼叫時日誌被輸出多次。
    logger.handlers.clear()

    # 關閉向父 Logger 傳播，確保訊息只由本 Logger 處理。
    logger.propagate = False

    log_file = Path(log_file)
    # 確保日誌目錄存在，不存在時自動建立。
    log_file.parent.mkdir(parents=True, exist_ok=True)

    # 統一日誌格式：時間戳記 | 等級 | 訊息。
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # 終端機 Handler：輸出到 stdout，方便訓練時即時監控進度。
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)

    # 檔案 Handler：以 UTF-8 編碼寫入日誌檔，保留完整訓練紀錄。
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)

    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    return logger
