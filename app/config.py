from __future__ import annotations

from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
META_DATA_BACKUP_DIR = ROOT_DIR / "meta_data_backup"
FUND_REPORT_DATA_FILE = DATA_DIR / "fund_report_data.json"
FUND_SIGNALS_FILE = DATA_DIR / "signals.json"
STOCK_RESULT_FILE = DATA_DIR / "stock_advanced_strategy_results.json"
STOCK_OPTIMIZED_CONFIG_FILE = DATA_DIR / "stock_strategy_optimized_config.json"
STOCK_OPTIMIZED_CONFIG_BACKUP_FILE = META_DATA_BACKUP_DIR / "stock_strategy_optimized_config.json"


class AppConfig:
    ROOT_DIR = ROOT_DIR
    DATA_DIR = DATA_DIR
    FUND_REPORT_DATA_FILE = FUND_REPORT_DATA_FILE
    FUND_SIGNALS_FILE = FUND_SIGNALS_FILE
    STOCK_RESULT_FILE = STOCK_RESULT_FILE
    STOCK_OPTIMIZED_CONFIG_FILE = STOCK_OPTIMIZED_CONFIG_FILE
    STOCK_OPTIMIZED_CONFIG_BACKUP_FILE = STOCK_OPTIMIZED_CONFIG_BACKUP_FILE
