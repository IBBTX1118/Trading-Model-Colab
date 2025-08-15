# 檔名: 01_data_acquisition.py
# 描述: 從 MetaTrader 5 獲取、處理、驗證並儲存市場數據。
# 版本: 2.2 (實現結束日期自動化)

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timezone
import time
import sys
import os
import logging
from typing import List, Tuple, Dict, Any, Optional
from pathlib import Path
import concurrent.futures
from threading import Lock

from pydantic import BaseModel, Field, ValidationError
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

matplotlib.use("Agg")

# ==============================================================================
#                            1. Pydantic 配置模型
# ==============================================================================
class OutputSettings(BaseModel):
    log_level: str = "INFO"
    output_base_folder_name: str = "Output_Data_Pipeline_v2"
    market_data_subfolder: str = "MarketData"
    charts_subfolder: str = "Charts"
    save_cleaned_data: bool = True
    generate_charts: bool = True

class MT5ConnectionSettings(BaseModel):
    retry_attempts: int = 3
    retry_delay_seconds: int = 5
    initial_sync_delay_seconds: int = 3
    delay_between_requests_seconds: float = 0.2

class DataRequestSettings(BaseModel):
    start_year: int = Field(..., ge=1970, le=datetime.now().year)
    start_month: int = Field(..., ge=1, le=12)
    start_day: int = Field(..., ge=1, le=31)
    # ★★★ 關鍵修改：讓結束日期自動設為當前日期 ★★★
    end_year: int = Field(default_factory=lambda: datetime.now().year)
    end_month: int = Field(default_factory=lambda: datetime.now().month)
    end_day: int = Field(default_factory=lambda: datetime.now().day)
    symbol_list: List[str]
    timeframe_str_list: List[str]
    strict_mode: bool = False

class ValidationThresholds(BaseModel):
    price_jump_threshold_pct: float = 15.0
    intraday_extreme_range_std_dev_factor: float = 5.0
    rolling_window_for_volume_check: int = 20
    volume_std_dev_factor_threshold: float = 3.0

class PipelineConfig(BaseModel):
    output_settings: OutputSettings
    mt5_connection: MT5ConnectionSettings
    data_request: DataRequestSettings
    data_validation_thresholds: ValidationThresholds
    timezone_setting: str = "UTC"

# ==============================================================================
#                      2. 數據處理器類別 (DataProcessor)
# ==============================================================================
class DataProcessor:
    # ... 此類別無變動，內容與您提供的版本相同 ...
    def __init__(self, config: PipelineConfig, logger: logging.Logger):
        self.config = config; self.logger = logger
        self.strict_mode = self.config.data_request.strict_mode
    def process_raw_data_to_df(self, raw_rates, symbol, timeframe_str):
        if raw_rates is None or len(raw_rates) == 0: return None
        try:
            rates_df = pd.DataFrame(raw_rates)
            rates_df["time"] = pd.to_datetime(rates_df["time"], unit="s", utc=True)
            rates_df.set_index("time", inplace=True); return rates_df
        except Exception as e:
            self.logger.error(f"[{symbol}/{timeframe_str}] 處理數據轉換為 DataFrame 時出錯: {e}", exc_info=True); return None
    def clean_and_verify_df(self, rates_df, symbol, timeframe_str):
        if rates_df is None or rates_df.empty: return None
        df_cleaned = rates_df.copy(); initial_rows = len(df_cleaned)
        df_cleaned = self._verify_ohlc_integrity(df_cleaned, symbol, timeframe_str)
        if initial_rows != len(df_cleaned): self.logger.info(f"[{symbol}/{timeframe_str}] 清洗後數據筆數: {len(df_cleaned)} (原始: {initial_rows})")
        return df_cleaned
    def _verify_ohlc_integrity(self, df, symbol, timeframe_str):
        invalid_hl = df["high"] < df["low"]
        if invalid_hl.any():
            msg = f"[{symbol}/{timeframe_str}] 發現 {invalid_hl.sum()} 筆記錄 High < Low。"
            if self.strict_mode: self.logger.warning(f"{msg} [嚴格模式]：將移除這些記錄。"); df = df[~invalid_hl]
            else: self.logger.warning(msg)
        return df
    def perform_advanced_df_validation(self, df, symbol, timeframe_str): pass

# ==============================================================================
#                      3. 主流程控制類別 (MT5DataPipeline)
# ==============================================================================
class MT5DataPipeline:
    # ... 此類別無變動，內容與您提供的版本相同 ...
    def __init__(self, config: PipelineConfig):
        self.config = config; self.logger = self._setup_logger(); self.processor = DataProcessor(config, self.logger); self.mt5_lock = Lock()
        self.base_dir = Path(os.getcwd()) # 使用當前工作目錄
        self.output_base_dir = self.base_dir / self.config.output_settings.output_base_folder_name
        self.market_data_dir = self.output_base_dir / self.config.output_settings.market_data_subfolder
        self.charts_dir = self.output_base_dir / self.config.output_settings.charts_subfolder
        self._initialize_paths()
    def _setup_logger(self):
        logger = logging.getLogger(self.__class__.__name__); logger.setLevel(self.config.output_settings.log_level.upper())
        if logger.hasHandlers(): logger.handlers.clear()
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - [%(threadName)s] - %(message)s")
        sh = logging.StreamHandler(sys.stdout); sh.setFormatter(formatter); logger.addHandler(sh); return logger
    def _initialize_paths(self):
        try:
            self.output_base_dir.mkdir(exist_ok=True); self.market_data_dir.mkdir(exist_ok=True)
            if self.config.output_settings.generate_charts: self.charts_dir.mkdir(exist_ok=True)
            self.logger.info(f"輸出總目錄: {self.output_base_dir}")
        except OSError as e: self.logger.critical(f"創建輸出目錄失敗: {e}", exc_info=True); raise
    def _connect_mt5(self):
        with self.mt5_lock:
            if mt5.initialize(): self.logger.info("MetaTrader 5 初始化成功。"); return True
            else: self.logger.error(f"初始化 MetaTrader 5 失敗, 錯誤代碼 = {mt5.last_error()}"); return False
    def _disconnect_mt5(self):
        with self.mt5_lock: mt5.shutdown(); self.logger.info("MetaTrader 5 連接已關閉。")
    @staticmethod
    def _map_timeframe(tf_str): return getattr(mt5, f"TIMEFRAME_{tf_str.upper()}", None)
    def _fetch_raw_rates_chunked(self, symbol, timeframe_mt5, start_dt, end_dt):
        self.logger.info(f"[{symbol}] 開始分塊獲取 {start_dt.year} 至 {end_dt.year} 年的數據..."); all_rates = []
        for year in range(start_dt.year, end_dt.year + 1):
            chunk_start = datetime(year, 1, 1, tzinfo=timezone.utc); chunk_end = datetime(year + 1, 1, 1, tzinfo=timezone.utc)
            request_start = max(start_dt, chunk_start); request_end = min(end_dt, chunk_end)
            if request_start >= request_end: continue
            try:
                with self.mt5_lock: rates_chunk = mt5.copy_rates_range(symbol, timeframe_mt5, request_start, request_end)
                if rates_chunk is not None and len(rates_chunk) > 0: all_rates.append(rates_chunk)
                time.sleep(self.config.mt5_connection.delay_between_requests_seconds)
            except Exception as e: self.logger.error(f"[{symbol}] 獲取 {year} 年數據塊時出錯: {e}", exc_info=True)
        if not all_rates: self.logger.warning(f"[{symbol}] 在指定時間範圍內未獲取到任何數據。"); return None
        return np.concatenate(all_rates)
    def _save_df_to_parquet(self, df, symbol_folder, filename_base):
        try:
            output_path = symbol_folder / f"{filename_base}.parquet"; df.to_parquet(output_path, engine="pyarrow")
            self.logger.info(f"數據已成功保存到: {output_path}")
        except Exception as e: self.logger.error(f"保存 Parquet 文件時發生錯誤: {e}", exc_info=True)
    def _generate_and_save_chart(self, df, symbol_folder, filename_base):
        # ... 圖表生成邏輯無變動 ...
        pass
    def _process_task(self, symbol, timeframe_str):
        task_name = f"{symbol}/{timeframe_str}"
        self.logger.info(f"======== 開始處理任務: {task_name} ========")
        timeframe_mt5 = self._map_timeframe(timeframe_str)
        if timeframe_mt5 is None: return f"[{task_name}] 失敗：無法識別的時間週期。"
        sanitized_symbol_name = "".join([c if c.isalnum() else "_" for c in symbol])
        symbol_folder = self.market_data_dir / sanitized_symbol_name; symbol_folder.mkdir(exist_ok=True)
        filename_base = f"{sanitized_symbol_name}_{timeframe_str}"; output_path = symbol_folder / f"{filename_base}.parquet"
        req_conf = self.config.data_request
        start_dt = datetime(req_conf.start_year, req_conf.start_month, req_conf.start_day, tzinfo=timezone.utc)
        end_dt = datetime(req_conf.end_year, req_conf.end_month, req_conf.end_day, 23, 59, 59, tzinfo=timezone.utc)
        existing_df = None
        if output_path.exists():
            try:
                existing_df = pd.read_parquet(output_path)
                if not existing_df.empty: start_dt = existing_df.index[-1]
            except Exception as e: self.logger.warning(f"[{task_name}] 讀取現有檔案 {output_path} 失敗: {e}。將執行完整下載。"); existing_df = None
        if start_dt >= end_dt: return f"[{task_name}] 成功：數據已是最新，共 {len(existing_df)} 筆數據。"
        raw_rates = self._fetch_raw_rates_chunked(symbol, timeframe_mt5, start_dt, end_dt)
        newly_fetched_df = self.processor.process_raw_data_to_df(raw_rates, symbol, timeframe_str)
        if newly_fetched_df is None or newly_fetched_df.empty:
            if existing_df is not None: return f"[{task_name}] 成功：未獲取到新數據，維持 {len(existing_df)} 筆數據。"
            else: return f"[{task_name}] 失敗：未獲取到任何數據。"
        cleaned_new_df = self.processor.clean_and_verify_df(newly_fetched_df, symbol, timeframe_str)
        if existing_df is not None and cleaned_new_df is not None:
            final_df = pd.concat([existing_df, cleaned_new_df]); final_df = final_df[~final_df.index.duplicated(keep="first")]; final_df.sort_index(inplace=True)
        else: final_df = cleaned_new_df
        if final_df is None or final_df.empty: return f"[{task_name}] 失敗：數據清洗後無有效數據。"
        self.processor.perform_advanced_df_validation(final_df, symbol, timeframe_str)
        if self.config.output_settings.save_cleaned_data: self._save_df_to_parquet(final_df, symbol_folder, filename_base)
        if self.config.output_settings.generate_charts: self._generate_and_save_chart(final_df, symbol_folder, filename_base)
        return f"[{task_name}] 成功：處理後共 {len(final_df)} 筆數據。"
    def run_pipeline(self):
        main_start_time = time.time()
        self.logger.info(f"========= 數據獲取管道開始執行 (版本 2.2) =========")
        if not self._connect_mt5(): self.logger.critical("MT5 初始化失敗，程式終止。"); return
        tasks = [(symbol, tf) for symbol in self.config.data_request.symbol_list for tf in self.config.data_request.timeframe_str_list]
        self.logger.info(f"準備執行 {len(tasks)} 個任務..."); results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=5, thread_name_prefix="MT5_Worker") as executor:
            future_to_task = {executor.submit(self._process_task, symbol, tf): (symbol, tf) for symbol, tf in tasks}
            for future in concurrent.futures.as_completed(future_to_task):
                task = future_to_task[future]
                try: result_msg = future.result(); self.logger.info(result_msg); results.append(result_msg)
                except Exception as exc: error_msg = f"任務 {task} 產生例外: {exc}"; self.logger.error(error_msg, exc_info=True); results.append(error_msg)
        self._disconnect_mt5()
        self.logger.info(f"========= 所有任務執行完畢，總耗時: {time.time() - main_start_time:.2f} 秒 =========")

if __name__ == "__main__":
    # ... 主執行區塊無變動 ...
    config_dict = {
        "output_settings": {"log_level": "INFO", "output_base_folder_name": "Output_Data_Pipeline_v2", "market_data_subfolder": "MarketData", "charts_subfolder": "Charts", "save_cleaned_data": True, "generate_charts": False,},
        "mt5_connection": {"retry_attempts": 3, "retry_delay_seconds": 5, "initial_sync_delay_seconds": 3, "delay_between_requests_seconds": 0.1,},
        "data_request": {"start_year": 2021, "start_month": 1, "start_day": 1, "symbol_list": ["EURUSD.sml", "USDJPY.sml", "GBPUSD.sml", "AUDUSD.sml"], "timeframe_str_list": ["H1", "H4", "D1"], "strict_mode": True,},
        "data_validation_thresholds": {"price_jump_threshold_pct": 15.0, "intraday_extreme_range_std_dev_factor": 5.0, "rolling_window_for_volume_check": 20, "volume_std_dev_factor_threshold": 3.0,},
        "timezone_setting": "UTC",
    }
    try:
        pipeline_config = PipelineConfig(**config_dict); data_pipeline = MT5DataPipeline(config=pipeline_config); data_pipeline.run_pipeline()
    except ValidationError as e: print(f"配置錯誤: \n{e}"); sys.exit(1)
    except Exception as e: logging.critical(f"執行數據管道時發生未預期的嚴重錯誤: {e}", exc_info=True); sys.exit(1)
