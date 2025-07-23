# 檔名: 01_data_acquisition.py
# 描述: 從 MetaTrader 5 獲取、處理、驗證並儲存市場數據。
#       包含增量更新機制，只下載最新的數據以提升效率。
# 版本: 2.1 (實現增量更新)

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

# Pydantic 用於配置模型驗證
from pydantic import BaseModel, Field, ValidationError

# Matplotlib 和 Seaborn 用於圖表生成
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

# 設定 Matplotlib 使用無 GUI 的後端，以便在伺服器上運行
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
    """
    專門負責數據的處理、清洗和驗證邏輯。
    """

    def __init__(self, config: PipelineConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.strict_mode = self.config.data_request.strict_mode

    def process_raw_data_to_df(
        self, raw_rates: Optional[np.ndarray], symbol: str, timeframe_str: str
    ) -> Optional[pd.DataFrame]:
        if raw_rates is None or len(raw_rates) == 0:
            self.logger.warning(
                f"[{symbol}/{timeframe_str}] 傳入的原始數據為空，無法處理。"
            )
            return None
        try:
            rates_df = pd.DataFrame(raw_rates)
            # 將 time 從 epoch seconds 轉為 UTC datetime 物件
            rates_df["time"] = pd.to_datetime(rates_df["time"], unit="s", utc=True)
            rates_df.set_index("time", inplace=True)
            return rates_df
        except Exception as e:
            self.logger.error(
                f"[{symbol}/{timeframe_str}] 處理數據轉換為 DataFrame 時出錯: {e}",
                exc_info=True,
            )
            return None

    def clean_and_verify_df(
        self, rates_df: Optional[pd.DataFrame], symbol: str, timeframe_str: str
    ) -> Optional[pd.DataFrame]:
        if rates_df is None or rates_df.empty:
            self.logger.warning(
                f"[{symbol}/{timeframe_str}] 傳入 clean_and_verify_df 的數據為空。"
            )
            return None

        self.logger.info(f"--- [{symbol}/{timeframe_str}] 開始數據清洗與基礎驗證 ---")
        df_cleaned = rates_df.copy()
        initial_rows = len(df_cleaned)

        df_cleaned = self._verify_ohlc_integrity(df_cleaned, symbol, timeframe_str)

        rows_after_cleaning = len(df_cleaned)
        if initial_rows != rows_after_cleaning:
            self.logger.info(
                f"[{symbol}/{timeframe_str}] 清洗後數據筆數: {rows_after_cleaning} (原始: {initial_rows})"
            )
        else:
            self.logger.info(f"[{symbol}/{timeframe_str}] 基礎數據清洗完成。")

        return df_cleaned

    def _verify_ohlc_integrity(
        self, df: pd.DataFrame, symbol: str, timeframe_str: str
    ) -> pd.DataFrame:
        invalid_hl = df["high"] < df["low"]
        if invalid_hl.any():
            msg = f"[{symbol}/{timeframe_str}] 發現 {invalid_hl.sum()} 筆記錄 High < Low。"
            if self.strict_mode:
                self.logger.warning(f"{msg} [嚴格模式]：將移除這些記錄。")
                df = df[~invalid_hl]
            else:
                self.logger.warning(msg)
        return df

    def perform_advanced_df_validation(
        self, df: Optional[pd.DataFrame], symbol: str, timeframe_str: str
    ) -> None:
        if df is None or df.empty:
            return
        self.logger.info(
            f"--- [{symbol}/{timeframe_str}] 開始進階數據驗證 (目前為佔位符) ---"
        )
        pass


# ==============================================================================
#                      3. 主流程控制類別 (MT5DataPipeline)
# ==============================================================================
class MT5DataPipeline:
    """
    專注於流程控制、與 MT5 的互動、併發任務管理。
    """

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.logger = self._setup_logger()
        self.processor = DataProcessor(config, self.logger)
        self.mt5_lock = Lock()

        self.base_dir = Path(__file__).parent
        self.output_base_dir = (
            self.base_dir / self.config.output_settings.output_base_folder_name
        )
        self.market_data_dir = (
            self.output_base_dir / self.config.output_settings.market_data_subfolder
        )
        self.charts_dir = (
            self.output_base_dir / self.config.output_settings.charts_subfolder
        )
        self._initialize_paths()

    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger(self.__class__.__name__)
        logger.setLevel(self.config.output_settings.log_level.upper())
        if logger.hasHandlers():
            logger.handlers.clear()
        formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - [%(threadName)s] - %(message)s"
        )

        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(formatter)
        logger.addHandler(sh)

        # 可選：新增檔案日誌記錄器
        # log_file = self.output_base_dir / "01_data_acquisition.log"
        # fh = logging.FileHandler(log_file)
        # fh.setFormatter(formatter)
        # logger.addHandler(fh)

        return logger

    def _initialize_paths(self) -> None:
        try:
            self.output_base_dir.mkdir(exist_ok=True)
            self.market_data_dir.mkdir(exist_ok=True)
            if self.config.output_settings.generate_charts:
                self.charts_dir.mkdir(exist_ok=True)
            self.logger.info(f"輸出總目錄: {self.output_base_dir}")
        except OSError as e:
            self.logger.critical(f"創建輸出目錄失敗: {e}", exc_info=True)
            raise

    def _connect_mt5(self) -> bool:
        with self.mt5_lock:
            if mt5.initialize():
                self.logger.info("MetaTrader 5 初始化成功。")
                return True
            else:
                self.logger.error(
                    f"初始化 MetaTrader 5 失敗, 錯誤代碼 = {mt5.last_error()}"
                )
                return False

    def _disconnect_mt5(self) -> None:
        with self.mt5_lock:
            mt5.shutdown()
        self.logger.info("MetaTrader 5 連接已關閉。")

    @staticmethod
    def _map_timeframe(tf_str: str) -> Optional[int]:
        return getattr(mt5, f"TIMEFRAME_{tf_str.upper()}", None)

    def _fetch_raw_rates_chunked(
        self, symbol: str, timeframe_mt5: int, start_dt: datetime, end_dt: datetime
    ) -> Optional[np.ndarray]:
        self.logger.info(
            f"[{symbol}] 開始分塊獲取 {start_dt.year} 至 {end_dt.year} 年的數據..."
        )
        all_rates = []
        for year in range(start_dt.year, end_dt.year + 1):
            chunk_start = datetime(year, 1, 1, tzinfo=timezone.utc)
            chunk_end = datetime(year + 1, 1, 1, tzinfo=timezone.utc)

            request_start = max(start_dt, chunk_start)
            request_end = min(end_dt, chunk_end)

            if request_start >= request_end:
                continue

            self.logger.debug(f"[{symbol}] 正在獲取 {year} 年的數據塊...")
            try:
                with self.mt5_lock:
                    rates_chunk = mt5.copy_rates_range(
                        symbol, timeframe_mt5, request_start, request_end
                    )

                if rates_chunk is not None and len(rates_chunk) > 0:
                    all_rates.append(rates_chunk)
                    self.logger.debug(
                        f"[{symbol}] 成功獲取 {len(rates_chunk)} 筆 {year} 年的數據。"
                    )
                time.sleep(self.config.mt5_connection.delay_between_requests_seconds)

            except Exception as e:
                self.logger.error(
                    f"[{symbol}] 獲取 {year} 年數據塊時出錯: {e}", exc_info=True
                )

        if not all_rates:
            self.logger.warning(f"[{symbol}] 在指定時間範圍內未獲取到任何數據。")
            return None

        return np.concatenate(all_rates)

    def _save_df_to_parquet(
        self, df: pd.DataFrame, symbol_folder: Path, filename_base: str
    ) -> None:
        try:
            output_path = symbol_folder / f"{filename_base}.parquet"
            df.to_parquet(output_path, engine="pyarrow")
            self.logger.info(f"數據已成功保存到: {output_path}")
        except Exception as e:
            self.logger.error(f"保存 Parquet 文件時發生錯誤: {e}", exc_info=True)

    def _generate_and_save_chart(
        self, df: pd.DataFrame, symbol_folder: Path, filename_base: str
    ) -> None:
        self.logger.info(f"正在為 {filename_base} 生成圖表...")
        try:
            # 為繪圖創建一個副本，以免影響原始數據
            plot_df = df.copy()
            sns.set_theme(style="whitegrid")
            fig, (ax1, ax2) = plt.subplots(
                2,
                1,
                figsize=(16, 9),
                sharex=True,
                gridspec_kw={"height_ratios": [3, 1]},
            )

            ax1.plot(
                plot_df.index,
                plot_df["close"],
                label="Close",
                color="skyblue",
                linewidth=1.5,
            )
            if len(plot_df) > 20:
                plot_df["MA20"] = plot_df["close"].rolling(20).mean()
                ax1.plot(
                    plot_df.index,
                    plot_df["MA20"],
                    label="MA20",
                    color="orange",
                    linewidth=1,
                )
            if len(plot_df) > 60:
                plot_df["MA60"] = plot_df["close"].rolling(60).mean()
                ax1.plot(
                    plot_df.index,
                    plot_df["MA60"],
                    label="MA60",
                    color="red",
                    linewidth=1,
                )

            ax1.set_title(f"{filename_base} - Price & Volume Overview", fontsize=16)
            ax1.set_ylabel("Price")
            ax1.legend()
            ax1.grid(True)

            ax2.bar(plot_df.index, plot_df["tick_volume"], color="lightgrey")
            ax2.set_ylabel("Tick Volume")
            ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

            plt.tight_layout()

            chart_output_folder = self.charts_dir / symbol_folder.name
            chart_output_folder.mkdir(exist_ok=True)

            chart_path = chart_output_folder / f"{filename_base}_overview.png"
            plt.savefig(chart_path)
            plt.close(fig)
            self.logger.info(f"圖表已成功保存到: {chart_path}")

        except Exception as e:
            self.logger.error(f"生成圖表時發生錯誤: {e}", exc_info=True)

    def _process_task(self, symbol: str, timeframe_str: str) -> str:
        """單一任務的完整處理流程，用於併發調用。包含增量更新邏輯。"""
        task_name = f"{symbol}/{timeframe_str}"
        self.logger.info(f"======== 開始處理任務: {task_name} ========")

        timeframe_mt5 = self._map_timeframe(timeframe_str)
        if timeframe_mt5 is None:
            return f"[{task_name}] 失敗：無法識別的時間週期。"

        # --- ☆☆☆ START: 增量更新邏輯 ☆☆☆ ---

        # 1. 確定輸出檔案的路徑
        sanitized_symbol_name = "".join([c if c.isalnum() else "_" for c in symbol])
        symbol_folder = self.market_data_dir / sanitized_symbol_name
        symbol_folder.mkdir(exist_ok=True)
        filename_base = f"{sanitized_symbol_name}_{timeframe_str}"
        output_path = symbol_folder / f"{filename_base}.parquet"

        req_conf = self.config.data_request
        start_dt = datetime(
            req_conf.start_year,
            req_conf.start_month,
            req_conf.start_day,
            tzinfo=timezone.utc,
        )
        end_dt = datetime(
            req_conf.end_year,
            req_conf.end_month,
            req_conf.end_day,
            23,
            59,
            59,
            tzinfo=timezone.utc,
        )

        existing_df = None
        # 2. 檢查檔案是否存在，若存在則修改 start_dt
        if output_path.exists():
            try:
                self.logger.info(f"[{task_name}] 發現已存在檔案: {output_path}")
                existing_df = pd.read_parquet(output_path)
                if not existing_df.empty:
                    # 獲取最後一筆數據的時間，並以此作為新的請求起點
                    last_timestamp = existing_df.index[-1]
                    self.logger.info(
                        f"[{task_name}] 檔案中最後一筆數據時間為: {last_timestamp}"
                    )
                    start_dt = last_timestamp
            except Exception as e:
                self.logger.warning(
                    f"[{task_name}] 讀取現有檔案 {output_path} 失敗: {e}。將執行完整下載。"
                )
                existing_df = None  # 如果讀取失敗，則退回完整下載模式

        # 如果新的開始時間已經晚於或等於結束時間，則無需下載
        if start_dt >= end_dt:
            self.logger.info(f"[{task_name}] 本地數據已是最新，無需下載。任務結束。")
            return f"[{task_name}] 成功：數據已是最新，共 {len(existing_df)} 筆數據。"

        # --- ☆☆☆ END: 增量更新邏輯 ☆☆☆ ---

        # 3. 使用可能已更新的 start_dt 獲取數據
        raw_rates = self._fetch_raw_rates_chunked(
            symbol, timeframe_mt5, start_dt, end_dt
        )

        newly_fetched_df = self.processor.process_raw_data_to_df(
            raw_rates, symbol, timeframe_str
        )

        # 如果沒有獲取到新數據
        if newly_fetched_df is None or newly_fetched_df.empty:
            if existing_df is not None:
                self.logger.info(f"[{task_name}] 未獲取到新的增量數據。任務結束。")
                return f"[{task_name}] 成功：未獲取到新數據，維持 {len(existing_df)} 筆數據。"
            else:
                return f"[{task_name}] 失敗：在指定範圍內未獲取到任何數據。"

        cleaned_new_df = self.processor.clean_and_verify_df(
            newly_fetched_df, symbol, timeframe_str
        )

        # 4. 合併新舊數據
        if existing_df is not None and cleaned_new_df is not None:
            self.logger.info(
                f"[{task_name}] 正在合併 {len(existing_df)} 筆舊數據與 {len(cleaned_new_df)} 筆新數據..."
            )
            final_df = pd.concat([existing_df, cleaned_new_df])
            # 關鍵一步：去除因時間重疊可能導致的重複索引，並保持排序
            final_df = final_df[~final_df.index.duplicated(keep="first")]
            final_df.sort_index(inplace=True)
        else:
            final_df = cleaned_new_df

        if final_df is None or final_df.empty:
            return f"[{task_name}] 失敗：數據清洗或驗證後無有效數據。"

        self.processor.perform_advanced_df_validation(final_df, symbol, timeframe_str)

        # 5. 儲存與圖表生成
        if self.config.output_settings.save_cleaned_data:
            self._save_df_to_parquet(final_df, symbol_folder, filename_base)

        if self.config.output_settings.generate_charts:
            self._generate_and_save_chart(final_df, symbol_folder, filename_base)

        return f"[{task_name}] 成功：處理後共 {len(final_df)} 筆數據。"

    def run_pipeline(self) -> None:
        main_start_time = time.time()
        self.logger.info(f"========= 數據獲取管道開始執行 (版本 2.1) =========")

        if not self._connect_mt5():
            self.logger.critical("MT5 初始化失敗，程式終止。")
            return

        tasks = [
            (symbol, tf)
            for symbol in self.config.data_request.symbol_list
            for tf in self.config.data_request.timeframe_str_list
        ]

        self.logger.info(f"準備執行 {len(tasks)} 個任務...")
        results = []

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=5, thread_name_prefix="MT5_Worker"
        ) as executor:
            future_to_task = {
                executor.submit(self._process_task, symbol, tf): (symbol, tf)
                for symbol, tf in tasks
            }

            for future in concurrent.futures.as_completed(future_to_task):
                task = future_to_task[future]
                try:
                    result_msg = future.result()
                    self.logger.info(result_msg)
                    results.append(result_msg)
                except Exception as exc:
                    error_msg = f"任務 {task} 產生例外: {exc}"
                    self.logger.error(error_msg, exc_info=True)
                    results.append(error_msg)

        self._disconnect_mt5()
        main_end_time = time.time()
        self.logger.info(
            f"========= 所有任務執行完畢，總耗時: {main_end_time - main_start_time:.2f} 秒 ========="
        )


# ==============================================================================
#                            主程式執行區塊
# ==============================================================================
if __name__ == "__main__":
    config_dict = {
        "output_settings": {
            "log_level": "INFO",
            "output_base_folder_name": "Output_Data_Pipeline_v2",
            "market_data_subfolder": "MarketData",
            "charts_subfolder": "Charts",
            "save_cleaned_data": True,
            "generate_charts": True,
        },
        "mt5_connection": {
            "retry_attempts": 3,
            "retry_delay_seconds": 5,
            "initial_sync_delay_seconds": 3,
            "delay_between_requests_seconds": 0.1,
        },
        "data_request": {
            "start_year": 2022,
            "start_month": 1,
            "start_day": 1,
            # end_year/month/day 會自動設為當前日期
            "symbol_list": ["EURUSD.sml", "USDJPY.sml", "GBPUSD.sml"],
            "timeframe_str_list": ["H1", "H4", "D1"],
            "strict_mode": True,
        },
        "data_validation_thresholds": {
            "price_jump_threshold_pct": 15.0,
            "intraday_extreme_range_std_dev_factor": 5.0,
            "rolling_window_for_volume_check": 20,
            "volume_std_dev_factor_threshold": 3.0,
        },
        "timezone_setting": "UTC",
    }

    try:
        pipeline_config = PipelineConfig(**config_dict)
        data_pipeline = MT5DataPipeline(config=pipeline_config)
        data_pipeline.run_pipeline()

    except ValidationError as e:
        print(f"配置錯誤，請檢查您的設定: \n{e}")
        sys.exit(1)
    except Exception as e:
        logging.critical(f"執行數據管道時發生未預期的嚴重錯誤: {e}", exc_info=True)
        sys.exit(1)
