# 檔名: 02_feature_engineering.py
# 描述: 讀取處理好的市場數據，並使用 pandas-ta 批量添加技術指標特徵。
# 版本: 2.1 (使用 ta.Strategy 物件)

"""
此腳本為特徵工程階段。

它會讀取由 01_data_acquisition.py 產生的 Parquet 格式市場數據，
然後使用 pandas-ta 函式庫為數據批量添加預先定義好的技術指標，
最終將帶有特徵的數據儲存到新的輸出目錄中。
"""

import logging
import sys
from pathlib import Path
from typing import List

# 註：請確保您的環境中安裝的是 pandas-ta-openbb 套件，以相容 Numpy 2.0+
# 安裝指令: pip install pandas-ta-openbb
import pandas as pd
import pandas_ta as ta


# ==============================================================================
#                            1. 配置區塊
# ==============================================================================
class Config:
    """儲存腳本所需的所有配置參數。"""

    # 輸入路徑：指向上一階段(01_...)的輸出資料夾
    INPUT_BASE_DIR = Path("Output_Data_Pipeline_v2/MarketData")

    # 輸出路徑：儲存帶有特徵的新數據
    OUTPUT_BASE_DIR = Path("Output_Feature_Engineering/MarketData_with_Features")

    # 1. 定義要計算的技術指標列表
    TA_STRATEGY_LIST = [
        # --- 動能指標 (Momentum) ---
        {"kind": "sma", "length": 20},
        {"kind": "sma", "length": 50},
        {"kind": "ema", "length": 20},
        {"kind": "ema", "length": 50},
        {"kind": "rsi"},  # 預設 length=14
        {"kind": "macd"},  # 預設 fast=12, slow=26, signal=9
        # --- 波動率指標 (Volatility) ---
        {"kind": "bbands", "length": 20, "std": 2},  # Bollinger Bands
        {"kind": "atr"},  # Average True Range, 預設 length=14
        # --- 成交量相關指標 (Volume) ---
        {"kind": "obv"},  # On-Balance Volume
    ]

    # 2. 根據上面的列表，創建一個 Strategy 物件
    MyStrategy = ta.Strategy(
        name="My Custom Strategy",
        description="A collection of common technical indicators.",
        ta=TA_STRATEGY_LIST,
    )

    # 日誌設定
    LOG_LEVEL = "INFO"


# ==============================================================================
#                            2. 特徵工程師類別
# ==============================================================================
class FeatureEngineer:
    """
    一個用於給時間序列數據批量添加技術指標的類別。
    """

    def __init__(self, config: Config):
        self.config = config
        self.logger = self._setup_logger()
        # 確保輸出目錄存在
        self.config.OUTPUT_BASE_DIR.mkdir(parents=True, exist_ok=True)

    def _setup_logger(self) -> logging.Logger:
        """配置日誌記錄器。"""
        logger = logging.getLogger(self.__class__.__name__)
        logger.setLevel(self.config.LOG_LEVEL.upper())
        if logger.hasHandlers():
            logger.handlers.clear()

        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(formatter)
        logger.addHandler(sh)
        return logger

    def find_input_files(self) -> List[Path]:
        """遞迴地尋找所有輸入的 Parquet 檔案。"""
        self.logger.info(f"正在從 '{self.config.INPUT_BASE_DIR}' 尋找輸入檔案...")
        files = list(self.config.INPUT_BASE_DIR.rglob("*.parquet"))
        self.logger.info(f"找到了 {len(files)} 個 Parquet 檔案。")
        return files

    def add_features_to_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        使用預定義的策略為 DataFrame 添加技術指標。
        """
        if df.empty:
            return df

        # 使用定義好的 Strategy 物件來批量應用指標
        # 關鍵：將方法改回 strategy()，並傳入 MyStrategy 物件
        df.ta.strategy(self.config.MyStrategy, append=True)

        return df

    def process_file(self, file_path: Path) -> None:
        """
        處理單一檔案：讀取、添加特徵、儲存。
        """
        try:
            self.logger.info(f"--- 開始處理檔案: {file_path.name} ---")
            df = pd.read_parquet(file_path)

            initial_cols = df.shape[1]
            df_with_features = self.add_features_to_dataframe(df)

            df_with_features.dropna(inplace=True)

            final_cols = df_with_features.shape[1]
            self.logger.info(f"成功添加了 {final_cols - initial_cols} 個新特徵。")

            relative_path = file_path.relative_to(self.config.INPUT_BASE_DIR)
            output_path = self.config.OUTPUT_BASE_DIR / relative_path.with_name(
                f"{file_path.stem}_features.parquet"
            )

            output_path.parent.mkdir(parents=True, exist_ok=True)

            df_with_features.to_parquet(output_path)
            self.logger.info(f"已儲存帶有特徵的檔案到: {output_path}")

        except Exception as e:
            self.logger.error(
                f"處理檔案 {file_path.name} 時發生錯誤: {e}", exc_info=True
            )

    def run(self) -> None:
        """
        執行完整的特徵工程流程。
        """
        self.logger.info("========= 特徵工程流程開始 =========")
        input_files = self.find_input_files()

        if not input_files:
            self.logger.warning("在輸入目錄中沒有找到任何 Parquet 檔案，流程結束。")
            return

        for file_path in input_files:
            self.process_file(file_path)

        self.logger.info("========= 所有檔案處理完畢，特徵工程流程結束 =========")


# ==============================================================================
#                            3. 主程式執行區塊
# ==============================================================================
if __name__ == "__main__":
    try:
        config = Config()
        engineer = FeatureEngineer(config)
        engineer.run()
    except Exception as e:
        logging.critical(f"特徵工程腳本執行時發生未預期的嚴重錯誤: {e}", exc_info=True)
        sys.exit(1)
