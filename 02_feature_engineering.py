# 檔名: 02_feature_engineering.py
# 版本: 5.0 (正式版：使用 finta)

"""
此腳本為特徵工程的正式版本。

它會讀取處理好的市場數據，並使用 finta (純 Python 函式庫)
來為數據批量添加技術指標，解決 TA-Lib 的安裝問題。
"""

import logging
import sys
from pathlib import Path
from typing import List

import pandas as pd
from finta import TA  # 導入 finta


# ==============================================================================
#                            1. 配置區塊
# ==============================================================================
class Config:
    """儲存腳本所需的所有配置參數。"""
    INPUT_BASE_DIR = Path("Output_Data_Pipeline_v2/MarketData")
    OUTPUT_BASE_DIR = Path("Output_Feature_Engineering/MarketData_with_Features")
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
        self.config.OUTPUT_BASE_DIR.mkdir(parents=True, exist_ok=True)

    def _setup_logger(self) -> logging.Logger:
        """配置日誌記錄器。"""
        # ... (此函數內容與之前版本相同) ...
        logger = logging.getLogger(self.__class__.__name__)
        logger.setLevel(self.config.LOG_LEVEL.upper())
        if logger.hasHandlers():
            logger.handlers.clear()
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(formatter)
        logger.addHandler(sh)
        return logger

    def find_input_files(self) -> List[Path]:
        """遞迴地尋找所有輸入的 Parquet 檔案。"""
        # ... (此函數內容與之前版本相同) ...
        self.logger.info(f"正在從 '{self.config.INPUT_BASE_DIR}' 尋找輸入檔案...")
        files = list(self.config.INPUT_BASE_DIR.rglob("*.parquet"))
        self.logger.info(f"找到了 {len(files)} 個 Parquet 檔案。")
        return files

    def add_features_to_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        使用 finta 為 DataFrame 添加技術指標。
        finta 會自動尋找 'open', 'high', 'low', 'close', 'volume' 欄位。
        """
        if df.empty:
            return df
        
        self.logger.info("使用 finta 計算技術指標...")

        # --- 動能指標 (Momentum) ---
        df['SMA_20'] = TA.SMA(df, period=20)
        df['SMA_50'] = TA.SMA(df, period=50)
        df['EMA_20'] = TA.EMA(df, period=20)
        df['EMA_50'] = TA.EMA(df, period=50)
        df['RSI_14'] = TA.RSI(df, period=14)
        
        # MACD 會回傳一個包含 MACD 和 SIGNAL 欄位的 DataFrame，我們將它合併回來
        macd = TA.MACD(df)
        df = df.join(macd)

        # --- 波動率指標 (Volatility) ---
        # 布林帶也會回傳一個包含多個欄位的 DataFrame
        bbands = TA.BBANDS(df)
        df = df.join(bbands)
        
        df['ATR_14'] = TA.ATR(df, period=14)

        # --- 成交量相關指標 (Volume) ---
        # finta 的 volume 欄位名稱預設是 'volume'，我們的是 'tick_volume'
        # 我們需要先將欄位重命名，計算完再改回來
        df.rename(columns={'tick_volume': 'volume'}, inplace=True)
        df['OBV'] = TA.OBV(df)
        df.rename(columns={'volume': 'tick_volume'}, inplace=True)
        
        return df

    def process_file(self, file_path: Path) -> None:
        """
        處理單一檔案：讀取、添加特徵、儲存。
        """
        # ... (此函數內容與之前版本相同) ...
        try:
            self.logger.info(f"--- 開始處理檔案: {file_path.name} ---")
            df = pd.read_parquet(file_path)
            initial_cols = df.shape[1]
            df_with_features = self.add_features_to_dataframe(df)
            df_with_features.dropna(inplace=True)
            final_cols = df_with_features.shape[1]
            self.logger.info(f"成功添加了 {final_cols - initial_cols} 個新特徵。")
            relative_path = file_path.relative_to(self.config.INPUT_BASE_DIR)
            output_path = self.config.OUTPUT_BASE_DIR / relative_path.with_name(f"{file_path.stem}_features.parquet")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            df_with_features.to_parquet(output_path)
            self.logger.info(f"已儲存帶有特徵的檔案到: {output_path}")
        except Exception as e:
            self.logger.error(f"處理檔案 {file_path.name} 時發生錯誤: {e}", exc_info=True)


    def run(self) -> None:
        """
        執行完整的特徵工程流程。
        """
        # ... (此函數內容與之前版本相同) ...
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
