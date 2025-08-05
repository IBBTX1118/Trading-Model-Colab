# 檔名: 02_feature_engineering.py
# 版本: 4.0 (深度特徵工程版)

"""
此腳本為特徵工程的最終擴展版。
除了標準技術指標，還加入了基於價格行為（K線結構）、
市場狀態（趨勢/盤整）和突破分析的進階複合特徵。
"""

import logging
import sys
from pathlib import Path
from typing import List
import pandas as pd
import numpy as np
from finta import TA
import pandas_ta as pta # ★ 導入新的函式庫

# ... (Config 和 FeatureEngineer 的 __init__, _setup_logger, find_input_files 維持不變) ...

class Config:
    INPUT_BASE_DIR = Path("Output_Data_Pipeline_v2/MarketData")
    OUTPUT_BASE_DIR = Path("Output_Feature_Engineering/MarketData_with_Deep_Features") # 使用新目錄
    LOG_LEVEL = "INFO"

class FeatureEngineer:
    def __init__(self, config: Config):
        self.config = config
        self.logger = self._setup_logger()
        self.config.OUTPUT_BASE_DIR.mkdir(parents=True, exist_ok=True)

    def _setup_logger(self) -> logging.Logger:
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
        self.logger.info(f"正在從 '{self.config.INPUT_BASE_DIR}' 尋找輸入檔案...")
        files = list(self.config.INPUT_BASE_DIR.rglob("*.parquet"))
        self.logger.info(f"找到了 {len(files)} 個 Parquet 檔案。")
        return files

    def add_features_to_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        
        self.logger.info("計算標準技術指標...")
        df_finta = df.copy()
        df_finta.rename(columns={
            'open': 'open', 'high': 'high', 'low': 'low', 
            'close': 'close', 'tick_volume': 'volume'
        }, inplace=True)

        # --- 標準 finta 指標 ---
        standard_indicators = [
            'SMA_10', 'SMA_20', 'SMA_50', 'EMA_10', 'EMA_20', 'EMA_50',
            'RSI_14', 'WILLIAMS', 'CCI', 'SAR', 'OBV', 'MFI', 'ATR_14'
        ]
        for indicator in standard_indicators:
            method_to_call = getattr(TA, indicator.split('_')[0])
            if '_' in indicator:
                period = int(indicator.split('_')[1])
                df_finta[indicator] = method_to_call(df_finta, period=period)
            else:
                df_finta[indicator] = method_to_call(df_finta)
        
        df_finta = df_finta.join(TA.DMI(df_finta))
        df_finta = df_finta.join(TA.STOCH(df_finta))
        df_finta = df_finta.join(TA.MACD(df_finta))
        df_finta = df_finta.join(TA.BBANDS(df_finta))

        # --- ★★★ 第一階段：進階特徵工程 ★★★ ---
        self.logger.info("計算進階複合特徵...")

        # 1. K線本體與影線分析
        df_finta['body_size'] = abs(df_finta['close'] - df_finta['open'])
        df_finta['upper_wick'] = df_finta['high'] - df_finta[['open', 'close']].max(axis=1)
        df_finta['lower_wick'] = df_finta[['open', 'close']].min(axis=1) - df_finta['low']
        df_finta['body_vs_wick'] = df_finta['body_size'] / (df_finta['high'] - df_finta['low'] + 1e-9)

        # 2. 趨勢強度與盤整識別 (使用 pandas_ta 計算 ADX)
        adx_df = pta.adx(df_finta['high'], df_finta['low'], df_finta['close'], length=14)
        df_finta['ADX_14'] = adx_df[f'ADX_14']
        df_finta['DMI_DIFF'] = abs(df_finta['DI+'] - df_finta['DI-'])

        # 3. 價格突破與通道位置
        df_finta['ROLLING_HIGH_20'] = df_finta['high'].rolling(window=20).max()
        df_finta['ROLLING_LOW_20'] = df_finta['low'].rolling(window=20).min()
        rolling_range = df_finta['ROLLING_HIGH_20'] - df_finta['ROLLING_LOW_20']
        df_finta['CLOSE_vs_ROLLING_RANGE'] = (df_finta['close'] - df_finta['ROLLING_LOW_20']) / (rolling_range + 1e-9)

        # 4. 更多既有指標的複合使用
        df_finta['CLOSE_vs_SMA50'] = df_finta['close'] / df_finta['SMA_50']
        df_finta['SMA_ratio_10_50'] = df_finta['SMA_10'] / df_finta['SMA_50']

        # --- 清理與收尾 ---
        df_finta.rename(columns={'volume': 'tick_volume'}, inplace=True)
        df_finta.replace([np.inf, -np.inf], np.nan, inplace=True)
        return df_finta

    # ... (process_file 和 run 方法維持不變, 只需確認輸出目錄為新目錄) ...
    def process_file(self, file_path: Path) -> None:
        try:
            self.logger.info(f"--- 開始處理檔案: {file_path.name} ---")
            df = pd.read_parquet(file_path)
            
            initial_cols = df.shape[1]
            df_with_features = self.add_features_to_dataframe(df)
            
            rows_before_dropna = len(df_with_features)
            df_with_features.dropna(inplace=True)
            rows_after_dropna = len(df_with_features)
            self.logger.info(f"移除了 {rows_before_dropna - rows_after_dropna} 行包含 NaN 的數據。")
            
            final_cols = df_with_features.shape[1]
            self.logger.info(f"成功添加了 {final_cols - initial_cols} 個新特徵。總特徵數: {final_cols}")
            
            relative_path = file_path.relative_to(self.config.INPUT_BASE_DIR)
            output_path = self.config.OUTPUT_BASE_DIR / relative_path
            
            output_path.parent.mkdir(parents=True, exist_ok=True)
            df_with_features.to_parquet(output_path)
            self.logger.info(f"已儲存帶有深度特徵的檔案到: {output_path}")

        except Exception as e:
            self.logger.error(f"處理檔案 {file_path.name} 時發生錯誤: {e}", exc_info=True)

    def run(self) -> None:
        self.logger.info("========= 深度特徵工程流程開始 (v4.0) =========")
        input_files = self.find_input_files()
        
        if not input_files:
            self.logger.warning("在輸入目錄中沒有找到任何 Parquet 檔案，流程結束。")
            return

        for file_path in input_files:
            self.process_file(file_path)
            
        self.logger.info("========= 所有檔案處理完畢，深度特徵工程流程結束 =========")

if __name__ == "__main__":
    try:
        config = Config()
        engineer = FeatureEngineer(config)
        engineer.run()
    except Exception as e:
        logging.critical(f"特徵工程腳本執行時發生未預期的嚴重錯誤: {e}", exc_info=True)
        sys.exit(1)
