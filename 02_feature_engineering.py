# 檔名: 02_feature_engineering.py
# 版本: 5.0 (綜合特徵版 - 整合多週期、跨頻率、互動特徵)

import logging
import sys
from pathlib import Path
from typing import List, Dict
import pandas as pd
import numpy as np
from finta import TA
from collections import defaultdict

class Config:
    INPUT_BASE_DIR = Path("Output_Data_Pipeline_v2/MarketData")
    # ★★★ 使用新的輸出目錄以區分版本 ★★★
    OUTPUT_BASE_DIR = Path("Output_Feature_Engineering/MarketData_with_Combined_Features_v3")
    LOG_LEVEL = "INFO"
    # 定義要計算的時間頻率順序 (從高到低)
    TIMEFRAME_ORDER = ['D1', 'H4', 'H1']

class FeatureEngineer:
    def __init__(self, config: Config):
        self.config = config
        self.logger = self._setup_logger()
        self.config.OUTPUT_BASE_DIR.mkdir(parents=True, exist_ok=True)

    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger(self.__class__.__name__)
        logger.setLevel(self.config.LOG_LEVEL.upper())
        if not logger.hasHandlers():
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            sh = logging.StreamHandler(sys.stdout)
            sh.setFormatter(formatter)
            logger.addHandler(sh)
        return logger

    def _add_base_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """計算基礎技術指標 (源自舊版 v4.1)"""
        df_finta = df.copy()
        df_finta.rename(columns={'tick_volume': 'volume'}, inplace=True)
        
        # --- finta 指標 ---
        df_finta = df_finta.join(TA.DMI(df_finta))
        standard_indicators = ['SMA', 'EMA', 'RSI', 'WILLIAMS', 'CCI', 'SAR', 'OBV', 'MFI', 'ATR']
        periods = [10, 20, 50]
        for indicator in standard_indicators:
            method = getattr(TA, indicator)
            if indicator in ['RSI', 'ATR']:
                 df_finta[f'{indicator}_14'] = method(df_finta, period=14)
            elif indicator in ['WILLIAMS', 'CCI', 'SAR', 'OBV', 'MFI']:
                 df_finta[indicator] = method(df_finta)
            else: # SMA, EMA
                for p in periods:
                    df_finta[f'{indicator}_{p}'] = method(df_finta, period=p)

        df_finta = df_finta.join(TA.STOCH(df_finta))
        df_finta = df_finta.join(TA.MACD(df_finta))
        df_finta = df_finta.join(TA.BBANDS(df_finta))
        
        # --- K線結構特徵 ---
        df_finta['body_size'] = abs(df_finta['close'] - df_finta['open'])
        df_finta['upper_wick'] = df_finta['high'] - df_finta[['open', 'close']].max(axis=1)
        df_finta['lower_wick'] = df_finta[['open', 'close']].min(axis=1) - df_finta['low']
        df_finta['body_vs_wick'] = df_finta['body_size'] / (df_finta['high'] - df_finta['low'] + 1e-9)
        df_finta['DMI_DIFF'] = abs(df_finta['DI+'] - df_finta['DI-'])

        df_finta.rename(columns={'volume': 'tick_volume'}, inplace=True)
        return df_finta

    def _add_multi_period_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """★★★ 新增：計算多週期的時間序列特徵 ★★★"""
        df_out = df.copy()
        for n in [5, 10, 20, 60]:
            # N週期報酬率
            df_out[f'returns_{n}p'] = df_out['close'].pct_change(periods=n)
            # N週期波動率
            df_out[f'volatility_{n}p'] = df_out['close'].pct_change().rolling(window=n).std() * np.sqrt(n)
        return df_out

    def _add_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """★★★ 新增：計算互動特徵 ★★★"""
        df_out = df.copy()
        # 範例：RSI 與波動率的互動，可能表示在低波動時 RSI 更有效
        if 'RSI_14' in df_out.columns and 'volatility_20p' in df_out.columns:
            df_out['RSI_x_volatility'] = (df_out['RSI_14'] - 50) * df_out['volatility_20p']
        # 範例：趨勢強度與 ATR 的互動
        if 'DMI_DIFF' in df_out.columns and 'ATR_14' in df_out.columns:
            df_out['DMI_DIFF_x_ATR'] = df_out['DMI_DIFF'] * df_out['ATR_14']
        return df_out

    def process_symbol_group(self, symbol: str, paths: Dict[str, Path]):
        """處理單一商品的所有時間頻率數據，並實現跨頻率特徵融合"""
        self.logger.info(f"--- 開始處理商品組: {symbol} ---")
        
        dataframes = {}
        # 1. 讀取並為每個時間頻率計算基礎特徵
        for tf in self.config.TIMEFRAME_ORDER:
            if tf in paths:
                self.logger.info(f"[{symbol}] 正在處理 {tf} 數據...")
                df = pd.read_parquet(paths[tf])
                df_with_features = self._add_base_features(df)
                df_with_features = self._add_multi_period_features(df_with_features)
                df_with_features = self._add_interaction_features(df_with_features)
                dataframes[tf] = df_with_features

        # 2. ★★★ 核心邏輯：從高頻率向低頻率融合特徵 ★★★
        final_dataframes = {}
        # D1 數據只有自己的特徵
        if 'D1' in dataframes:
            final_dataframes['D1'] = dataframes['D1']

        # 為 H4 數據融合 D1 特徵
        if 'H4' in dataframes and 'D1' in dataframes:
            self.logger.info(f"[{symbol}] 正在為 H4 數據融合 D1 特徵...")
            df_d1_renamed = dataframes['D1'].rename(columns=lambda c: f"D1_{c}" if c not in ['open', 'high', 'low', 'close', 'tick_volume'] else c)
            final_dataframes['H4'] = pd.merge_asof(
                dataframes['H4'].sort_index(),
                df_d1_renamed,
                left_index=True, right_index=True, direction='backward'
            )
        elif 'H4' in dataframes:
             final_dataframes['H4'] = dataframes['H4']

        # 為 H1 數據融合 H4 (已包含D1) 的特徵
        if 'H1' in dataframes and 'H4' in final_dataframes:
            self.logger.info(f"[{symbol}] 正在為 H1 數據融合 H4/D1 特徵...")
            df_h4_renamed = final_dataframes['H4'].rename(columns=lambda c: f"H4_{c}" if c not in ['open', 'high', 'low', 'close', 'tick_volume'] else c)
            final_dataframes['H1'] = pd.merge_asof(
                dataframes['H1'].sort_index(),
                df_h4_renamed,
                left_index=True, right_index=True, direction='backward'
            )
        elif 'H1' in dataframes:
            final_dataframes['H1'] = dataframes['H1']

        # 3. 儲存所有處理完成的數據集
        for tf, df_final in final_dataframes.items():
            df_final.replace([np.inf, -np.inf], np.nan, inplace=True)
            
            rows_before = len(df_final)
            df_final.dropna(inplace=True)
            rows_after = len(df_final)
            self.logger.info(f"[{symbol}/{tf}] 移除了 {rows_before - rows_after} 行包含 NaN 的數據。剩餘 {rows_after} 筆。")
            
            # 建立與輸入對應的輸出路徑
            relative_path = paths[tf].relative_to(self.config.INPUT_BASE_DIR)
            output_path = self.config.OUTPUT_BASE_DIR / relative_path
            
            output_path.parent.mkdir(parents=True, exist_ok=True)
            df_final.to_parquet(output_path)
            self.logger.info(f"[{symbol}/{tf}] 已儲存綜合特徵檔案到: {output_path}")

    def run(self):
        """★★★ 全新執行邏輯：按商品分組處理 ★★★"""
        self.logger.info(f"========= 綜合特徵工程流程開始 (v5.0) =========")
        
        # 尋找所有輸入檔案
        input_files = list(self.config.INPUT_BASE_DIR.rglob("*.parquet"))
        if not input_files:
            self.logger.warning("在輸入目錄中沒有找到任何 Parquet 檔案，流程結束。")
            return

        # 按金融商品名稱分組
        symbol_groups = defaultdict(dict)
        for file_path in input_files:
            try:
                # 檔名格式為 SYMBOL_TIMEFRAME.parquet
                parts = file_path.stem.split('_')
                symbol = '_'.join(parts[:-1])
                timeframe = parts[-1]
                if timeframe in self.config.TIMEFRAME_ORDER:
                    symbol_groups[symbol][timeframe] = file_path
            except IndexError:
                self.logger.warning(f"無法解析檔名: {file_path.name}，已跳過。")
        
        self.logger.info(f"共找到 {len(symbol_groups)} 個商品組需要處理。")

        for symbol, paths in symbol_groups.items():
            self.process_symbol_group(symbol, paths)
            
        self.logger.info("========= 所有檔案處理完畢，綜合特徵工程流程結束 =========")

if __name__ == "__main__":
    try:
        config = Config()
        engineer = FeatureEngineer(config)
        engineer.run()
    except Exception as e:
        logging.critical(f"特徵工程腳本執行時發生未預期的嚴重錯誤: {e}", exc_info=True)
        sys.exit(1)
