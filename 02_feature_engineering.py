# 檔名: 02_feature_engineering.py
# 版本: 5.9 (大幅增強特徵工程)
# 描述: 整合多週期、跨頻率、互動以及宏觀事件、全局因子與多維度高級特徵。

import logging
import sys
from pathlib import Path
from typing import Dict, Optional
import pandas as pd
import numpy as np
from finta import TA
from collections import defaultdict

class Config:
    INPUT_BASE_DIR = Path("Output_Data_Pipeline_v2/MarketData")
    OUTPUT_BASE_DIR = Path("Output_Feature_Engineering/MarketData_with_Combined_Features_v3")
    LOG_LEVEL = "INFO"
    TIMEFRAME_ORDER = ['D1', 'H4', 'H1']
    EVENTS_FILE_PATH = Path("economic_events.csv")
    GLOBAL_FACTORS_FILE_PATH = Path("global_factors.parquet")

class FeatureEngineer:
    def __init__(self, config: Config):
        self.config = config
        self.logger = self._setup_logger()
        self.config.OUTPUT_BASE_DIR.mkdir(parents=True, exist_ok=True)

    def _setup_logger(self) -> logging.Logger:
        # ... 此函數無變動 ...
        logger = logging.getLogger(self.__class__.__name__)
        logger.setLevel(self.config.LOG_LEVEL.upper())
        if not logger.hasHandlers():
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            sh = logging.StreamHandler(sys.stdout)
            sh.setFormatter(formatter)
            logger.addHandler(sh)
        return logger
        
    def _add_global_factor_features(self, df: pd.DataFrame, factors_df: Optional[pd.DataFrame]) -> pd.DataFrame:
        # ... 此函數無變動 ...
        if factors_df is None or factors_df.empty: return df
        df_merged = pd.merge_asof(df.sort_index(), factors_df, left_index=True, right_index=True, direction='backward')
        if 'DXY_close' in df_merged.columns:
            df_merged['DXY_return_1d'] = df_merged['DXY_close'].pct_change(periods=1)
            df_merged['DXY_return_5d'] = df_merged['DXY_close'].pct_change(periods=5)
        if 'VIX_close' in df_merged.columns:
            df_merged['VIX_SMA_20'] = df_merged['VIX_close'].rolling(window=20).mean()
            if 'VIX_SMA_20' in df_merged.columns and df_merged['VIX_SMA_20'].notna().any():
                 df_merged['VIX_vs_SMA20'] = (df_merged['VIX_close'] - df_merged['VIX_SMA_20']) / (df_merged['VIX_SMA_20'] + 1e-9)
        self.logger.debug("Added global factor features (DXY, VIX).")
        return df_merged

    def _add_base_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df_finta = df.copy()
        df_finta.rename(columns={'tick_volume': 'volume'}, inplace=True)
        
        # ★★★ 修改：增加更多週期的 ATR 計算，供進階特徵使用 ★★★
        standard_indicators = ['SMA', 'EMA', 'RSI', 'WILLIAMS', 'CCI', 'SAR', 'OBV', 'MFI', 'ATR']
        for p in [5, 14, 20]: df_finta[f'ATR_{p}'] = TA.ATR(df_finta, period=p)
        
        # ... 其餘基礎指標計算無變動 ...
        df_finta = df_finta.join(TA.DMI(df_finta))
        periods = [10, 20, 50]
        for indicator in ['SMA', 'EMA', 'RSI', 'WILLIAMS', 'CCI', 'SAR', 'OBV', 'MFI']:
            method = getattr(TA, indicator)
            if indicator in ['RSI']: df_finta[f'{indicator}_14'] = method(df_finta, period=14)
            elif indicator in ['WILLIAMS', 'CCI', 'SAR', 'OBV', 'MFI']: df_finta[indicator] = method(df_finta)
            else:
                for p in periods: df_finta[f'{indicator}_{p}'] = method(df_finta, period=p)
        df_finta = df_finta.join(TA.STOCH(df_finta)); df_finta = df_finta.join(TA.MACD(df_finta)); df_finta = df_finta.join(TA.BBANDS(df_finta))
        
        df_finta['body_size'] = abs(df_finta['close'] - df_finta['open'])
        df_finta['upper_wick'] = df_finta['high'] - df_finta[['open', 'close']].max(axis=1)
        df_finta['lower_wick'] = df_finta[['open', 'close']].min(axis=1) - df_finta['low']
        df_finta['body_vs_wick'] = df_finta['body_size'] / (df_finta['high'] - df_finta['low'] + 1e-9)
        if 'DI+' in df_finta.columns and 'DI-' in df_finta.columns: df_finta['DMI_DIFF'] = abs(df_finta['DI+'] - df_finta['DI-'])

        if 'ATR_14' in df_finta.columns:
            atr_series = df_finta['ATR_14'] + 1e-9
            df_finta['body_size_norm'] = df_finta['body_size'] / atr_series
            if 'DMI_DIFF' in df_finta.columns: df_finta['DMI_DIFF_norm'] = df_finta['DMI_DIFF'] / atr_series
        
        df_finta.rename(columns={'volume': 'tick_volume'}, inplace=True)
        return df_finta

    # ★★★ 新增：進階時間特徵 ★★★
    def _add_advanced_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df_out = df.copy()
        hour = df_out.index.hour
        df_out['is_tokyo_session'] = ((hour >= 0) & (hour <= 8)).astype(int)
        df_out['is_london_session'] = ((hour >= 8) & (hour <= 16)).astype(int)
        df_out['is_ny_session'] = ((hour >= 13) & (hour <= 21)).astype(int)
        df_out['is_london_ny_overlap'] = ((hour >= 13) & (hour <= 16)).astype(int)
        df_out['day_of_week'] = df_out.index.dayofweek
        df_out['month_of_year'] = df_out.index.month
        self.logger.debug("Added advanced time-based features.")
        return df_out

    # ★★★ 新增：價格行為特徵 ★★★
    def _add_price_action_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df_out = df.copy()
        if 'SMA_50' in df_out.columns and 'ATR_14' in df_out.columns:
            df_out['price_vs_sma50_norm'] = (df_out['close'] - df_out['SMA_50']) / (df_out['ATR_14'] + 1e-9)
        
        window = 50
        df_out[f'bars_since_{window}p_high'] = df_out['high'].rolling(window).apply(lambda x: len(x) - np.argmax(x) - 1, raw=True)
        df_out[f'bars_since_{window}p_low'] = df_out['low'].rolling(window).apply(lambda x: len(x) - np.argmin(x) - 1, raw=True)
        self.logger.debug("Added price action features.")
        return df_out

    # ★★★ 新增：進階波動率特徵 ★★★
    def _add_advanced_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df_out = df.copy()
        if 'ATR_5' in df_out.columns and 'ATR_20' in df_out.columns:
            df_out['atr_ratio_5_20'] = df_out['ATR_5'] / (df_out['ATR_20'] + 1e-9)
        if 'ATR_14' in df_out.columns:
            df_out['atr_volatility_20p'] = df_out['ATR_14'].rolling(window=20).std()
        self.logger.debug("Added advanced volatility features.")
        return df_out

    # ★★★ 新增：跨市場關聯特徵 ★★★
    def _add_cross_market_features(self, df: pd.DataFrame, symbol: str, all_market_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        df_out = df.copy()
        # 範例：僅為 GBPUSD 添加與 EURUSD 的滾動相關性
        if "GBPUSD" in symbol and "EURUSD_sml_H4" in all_market_data:
            eurusd_df = all_market_data["EURUSD_sml_H4"]
            # 確保兩個 DataFrame 的索引對齊
            aligned_df1, aligned_df2 = df_out.align(eurusd_df, join='left', axis=0)
            
            gbp_returns = aligned_df1['close'].pct_change()
            eur_returns = aligned_df2['close'].pct_change()
            
            df_out['corr_gbpusd_eurusd_50p'] = gbp_returns.rolling(window=50).corr(eur_returns)
            self.logger.debug("Added cross-market features for GBPUSD.")
        return df_out

    def _add_multi_period_features(self, df: pd.DataFrame) -> pd.DataFrame:
        # ... 此函數無變動 ...
        df_out = df.copy()
        for n in [5, 10, 20, 60]:
            df_out[f'returns_{n}p'] = df_out['close'].pct_change(periods=n)
            df_out[f'volatility_{n}p'] = df_out['close'].pct_change().rolling(window=n).std() * np.sqrt(n)
        return df_out

    def _add_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        # ... 此函數無變動 ...
        df_out = df.copy()
        if 'RSI_14' in df_out.columns and 'volatility_20p' in df_out.columns:
            df_out['RSI_x_volatility'] = (df_out['RSI_14'] - 50) * df_out['volatility_20p']
        if 'DMI_DIFF' in df_out.columns and 'ATR_14' in df_out.columns:
            df_out['DMI_DIFF_x_ATR'] = df_out['DMI_DIFF'] * df_out['ATR_14']
        return df_out
        
    def _add_event_features(self, df: pd.DataFrame, events_df: Optional[pd.DataFrame], symbol: str) -> pd.DataFrame:
        # ... 此函數無變動 ...
        df_out = df.copy()
        if events_df is None or events_df.empty:
            df_out['time_to_next_event'] = 999; df_out['next_event_importance'] = 0; return df_out
        currency1 = symbol[:3].upper(); currency2 = symbol[3:6].upper()
        relevant_events = events_df[events_df['currency'].isin([currency1, currency2])].copy()
        if relevant_events.empty:
            df_out['time_to_next_event'] = 999; df_out['next_event_importance'] = 0; return df_out
        relevant_events.reset_index(inplace=True)
        df_merged = pd.merge_asof(left=df_out.sort_index(), right=relevant_events.sort_values('timestamp_utc'), left_index=True, right_on='timestamp_utc', direction='forward')
        df_merged.index = df_out.index
        time_diff = df_merged['timestamp_utc'] - df_out.index
        df_out['time_to_next_event'] = time_diff.dt.total_seconds() / 3600
        df_out['next_event_importance'] = df_merged['importance_val']
        df_out.fillna({'time_to_next_event': 999, 'next_event_importance': 0}, inplace=True)
        return df_out

    # ★★★ 修改：調整函數簽名與特徵計算順序 ★★★
    def process_symbol_group(self, symbol: str, paths: Dict[str, Path], events_df: Optional[pd.DataFrame], factors_df: Optional[pd.DataFrame], all_market_data: Dict[str, pd.DataFrame]):
        self.logger.info(f"--- 開始處理商品組: {symbol} ---")
        
        dataframes = {}
        for tf in self.config.TIMEFRAME_ORDER:
            if tf in paths:
                self.logger.info(f"[{symbol}] 正在處理 {tf} 數據...")
                df = pd.read_parquet(paths[tf])
                
                # ★★★ 新的特徵計算流程 ★★★
                df = self._add_advanced_time_features(df)
                df = self._add_global_factor_features(df, factors_df)
                df = self._add_event_features(df, events_df, symbol)
                df = self._add_base_features(df)
                df = self._add_advanced_volatility_features(df)
                df = self._add_price_action_features(df)
                df = self._add_multi_period_features(df)
                df = self._add_cross_market_features(df, f"{symbol}_{tf}", all_market_data)
                df = self._add_interaction_features(df)
                
                if tf == 'D1':
                    df[f'trend_ma_200'] = TA.SMA(df, period=200)
                    df['is_uptrend'] = (df['close'] > df[f'trend_ma_200']).astype(int)
                    df.drop(columns=[f'trend_ma_200'], inplace=True)
                    self.logger.info(f"[{symbol}/{tf}] 已新增 'is_uptrend' 趨勢過濾特徵。")
                dataframes[tf] = df

        # ... 後續融合邏輯無變動 ...
        final_dataframes = {}
        cols_to_drop_before_merge = ['open', 'high', 'low', 'close', 'tick_volume', 'real_volume', 'spread']
        if 'D1' in dataframes: final_dataframes['D1'] = dataframes['D1']
        if 'H4' in dataframes and 'D1' in dataframes:
            self.logger.info(f"[{symbol}] 正在為 H4 數據融合 D1 特徵...")
            df_d1_renamed = dataframes['D1'].rename(columns=lambda c: f"D1_{c}" if c not in cols_to_drop_before_merge else c)
            df_d1_features_only = df_d1_renamed.drop(columns=cols_to_drop_before_merge, errors='ignore')
            final_dataframes['H4'] = pd.merge_asof(dataframes['H4'].sort_index(), df_d1_features_only, left_index=True, right_index=True, direction='backward')
        elif 'H4' in dataframes: final_dataframes['H4'] = dataframes['H4']
        if 'H1' in dataframes and 'H4' in final_dataframes:
            self.logger.info(f"[{symbol}] 正在為 H1 數據融合 H1 特徵...")
            df_h4_renamed = final_dataframes['H4'].rename(columns=lambda c: f"H4_{c}" if c not in cols_to_drop_before_merge else c)
            df_h4_features_only = df_h4_renamed.drop(columns=cols_to_drop_before_merge, errors='ignore')
            final_dataframes['H1'] = pd.merge_asof(dataframes['H1'].sort_index(), df_h4_features_only, left_index=True, right_index=True, direction='backward')
        elif 'H1' in dataframes: final_dataframes['H1'] = dataframes['H1']
        for tf, df_final in final_dataframes.items():
            df_final.replace([np.inf, -np.inf], np.nan, inplace=True)
            rows_before = len(df_final)
            df_final.dropna(inplace=True)
            rows_after = len(df_final)
            if rows_before > rows_after: self.logger.info(f"[{symbol}/{tf}] 移除了 {rows_before - rows_after} 行包含 NaN 的數據。剩餘 {rows_after} 筆。")
            relative_path = paths[tf].relative_to(self.config.INPUT_BASE_DIR)
            output_path = self.config.OUTPUT_BASE_DIR / relative_path
            output_path.parent.mkdir(parents=True, exist_ok=True)
            df_final.to_parquet(output_path)
            self.logger.info(f"[{symbol}/{tf}] 已儲存綜合特徵檔案到: {output_path}")

    # ★★★ 修改：主流程增加數據預加載步驟 ★★★
    def run(self):
        self.logger.info(f"========= 綜合特徵工程流程開始 (v5.9) =========")
        events_df, factors_df = None, None
        try:
            events_df = pd.read_csv(self.config.EVENTS_FILE_PATH, index_col='timestamp_utc', parse_dates=True)
            events_df.index = pd.to_datetime(events_df.index, utc=True)
            self.logger.info(f"成功從 {self.config.EVENTS_FILE_PATH} 讀取事件數據。")
        except FileNotFoundError: self.logger.warning(f"警告：找不到事件檔案 {self.config.EVENTS_FILE_PATH}！")
        try:
            factors_df = pd.read_parquet(self.config.GLOBAL_FACTORS_FILE_PATH)
            factors_df.index = pd.to_datetime(factors_df.index, utc=True)
            self.logger.info(f"成功從 {self.config.GLOBAL_FACTORS_FILE_PATH} 讀取全局因子數據。")
        except FileNotFoundError: self.logger.warning(f"警告：找不到全局因子檔案 {self.config.GLOBAL_FACTORS_FILE_PATH}！")
        
        input_files = list(self.config.INPUT_BASE_DIR.rglob("*.parquet"))
        if not input_files:
            self.logger.warning("在輸入目錄中沒有找到任何 Parquet 檔案，流程結束。"); return

        # ★★★ 新增：預加載所有市場數據以供跨市場分析 ★★★
        all_market_data = {}
        self.logger.info("正在預加載所有市場數據以進行跨市場分析...")
        for file_path in input_files:
             if "_H4.parquet" in file_path.name: # 暫時只加載 H4 數據
                try:
                    all_market_data[file_path.stem] = pd.read_parquet(file_path)
                except Exception as e:
                    self.logger.error(f"預加載檔案 {file_path.name} 失敗: {e}")
        self.logger.info(f"已成功預加載 {len(all_market_data)} 個市場的 H4 數據。")

        symbol_groups = defaultdict(dict)
        for file_path in input_files:
            try:
                parts = file_path.stem.split('_'); symbol = '_'.join(parts[:-1]); timeframe = parts[-1]
                if timeframe in self.config.TIMEFRAME_ORDER: symbol_groups[symbol][timeframe] = file_path
            except IndexError: self.logger.warning(f"無法解析檔名: {file_path.name}，已跳過。")
        
        self.logger.info(f"共找到 {len(symbol_groups)} 個商品組需要處理。")
        for symbol, paths in symbol_groups.items():
            self.process_symbol_group(symbol, paths, events_df, factors_df, all_market_data)
            
        self.logger.info("========= 所有檔案處理完畢，綜合特徵工程流程結束 =========")

if __name__ == "__main__":
    try:
        config = Config()
        engineer = FeatureEngineer(config)
        engineer.run()
    except Exception as e:
        logging.critical(f"特徵工程腳本執行時發生未預期的嚴重錯誤: {e}", exc_info=True)
        sys.exit(1)
